"""BC policy network (feature version 4).

Token layout (58 tokens, d_model each, fixed positions -> learned pos emb):
    [global, select-ctx, stadium, board x12, hand x30, looking x8,
     discard-bag x2, log-summary, game-memory, deck-bag]
Options are encoded, cross-attend to the encoded state tokens (so a "target
bench slot 3" option can read that slot's live HP/energy), then are scored
against the pooled state: logit_i = MLP([opt_i, h, opt_i * h]).

Heads: per-option policy logits, selection-count distribution (0..MAX_COUNT-1)
fed by [pooled state, masked mean of option reprs], blind value, and a train-only
oracle value head (asymmetric critic: sees the opponent-hand snapshot via a
side channel that never enters the encoder, so policy/blind-value stay clean
for inference and MCTS).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .features import (BOARD_F, CTX_F, GLOB_F, LOG_F, MAX_COUNT, MAX_HAND,
                       MAX_LOOK, MEM_F, N_MEM_ATK, OPT_F, ORA_F)

D_CARD = 64
D_ATK = 32
N_TOKENS = 3 + 12 + MAX_HAND + MAX_LOOK + 2 + 3  # 58


def _fmin(t: torch.Tensor) -> float:
    """Dtype-safe fill for masked logits: -1e9 overflows fp16 under AMP
    (newer torch raises), -inf turns fully-masked rows into NaN."""
    return torch.finfo(t.dtype).min


class CardRepr(nn.Module):
    """id embedding + projected static features (shared by every card slot).

    Row n_cards is a dedicated UNK: card ids beyond the frozen table (new
    sets released mid-competition) map there instead of crashing/aliasing.

    id_dropout (training only): zero the id-embedding half per slot with
    probability p while keeping the projected static attributes. Cards seen
    rarely (or never) in training have uninformative embeddings at play time;
    the net must stay functional on static attributes alone."""

    def __init__(self, card_feats: torch.Tensor):
        super().__init__()
        n_cards, f = card_feats.shape
        self.n_real = n_cards
        self.id_dropout = 0.0
        self.register_buffer("static",
                             torch.cat([card_feats, torch.zeros(1, f)], 0),
                             persistent=False)
        self.emb = nn.Embedding(n_cards + 1, D_CARD)
        self.proj = nn.Linear(f, D_CARD)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.clamp(min=0, max=self.n_real)
        e = self.emb(ids)
        if self.training and self.id_dropout > 0.0:
            keep = (torch.rand(ids.shape, device=ids.device)
                    >= self.id_dropout).unsqueeze(-1)
            e = e * keep
        return e + self.proj(self.static[ids])


class AttackRepr(nn.Module):
    def __init__(self, attack_feats: torch.Tensor):
        super().__init__()
        n_atk, f = attack_feats.shape
        self.n_real = n_atk
        self.id_dropout = 0.0
        self.register_buffer("static",
                             torch.cat([attack_feats, torch.zeros(1, f)], 0),
                             persistent=False)
        self.emb = nn.Embedding(n_atk + 1, D_ATK)
        self.proj = nn.Linear(f, D_ATK)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.clamp(min=0, max=self.n_real)
        e = self.emb(ids)
        if self.training and self.id_dropout > 0.0:
            keep = (torch.rand(ids.shape, device=ids.device)
                    >= self.id_dropout).unsqueeze(-1)
            e = e * keep
        return e + self.proj(self.static[ids])


class PolicyNet(nn.Module):
    def __init__(self, card_feats, attack_feats, d_model: int = 256,
                 n_layers: int = 3, n_heads: int = 4, ff: int | None = None,
                 dropout: float = 0.0, arch_f: int = 0, opt_f: int = OPT_F,
                 aux: bool = False, n_board: int = 12,
                 count_classes: int = MAX_COUNT):
        super().__init__()
        card_feats = torch.as_tensor(card_feats, dtype=torch.float32)
        attack_feats = torch.as_tensor(attack_feats, dtype=torch.float32)
        self.card = CardRepr(card_feats)
        self.attack = AttackRepr(attack_feats)
        self.arch_f = arch_f  # opponent-model feature width (0 = disabled)
        # option feature width this net was trained with; batches encoded by a
        # newer schema are sliced to this prefix (columns are append-only)
        self.opt_f = opt_f
        # board token count / count-head classes are feature-version bound
        # (v5: 12/22, v6: 18/24); both come from the checkpoint config
        self.n_board = n_board
        self.count_classes = count_classes
        d = d_model
        ff = ff or 2 * d_model

        self.glob_in = nn.Linear(GLOB_F + arch_f, d)
        self.ctx_in = nn.Linear(CTX_F + 2 * D_CARD, d)
        self.stadium_in = nn.Linear(D_CARD, d)
        self.board_in = nn.Linear(BOARD_F + 4 * D_CARD, d)
        self.hand_in = nn.Linear(D_CARD, d)
        self.look_in = nn.Linear(D_CARD + 2, d)
        self.disc_in = nn.Linear(D_CARD + 1, d)
        self.logs_in = nn.Linear(LOG_F + 2 * D_ATK + D_CARD, d)
        self.mem_in = nn.Linear(MEM_F + 2 * D_ATK + D_CARD, d)
        self.deck_in = nn.Linear(D_CARD + 1, d)
        # token types: 0 glob, 1 ctx, 2 stadium, 3 board, 4 hand, 5 look,
        #              6 disc, 7 logs, 8 memory, 9 deck-bag
        self.type_emb = nn.Embedding(10, d)
        self.pos_emb = nn.Embedding(3 + n_board + MAX_HAND + MAX_LOOK + 2 + 3, d)

        layer = nn.TransformerEncoderLayer(d, n_heads, ff, dropout=dropout,
                                           batch_first=True, norm_first=True)
        # nested-tensor fast path doesn't support norm_first; disable to
        # silence the per-process UserWarning (no behavior change)
        self.encoder = nn.TransformerEncoder(layer, n_layers,
                                             enable_nested_tensor=False)

        self.opt_in = nn.Linear(self.opt_f + 2 * D_CARD + D_ATK, d)
        self.opt_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.opt_ln = nn.LayerNorm(d)
        self.score = nn.Sequential(nn.Linear(3 * d, 256), nn.ReLU(), nn.Linear(256, 1))
        self.count_head = nn.Linear(2 * d, count_classes)
        self.value_head = nn.Linear(d, 1)
        # asymmetric critic (train only): pooled state + opponent-hand oracle
        self.oracle_in = nn.Linear(D_CARD + ORA_F, d)
        self.value_oracle = nn.Sequential(nn.Linear(2 * d, 256), nn.ReLU(),
                                          nn.Linear(256, 1))
        # auxiliary supervision head (train only, ignored at inference):
        # prizes taken by [me, opp] within the next 2 and 4 turns -- injects
        # a medium-horizon planning signal into the shared representation
        self.aux_head = (nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                                       nn.Linear(64, 4)) if aux else None)
        self._aux = None
        # BC: stop the oracle value loss from shaping the shared encoder (its
        # gradient teaches the policy trunk to expect information it will
        # never have at play time). PPO keeps it False: there the oracle IS
        # the training critic and must shape the representation.
        self.detach_oracle = False

    def encode_state(self, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """b: batch dict of tensors -> (encoded tokens (B,T,d), padding mask (B,T))."""
        B = b["glob"].shape[0]
        dev = b["glob"].device
        toks = []
        types = []
        # arch is sliced like opt_feats: v5 appends the unknown bit at the end
        g = (torch.cat([b["glob"], b["arch"][..., :self.arch_f]], -1)
             if self.arch_f else b["glob"])
        toks.append(self.glob_in(g).unsqueeze(1)); types.append(torch.zeros(B, 1, dtype=torch.long, device=dev))
        ctx_cards = self.card(b["ctx_ids"].long()).flatten(1)          # (B, 2*D_CARD)
        toks.append(self.ctx_in(torch.cat([b["ctx"], ctx_cards], -1)).unsqueeze(1))
        types.append(torch.full((B, 1), 1, dtype=torch.long, device=dev))
        toks.append(self.stadium_in(self.card(b["stadium_id"].long())).unsqueeze(1))
        types.append(torch.full((B, 1), 2, dtype=torch.long, device=dev))
        nb = b["board_ids"].shape[1]                     # 12 (v5) or 18 (v6)
        board_cards = self.card(b["board_ids"].long()).flatten(2)      # (B,nb,4*D_CARD)
        toks.append(self.board_in(torch.cat([b["board"], board_cards], -1)))
        types.append(torch.full((B, nb), 3, dtype=torch.long, device=dev))
        toks.append(self.hand_in(self.card(b["hand_ids"].long())))
        types.append(torch.full((B, b["hand_ids"].shape[1]), 4, dtype=torch.long, device=dev))
        toks.append(self.look_in(torch.cat([self.card(b["look_ids"].long()),
                                            b["look_feats"]], -1)))
        types.append(torch.full((B, b["look_ids"].shape[1]), 5, dtype=torch.long, device=dev))
        disc = self.card(b["disc_ids"].long())                          # (B,2,60,D_CARD)
        mask = (b["disc_ids"] > 0).unsqueeze(-1).float()
        cnt = mask.sum(2).clamp(min=1.0)
        disc_mean = (disc * mask).sum(2) / cnt                          # (B,2,D_CARD)
        toks.append(self.disc_in(torch.cat([disc_mean, cnt / 60.0], -1)))
        types.append(torch.full((B, 2), 6, dtype=torch.long, device=dev))
        my_atk = self.attack(b["logs_ids"][:, 0].long())
        op_atk = self.attack(b["logs_ids"][:, 1].long())
        rev = self.card(b["logs_ids"][:, 2:].long())                    # (B,4,D_CARD)
        rmask = (b["logs_ids"][:, 2:] > 0).unsqueeze(-1).float()
        rev_mean = (rev * rmask).sum(1) / rmask.sum(1).clamp(min=1.0)
        toks.append(self.logs_in(torch.cat([b["logs"], my_atk, op_atk, rev_mean],
                                           -1)).unsqueeze(1))
        types.append(torch.full((B, 1), 7, dtype=torch.long, device=dev))

        def _bag(ids_2d):  # (B,N) int -> masked mean card emb (B,D_CARD)
            e = self.card(ids_2d.long())
            m = (ids_2d > 0).unsqueeze(-1).float()
            return (e * m).sum(1) / m.sum(1).clamp(min=1.0)

        # attack-history embeddings (attack table), known-opp-hand bag (card table)
        na = N_MEM_ATK
        am = (b["mem_ids"][:, :na] > 0).unsqueeze(-1).float()
        ae = self.attack(b["mem_ids"][:, :na].long())
        mem_my = (ae * am).sum(1) / am.sum(1).clamp(min=1.0)
        om = (b["mem_ids"][:, na:2 * na] > 0).unsqueeze(-1).float()
        oe = self.attack(b["mem_ids"][:, na:2 * na].long())
        mem_op = (oe * om).sum(1) / om.sum(1).clamp(min=1.0)
        mem_rev = _bag(b["mem_ids"][:, 2 * na:])
        toks.append(self.mem_in(torch.cat([b["mem"], mem_my, mem_op, mem_rev],
                                          -1)).unsqueeze(1))
        types.append(torch.full((B, 1), 8, dtype=torch.long, device=dev))

        dk_cnt = (b["deck_ids"] > 0).sum(1, keepdim=True).float()
        toks.append(self.deck_in(torch.cat([_bag(b["deck_ids"]), dk_cnt / 60.0],
                                           -1)).unsqueeze(1))
        types.append(torch.full((B, 1), 9, dtype=torch.long, device=dev))

        x = torch.cat(toks, 1) + self.type_emb(torch.cat(types, 1))
        x = x + self.pos_emb(torch.arange(x.shape[1], device=dev)).unsqueeze(0)
        # padding mask: empty hand slots / absent looking slots
        pad = torch.zeros(B, x.shape[1], dtype=torch.bool, device=dev)
        h0 = 3 + nb
        pad[:, h0:h0 + b["hand_ids"].shape[1]] = b["hand_ids"] == 0
        l0 = h0 + b["hand_ids"].shape[1]
        pad[:, l0:l0 + b["look_ids"].shape[1]] = b["look_feats"][..., 0] == 0
        x = self.encoder(x, src_key_padding_mask=pad)
        return x, pad

    def option_repr(self, b: dict, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """(B, K, d): option embeddings cross-attending to the state tokens."""
        oc = self.card(b["opt_card"].long())
        ot = self.card(b["opt_tgt"].long())
        oa = self.attack(b["opt_atk"].long())
        of = b["opt_feats"]
        if of.shape[-1] != self.opt_f:  # newer encoder: slice appended columns
            of = of[..., :self.opt_f]
        q = self.opt_in(torch.cat([of, oc, ot, oa], -1))
        ca, _ = self.opt_attn(q, x, x, key_padding_mask=pad, need_weights=False)
        return self.opt_ln(q + ca)

    def forward(self, b: dict):
        """-> (logits (B,K) with dtype-min on padding, count_logits (B,MAX_COUNT), value (B,),
        oracle value (B,) or None when the oracle side-channel is absent)."""
        x, pad = self.encode_state(b)
        h = x[:, 0]
        self._aux = self.aux_head(h) if self.aux_head is not None else None
        opt = self.option_repr(b, x, pad)                               # (B,K,d)
        hk = h.unsqueeze(1).expand_as(opt)
        logits = self.score(torch.cat([opt, hk, opt * hk], -1)).squeeze(-1)
        # finite mask value: -inf poisons gradients through p*log p terms,
        # and a fixed -1e9 overflows fp16 under AMP
        logits = logits.masked_fill(~b["opt_mask"], _fmin(logits))
        omask = b["opt_mask"].unsqueeze(-1).float()
        opt_mean = (opt * omask).sum(1) / omask.sum(1).clamp(min=1.0)
        count_logits = self.count_head(torch.cat([h, opt_mean], -1))
        v_ora = None
        if "ora" in b:
            hh = h.detach() if self.detach_oracle else h
            oe = self.card(b["ora_ids"].long())
            omk = (b["ora_ids"] > 0).unsqueeze(-1).float()
            obag = (oe * omk).sum(1) / omk.sum(1).clamp(min=1.0)
            orep = self.oracle_in(torch.cat([obag, b["ora"]], -1))
            v_ora = self.value_oracle(torch.cat([hh, orep], -1)).squeeze(-1)
        return logits, count_logits, self.value_head(h).squeeze(-1), v_ora

    # ------------------------------------------------------------------ PPO
    # Joint action = (k, ordered picks without replacement):
    #   log p = log p(k | count head, masked to [min,max])        (if min != max)
    #         + sum_t log softmax(logits masked by previous picks)[a_t]

    def _count_mask(self, count_logits, min_c, max_c, n_opt):
        """(B,10) bool mask of allowed selection counts."""
        B, C = count_logits.shape
        ar = torch.arange(C, device=count_logits.device).unsqueeze(0)
        hi = torch.minimum(max_c, torch.minimum(n_opt, torch.full_like(max_c, C - 1)))
        lo = torch.minimum(min_c, hi)
        return (ar >= lo.unsqueeze(1)) & (ar <= hi.unsqueeze(1))

    def action_logprob_entropy(self, b: dict, seq: torch.Tensor, kk: torch.Tensor):
        """seq (B, S) option indices padded with -1; kk (B,) selection counts.

        -> (logp (B,), entropy (B,), value (B,), oracle value or None,
        logits (B,K), count_logits (B,MAX_COUNT)). Differentiable.
        """
        logits, count_logits, value, v_ora = self.forward(b)
        B, K = logits.shape
        dev = logits.device
        lp = torch.zeros(B, device=dev)
        ent = torch.zeros(B, device=dev)

        n_opt = b["opt_mask"].sum(1).long()
        choice = b["min_c"] != b["max_c"]
        cmask = self._count_mask(count_logits, b["min_c"], b["max_c"], n_opt)
        cl = count_logits.masked_fill(~cmask, _fmin(count_logits))
        clog = torch.log_softmax(cl, 1)
        cp = clog.exp()
        c_ent = -(torch.where(cmask, cp * clog, torch.zeros_like(cp))).sum(1)
        idx = kk.clamp(0, clog.shape[1] - 1).unsqueeze(1)
        lp = lp + torch.where(choice, clog.gather(1, idx).squeeze(1), torch.zeros(B, device=dev))
        ent = ent + torch.where(choice, c_ent, torch.zeros(B, device=dev))

        mask = b["opt_mask"].clone()
        S = seq.shape[1]
        for t in range(S):
            active = kk > t
            if not active.any():
                break
            step = logits.masked_fill(~mask, _fmin(logits))
            ls = torch.log_softmax(step, 1)
            a = seq[:, t].clamp(min=0).unsqueeze(1)
            lp = lp + torch.where(active, ls.gather(1, a).squeeze(1), torch.zeros(B, device=dev))
            p = ls.exp()
            s_ent = -(torch.where(mask, p * ls, torch.zeros_like(p))).sum(1)
            ent = ent + torch.where(active, s_ent, torch.zeros(B, device=dev))
            mask = mask & ~(torch.nn.functional.one_hot(a.squeeze(1), K).bool()
                            & active.unsqueeze(1))
        return lp, ent, value, v_ora, logits, count_logits

    @torch.no_grad()
    def sample_action(self, b: dict, min_c: int, max_c: int, greedy: bool = False):
        """Single-obs batch (B=1).

        -> (indices list, logp float, blind value float, oracle value float —
        falls back to the blind value when the oracle side-channel is absent).
        """
        logits, count_logits, value, v_ora = self.forward(b)
        logits = logits[0]
        K = logits.shape[0]
        lp = 0.0
        hi = min(max_c, count_logits.shape[1] - 1, K)
        lo = min(min_c, hi)
        if min_c == max_c:
            k = min(min_c, K)
        else:
            cl = count_logits[0]
            cmask = torch.zeros(count_logits.shape[1], dtype=torch.bool, device=cl.device)
            cmask[lo:hi + 1] = True
            clog = torch.log_softmax(cl.masked_fill(~cmask, _fmin(cl)), 0)
            k = int(clog.argmax()) if greedy else int(torch.multinomial(clog.exp(), 1))
            lp += float(clog[k])
        seq: list[int] = []
        mask = torch.zeros(K, dtype=torch.bool, device=logits.device)
        for _ in range(k):
            ls = torch.log_softmax(logits.masked_fill(mask, _fmin(logits)), 0)
            a = int(ls.argmax()) if greedy else int(torch.multinomial(ls.exp(), 1))
            lp += float(ls[a])
            seq.append(a)
            mask[a] = True
        vo = float(v_ora[0]) if v_ora is not None else float(value[0])
        return seq, lp, float(value[0]), vo
