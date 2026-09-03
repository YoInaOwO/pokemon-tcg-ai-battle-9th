#!/usr/bin/env bash
# v6 training pipeline (remote training box, RTX 5090).
#
# Stage 0  card table v3 (semantic schema columns)
# Stage 1  all-sides v6 extraction (every side of every episode)
# Stage 2  shared base BC (all decks -> one generalist model)
# Stage 3  per-archetype fine-tunes hot-started from the base
# Stage 4  acceptance: bc_league + eval_gauntlet commands (printed, not run)
#
# Usage:
#   nohup bash tools/train_v6_pipeline.sh > logs/v6_pipeline.log 2>&1 &
#   ARCHS="ogerpon_hydrapple box" bash tools/train_v6_pipeline.sh
#   STAGE=2 bash tools/train_v6_pipeline.sh         # resume from a stage
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

STAGE="${STAGE:-0}"
DAYS="${DAYS:-}"                       # e.g. "0805,0806,0807,0808,0809"; empty = all zips
WORKERS="${WORKERS:-20}"
CARDS=data/cards_v3.npz
BC_ALL=data/bc_all_v6
BC_ARCH=data/bc_arch_v6
BASE_OUT=runs/bc_base_v6
# archetypes to fine-tune from the shared base. Names MUST match
# ARCHETYPE_RULES in tools/replay_loader.py (what extract_bc --arch filters
# on). Default = every anchor archetype in the opponent pool / league evals;
# mega_venusaur / mega_froslass are rule-known but too rare online to anchor.
ARCHS="${ARCHS:-ogerpon_hydrapple ogerpon mega_lopunny mega_lucario marnie_grimmsnarl cynthia ns_zoroark alakazam dragapult kangaskhan_box kangaskhan_crustle grookey_dipplin}"

log() { echo "=== [$(date +%H:%M:%S)] $*" >&2; }

if [ "$STAGE" -le 0 ]; then
  log "stage 0: cards_v3.npz"
  python tools/dump_card_texts.py
  python tools/annotate_schema.py
  python -m ptcg_rl.cards --out "$CARDS" --schema data/card_skill_schema.json
fi

if [ "$STAGE" -le 1 ]; then
  log "stage 1: all-sides v6 extraction"
  python -m ptcg_rl.extract_bc --all --v6 --cards "$CARDS" \
      --out-dir "$BC_ALL" --workers "$WORKERS" ${DAYS:+--days "$DAYS"}
fi

if [ "$STAGE" -le 2 ]; then
  log "stage 2: shared base BC (all decks)"
  python -m ptcg_rl.train_bc --data "$BC_ALL/*_all.npz" --cards "$CARDS" \
      --arch --amp --epochs 12 --patience 4 --bs 1024 --lr 3e-4 \
      --val-frac 0.05 --out "$BASE_OUT"
fi

if [ "$STAGE" -le 3 ]; then
  for ARCH in $ARCHS; do
    log "stage 3: fine-tune $ARCH from base"
    python -m ptcg_rl.extract_bc --arch "$ARCH" --v6 --cards "$CARDS" \
        --out-dir "$BC_ARCH" --workers "$WORKERS" ${DAYS:+--days "$DAYS"}
    # hot start + gentle lr: keep the base's general competence, adapt the
    # priors to this archetype's lines
    python -m ptcg_rl.train_bc --data "$BC_ARCH/*_${ARCH}.npz" --cards "$CARDS" \
        --arch --amp --epochs 20 --patience 5 --bs 1024 --lr 8e-5 \
        --val-frac 0.1 --val-unseen-opp \
        --init "$BASE_OUT/model.pt" --out "runs/bc_${ARCH}_v6"
  done
fi

log "stage 4: acceptance (run manually)"
cat <<'EOF'
# league of new anchors vs old (edit deck list to taste):
#   python -m ptcg_rl.bc_league --games 200 ...
# fixed-seed gauntlet for a single anchor:
#   python -m ptcg_rl.eval_gauntlet --ckpt runs/bc_<arch>_v6/model.pt \
#       --cards data/cards_v3.npz --games 200
# optional short PPO polish per anchor (v6 nets carry their widths in config):
#   python -m ptcg_rl.selfplay_ppo --init runs/bc_<arch>_v6/model.pt \
#       --out runs/ppo_<arch>_v6 --deck <deck.csv> --cards data/cards_v3.npz \
#       --opp-config configs/opponents_default.json --updates 150
# package (parity + self-test run automatically):
#   python -m ptcg_rl.make_submission --ckpt runs/ppo_<arch>_v6/best.pt \
#       --deck <deck.csv> --cards data/cards_v3.npz
EOF
log "pipeline complete"
