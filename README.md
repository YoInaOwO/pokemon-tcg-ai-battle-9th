# Lookahead as Input, Not Search

Rank 9 of 6,807 in the [Pokémon Trading Card Game AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
(Simulation track), from a 7.8M-parameter policy that never searches at play time.

The write-up is in [`reports/writeup_en.md`](reports/writeup_en.md)
(Chinese: [`reports/writeup_zh.md`](reports/writeup_zh.md)).

**Please read [`NOTICE.md`](NOTICE.md) first.** The competition engine, the card
database and the replay data are not in this repository and may not be
redistributed, so nothing here runs standalone.

## The idea

Every legal option is executed once inside the engine's search sandbox and its
public consequences are appended to that option's feature vector as 28 numbers:
damage, KO, prizes won and lost, hand and deck deltas, turn handover, win or
loss, a macro expansion over forced chains and coin flips, and a legality delta
counting how many attacks become legal afterwards. The network therefore reads
what a card *does* instead of memorising what its id means.

## Layout

| Path | What it is |
|---|---|
| `ptcg_rl/features.py` | observation encoding: 64 tokens, per-option features |
| `ptcg_rl/fwd_features.py` | the engine-lookahead probe (the 28 columns) |
| `ptcg_rl/model.py` | the policy: CardRepr, transformer trunk, option scoring, two critics |
| `ptcg_rl/np_model.py` | torch-free NumPy inference, used by the submitted bundle |
| `ptcg_rl/train_bc.py` | stage 1, behavioural cloning |
| `ptcg_rl/train_value.py` | stage 2, value fine-tune with a frozen trunk |
| `ptcg_rl/selfplay_ppo.py` | stage 3, PPO against the opponent pool |
| `ptcg_rl/mcts.py`, `mcts_agent.py` | root-parallel determinised MCTS (tried, did not help) |
| `ptcg_rl/make_submission.py` | bundle builder plus a NumPy/torch parity check |
| `tools/make_env_pool.py` | rebuilds the arena as an opponent pool from a replay dump |
| `tools/mutate_decks.py` | in-archetype decklist mutations, engine-validated |
| `tools/train_v6_pipeline.sh` | the full BC pipeline |
| `tools/run_ppo_hydra_15038_*.sh` | the two PPO runs the write-up compares |
| `analysis/` | post-competition ladder analysis and the write-up figures |

## What decided the result

Rollout size. Raising the decisions per update from 131k to 524k lifted the
whole learning curve, so the final run used 8.39M. It was slower per sample
early on and finished much higher. Section 4 of the write-up works through why
per-sample efficiency falls while the ceiling rises.

Search at play time made the agent worse, twice over: the time budget only
allows a few dozen simulations per tree, and determinised search plans a
different line for each sampled opponent hand while the agent has to commit to
one move for all of them.
