# Turning Lookahead into Features for PPO

Ninth place in the [Pokémon Trading Card Game AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
(Simulation track) with a 7.8M-parameter policy, without MCTS at play time.

The write-up is in [`reports/writeup_en.md`](reports/writeup_en.md)
(Chinese: [`reports/writeup_zh.md`](reports/writeup_zh.md);
Kaggle paste version: [`reports/writeup_kaggle_paste.txt`](reports/writeup_kaggle_paste.txt)).

**Please read [`NOTICE.md`](NOTICE.md) first.** The competition engine, the card
database and the replay data are not in this repository and may not be
redistributed, so nothing here runs standalone.

## The idea

For eligible main-phase, single-selection decisions, the engine probes up to
64 candidate actions and encodes their immediate consequences as 28 features.
These include damage, knockouts, Prize and resource changes, and newly legal
attacks. The probe follows forced continuations and coin-flip branches within
a bounded expansion; it does not read the opponent's actual hidden hand.

Each candidate uses cross-attention to read the encoded game state, and a
shared network scores it using both state and action features. Training proceeds
through behavioural cloning, value fine-tune and PPO against an arena-based
opponent pool. Both final submissions used the same Ogerpon–Hydrapple decklist
and model weights.

![Policy network](reports/model_redrawn.png)

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
| `tools/run_ppo_hydra_15038_*.sh` | saved PPO training configurations |
| `analysis/` | post-competition ladder analysis and the write-up figures |
| `analysis/draw_model.py` | generates the current model diagram as PNG, SVG and PDF |

## Training and results

Increasing decisions per update from 131k to 524k raised the training win-rate
plateau, prompting a direct jump to 8.39M. The final run used eight RTX 4090
GPUs and completed 57 updates in 56.3 hours. Its training score peaked at
83.3%; the report explains the opponent pool, metric and rollout tradeoffs.

Across 2,000 real Kaggle games, the final submissions scored 57.3%, counting
draws as half a win. The 1,800 games with a pre-game rating gap of at most 200
scored 53.5%. See the report for matchup results and how this sample differs
from the cloned training opponents.

Determinized MCTS and a larger network did not provide useful improvements.
The submitted policy retained the bounded engine probes as input features.
