#!/usr/bin/env bash
# ogerpon_hydrapple__15038d94: build mut pool → value tune → PPO (524288 rollout).
# 0812-weighted arena pool; sized for the 8-GPU / big-CPU box (150 actors).
#
# Usage (remote):
#   nohup bash tools/run_ppo_hydra_15038_vt.sh > logs/run_hydra_15038_vt.log 2>&1 &
#   tail -f logs/run_hydra_15038_vt.log
#
# Requires: configs/opponents_env_0812.json, runs/bc_ogerpon_hydrapple_v6/model.pt,
#           data/decks/ogerpon_hydrapple_15038d94.csv, data/cards_v3.npz
#
# Env overrides:
#   SKIP_POOL=1      skip mut-pool generation
#   SKIP_VALUE=1     skip gen_value_data + train_value (use existing value ckpt)
#   VALUE_GAMES=12000
#   PPO_UPDATES=800  PPO rollout fixed at 524288 (4x the 131072 recipe;
#                    800 updates ~= 3200 updates' worth of decisions)
#   NPROC=8          learner GPUs (torchrun data-parallel; 1 = plain python)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs build

DECK=data/decks/ogerpon_hydrapple_15038d94.csv
BC=runs/bc_ogerpon_hydrapple_v6/model.pt
CARDS=data/cards_v3.npz
BASE_OPP=configs/opponents_env_0812.json
MUT_ONLY=configs/opponents_mut_only_0812.json
OPP=configs/opponents_env_0812_mut_rb.json
VALUE_DATA=build/value_data_hydra_15038_rb   # _rb: regenerated on the pool with rule agents
VALUE_OUT=runs/value_hydra_15038_rb
PPO_OUT=runs/ppo_hydra_15038_vt
LOG_PPO=logs/ppo_hydra_15038_vt.log
WORKERS="${WORKERS:-150}"
NPROC="${NPROC:-8}"
MINIBATCH="${MINIBATCH:-4096}"   # per-GPU; 8192 OOMs a 48G card on big-K batches.
                                 # effective step batch = NPROC x MINIBATCH
VALUE_GAMES="${VALUE_GAMES:-12000}"
VALUE_EPOCHS="${VALUE_EPOCHS:-6}"
PPO_UPDATES="${PPO_UPDATES:-800}"
ROLLOUT=524288

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -f "$BC" ]] || die "missing $BC"
[[ -f "$DECK" ]] || die "missing $DECK"
[[ -f "$CARDS" ]] || die "missing $CARDS"
[[ -f "$BASE_OPP" ]] || die "missing $BASE_OPP"

if [[ "${SKIP_POOL:-0}" != "1" && ! -f "$OPP" ]]; then
  echo "=== [1/4] build $OPP (mirror 5% + mut 10% + script 10% + bc 75%) ==="
  python tools/mutate_decks.py --per-arch 2 --seed 1 \
    --budget 0.10 --out "$MUT_ONLY"
  export BASE_OPP MUT_ONLY OPP
  python - <<'PY'
import json, os
base = os.environ["BASE_OPP"]
mut_only = os.environ["MUT_ONLY"]
out = os.environ["OPP"]
bc = [dict(e) for e in json.load(open(base)) if e["kind"] == "bc"]
bc_mass = sum(e["weight"] for e in bc)
for e in bc:  # bc shrinks 85% -> 75% to make room for the rule-based agents
    e["weight"] = round(e["weight"] * 0.75 / bc_mass, 5)
mut = json.load(open(mut_only))
scripts = [
    {"kind": "script", "name": "rule_mega_lucario",
     "module": "ptcg_rl.rule_agents.rule_mega_lucario", "weight": 0.05},
    {"kind": "script", "name": "rule_dragapult",
     "module": "ptcg_rl.rule_agents.rule_dragapult", "weight": 0.03},
    {"kind": "script", "name": "rule_iono",
     "module": "ptcg_rl.rule_agents.rule_iono", "weight": 0.01},
    {"kind": "script", "name": "rule_mega_abomasnow",
     "module": "ptcg_rl.rule_agents.rule_mega_abomasnow", "weight": 0.01},
]
cfg = [{"kind": "mirror", "weight": 0.05}] + mut + scripts + bc
json.dump(cfg, open(out, "w"), indent=1)
print(f"written {out}: mirror 0.05 | mut {sum(e['weight'] for e in mut):.3f} "
      f"| script {sum(e['weight'] for e in scripts):.3f} "
      f"| bc {sum(e['weight'] for e in bc):.3f} | entries {len(cfg)}")
PY
else
  echo "=== [1/4] skip pool (SKIP_POOL=1 or $OPP exists) ==="
fi
[[ -f "$OPP" ]] || die "missing $OPP after pool step"

if [[ "${SKIP_VALUE:-0}" != "1" && ! -f "$VALUE_OUT/model.pt" ]]; then
  echo "=== [2/4] gen value data ($VALUE_GAMES games) ==="
  python -m ptcg_rl.gen_value_data \
    --ckpt "$BC" --deck "$DECK" --opp-config "$OPP" --cards "$CARDS" \
    --games "$VALUE_GAMES" --workers "$WORKERS" --out "$VALUE_DATA" \
    2>&1 | tee logs/gen_value_hydra_15038.log

  echo "=== [3/4] train value head ($VALUE_EPOCHS epochs) ==="
  python -m ptcg_rl.train_value \
    --init "$BC" --data "${VALUE_DATA}/*.npz" --cards "$CARDS" \
    --out "$VALUE_OUT" --epochs "$VALUE_EPOCHS" \
    2>&1 | tee logs/train_value_hydra_15038.log
else
  echo "=== [2-3/4] skip value tune (SKIP_VALUE=1 or $VALUE_OUT/model.pt exists) ==="
fi
[[ -f "$VALUE_OUT/model.pt" ]] || die "missing $VALUE_OUT/model.pt"

echo "=== [4/4] PPO rollout=$ROLLOUT updates=$PPO_UPDATES nproc=$NPROC → $PPO_OUT ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "$NPROC" -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nproc_per_node="$NPROC" -m ptcg_rl.selfplay_ppo)
else
  LAUNCH=(python -m ptcg_rl.selfplay_ppo)
fi
exec "${LAUNCH[@]}" \
  --init "$VALUE_OUT/model.pt" \
  --deck "$DECK" \
  --cards "$CARDS" \
  --opp-config "$OPP" \
  --out "$PPO_OUT" \
  --actors "$WORKERS" \
  --rollout "$ROLLOUT" \
  --minibatch "$MINIBATCH" \
  --updates "$PPO_UPDATES" \
  --prize-shaping 0.05 --shape-anneal 120 \
  --kl-bc 0.05 --kl-bc-anneal 130 \
  --adapt-opp 0 \
  --device cuda \
  2>&1 | tee "$LOG_PPO"
