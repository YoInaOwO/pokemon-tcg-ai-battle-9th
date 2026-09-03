#!/usr/bin/env bash
# ogerpon_hydrapple__15038d94: rollout-scaling limit probe (8388608 = 16x the
# 524288 recipe). Reuses the _rb pool and value ckpt from run_ppo_hydra_15038_vt.
#
# Expectations at 150 actors / 8 GPUs:
#   collect ~45-50 min/update (~120k games), train ~13 min -> ~1 h/update.
#   60 updates ~= 2.5 days and matches the 800x524288 run's total decisions.
#   rank0 host RAM peak ~150-200 GB (rollout buffer + one shard in flight).
#
# Usage (remote):
#   nohup bash tools/run_ppo_hydra_15038_r8m.sh > logs/run_hydra_15038_r8m.log 2>&1 &
#   tail -f logs/ppo_hydra_15038_r8m.log
#
# Env overrides: WORKERS, NPROC, MINIBATCH, PPO_UPDATES
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs build

DECK=data/decks/ogerpon_hydrapple_15038d94.csv
CARDS=data/cards_v3.npz
OPP=configs/opponents_env_0812_mut_rb.json
VALUE_OUT=runs/value_hydra_15038_rb
PPO_OUT=runs/ppo_hydra_15038_r8m
LOG_PPO=logs/ppo_hydra_15038_r8m.log
WORKERS="${WORKERS:-150}"
NPROC="${NPROC:-8}"
MINIBATCH="${MINIBATCH:-4096}"
PPO_UPDATES="${PPO_UPDATES:-60}"
ROLLOUT=8388608

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -f "$DECK" ]] || die "missing $DECK"
[[ -f "$CARDS" ]] || die "missing $CARDS"
[[ -f "$OPP" ]] || die "missing $OPP (run run_ppo_hydra_15038_vt.sh pool step first)"
[[ -f "$VALUE_OUT/model.pt" ]] || die "missing $VALUE_OUT/model.pt (value tune from the vt run)"

echo "=== PPO rollout=$ROLLOUT updates=$PPO_UPDATES nproc=$NPROC → $PPO_OUT ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "$NPROC" -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nproc_per_node="$NPROC" -m ptcg_rl.selfplay_ppo)
else
  LAUNCH=(python -m ptcg_rl.selfplay_ppo)
fi
# anneal horizons match the vt recipe in decisions: 120|130 x 524288 ~= 8 x 8388608
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
  --prize-shaping 0.05 --shape-anneal 8 \
  --kl-bc 0.05 --kl-bc-anneal 8 \
  --adapt-opp 0 \
  --device cuda \
  2>&1 | tee "$LOG_PPO"
