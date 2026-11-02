#!/usr/bin/env bash
# Core experiment matrix, 1 GPU per run (RunPod or any multi-GPU box).
#
# Every run uses FA3 and a wall-clock-fraction schedule (momentum warmup 0.18, warmdown
# 0.36 of the 600 s budget), so it is correct on any GPU without retuning step counts.
#
# Usage, from the repo root:
#   bash records/track_non_record_16mb/2026-10-28_GDN_Hybrid_FA3_H100/run_core.sh
#
# Environment:
#   CONFIGS   space-separated subset of: baseline gdn_h8k42 gdn_h5k64 kda_h8k42 kda_h5k64
#             (default: all five)
#   SEEDS     space-separated; "default" = the run script's per-arm seed (default: "default 42")
#   SEQ_LENS  space-separated (default: 32768). 16384 adds the crossover check.
#   GPU_IDS   space-separated GPU indices to use (default: every visible GPU)
#   DRY_RUN=1 print the job plan and exit
#   MAX_WALLCLOCK_SECONDS  passed through (default 600 via the run scripts)
#
# Runs whose log already has a final score are skipped, so the script can be re-run
# after a pod restart. RUN_ID = core_<config>_<gpu>_seq<L>_s<seed>. Training logs: logs/<RUN_ID>.txt; stdout: logs/run_core/<RUN_ID>.out
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"
[[ -x .venv/bin/python ]] && export PATH="$REPO_ROOT/.venv/bin:$PATH"

CONFIGS="${CONFIGS:-baseline gdn_h8k42 gdn_h5k64 kda_h8k42 kda_h5k64}"
SEEDS="${SEEDS:-default 42}"
SEQ_LENS="${SEQ_LENS:-32768}"
GPU_IDS="${GPU_IDS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}"
read -r -a GPUS <<< "$GPU_IDS"
GPU_TAG="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | grep -oiE 'h100|h200|a100|b200|l40s' | head -1 | tr 'A-Z' 'a-z')"
GPU_TAG="${GPU_TAG:-gpu}"

# config -> arm and extra environment
config_arm() { [[ "$1" == baseline ]] && echo baseline || echo hybrid; }
config_env() {
  case "$1" in
    baseline)  echo "" ;;
    gdn_h8k42) echo "GDN_MIXER=gdn" ;;
    gdn_h5k64) echo "GDN_MIXER=gdn GDN_N_HEADS=5 GDN_HEAD_DIM_RATIO=0.72" ;;
    kda_h8k42) echo "GDN_MIXER=kda" ;;
    kda_h5k64) echo "GDN_MIXER=kda GDN_N_HEADS=5 GDN_HEAD_DIM_RATIO=0.72" ;;
    *) echo "unknown config: $1" >&2; exit 1 ;;
  esac
}
# Per-arm default seeds of the run scripts, so RUN_IDs name the real seed
default_seed() {
  local arm="$1" seq="$2"
  case "$arm:$seq" in
    baseline:8192) echo 1337 ;; baseline:16384) echo 1338 ;; baseline:32768) echo 1339 ;;
    hybrid:8192)   echo 1340 ;; hybrid:16384)   echo 1341 ;; hybrid:32768)   echo 1342 ;;
  esac
}

JOBS=()
for seq in $SEQ_LENS; do
  for cfg in $CONFIGS; do
    config_env "$cfg" > /dev/null || exit 1
    arm="$(config_arm "$cfg")"
    for s in $SEEDS; do
      [[ "$s" == default ]] && s="$(default_seed "$arm" "$seq")"
      JOBS+=("$cfg $arm $seq $s")
    done
  done
done

run_id_of() { echo "core_${1}_${GPU_TAG}_seq${3}_s${4}"; }

echo "GPUs: ${GPUS[*]} (${GPU_TAG}) | ${#JOBS[@]} jobs"
for i in "${!JOBS[@]}"; do
  read -r cfg arm seq s <<< "${JOBS[$i]}"
  echo "  gpu ${GPUS[$((i % ${#GPUS[@]}))]}: $(run_id_of $cfg $arm $seq $s)  [$arm $(config_env "$cfg")]"
done
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0

mkdir -p logs/run_core

run_job() {
  local gpu="$1" cfg="$2" arm="$3" seq="$4" seed="$5"
  local run_id; run_id="$(run_id_of "$cfg" "$arm" "$seq" "$seed")"
  if grep -qs "final_int8_zlib_roundtrip val_loss" "logs/$run_id.txt"; then
    echo "[gpu $gpu] skip $run_id (already has a final score)"; return 0
  fi
  rm -f "logs/$run_id.txt"   # a partial log from an interrupted run would be appended to
  echo "[gpu $gpu] start $run_id"
  # shellcheck disable=SC2046
  env $(config_env "$cfg") \
    CUDA_VISIBLE_DEVICES="$gpu" \
    ATTN_BACKEND=fa3 \
    WARMDOWN_FRAC=0.36 MUON_MOMENTUM_WARMUP_FRAC=0.18 \
    SEED="$seed" RUN_ID="$run_id" \
    TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${run_id}" \
    TRITON_CACHE_DIR="/tmp/triton_${run_id}" \
    bash "$HERE/run_${arm}_long_context_single_h100.sh" "$seq" > "logs/run_core/$run_id.out" 2>&1
  local rc=$?
  echo "[gpu $gpu] done  $run_id rc=$rc $(grep -h 'final_int8_zlib_roundtrip val_loss' "logs/$run_id.txt" 2>/dev/null | tail -1)"
}

# One worker per GPU, jobs assigned round-robin
for w in "${!GPUS[@]}"; do
  (
    for i in "${!JOBS[@]}"; do
      (( i % ${#GPUS[@]} == w )) || continue
      read -r cfg arm seq s <<< "${JOBS[$i]}"
      run_job "${GPUS[$w]}" "$cfg" "$arm" "$seq" "$s"
    done
  ) &
done
wait
echo "all jobs finished; summarize with:"
echo "  python $HERE/summarize.py logs"
