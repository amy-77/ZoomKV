#!/usr/bin/env bash
# nsys profile: CDS density pipeline — decode-only by default
# Usage:
#   CUDA_DEVICE=1 bash run/run_nsys_throughput_cds.sh            # decode-only (delay+duration)
#   CUDA_DEVICE=1 NSYS_FULL=1 bash run/run_nsys_throughput_cds.sh  # full trace
#   CUDA_DEVICE=1 BATCH_SIZE=6 FINAL_TOPK=500 bash run/...        # override params
set -euo pipefail

# Optional compatibility shim for environments that ship a GLIBC stub.
GLIBC232_STUB="${CONDA_PREFIX:-}/glibc232_stub.so"
if [ -f "$GLIBC232_STUB" ]; then
    case ":${LD_PRELOAD:-}:" in
        *":$GLIBC232_STUB:"*) ;;
        *) export LD_PRELOAD="$GLIBC232_STUB${LD_PRELOAD:+:$LD_PRELOAD}" ;;
    esac
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_DATASETS_CACHE="$ROOT/.cache/hf_datasets"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"

# Ensure Nsight Systems is in PATH on this host.
if ! command -v nsys >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nsys ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi

# ── CDS pipeline env ──
export POLAR_PURE_RABITQ="${POLAR_PURE_RABITQ:-0}"
export POLAR_CDS_ENABLED="${POLAR_CDS_ENABLED:-1}"
export POLAR_CDS_DENSE_RATIO="${POLAR_CDS_DENSE_RATIO:-0.1}"
export POLAR_CDS_SPARSE_TOPK="${POLAR_CDS_SPARSE_TOPK:-4}"

# ── Hierarchical Quest ──
export POLAR_QUEST_HIERARCHICAL="${POLAR_QUEST_HIERARCHICAL:-1}"
export POLAR_QUEST_CHUNK="${POLAR_QUEST_CHUNK:-16}"
export POLAR_QUEST_LARGE_CHUNK="${POLAR_QUEST_LARGE_CHUNK:-256}"
export POLAR_QUEST_LARGE_RATIO="${POLAR_QUEST_LARGE_RATIO:-0.25}"
export POLAR_QUEST_SMALL_RATIO="${POLAR_QUEST_SMALL_RATIO:-0.20}"

export POLAR_CDS_RECALL_CHECK=0

# ── polar_cache module (env-overridable) ──
POLAR_MODULE="${POLAR_CACHE_MODULE:-cache_hub.polar_cache_kivi_quest_density_seperate_kernel}"
export POLAR_CACHE_MODULE="$POLAR_MODULE"

# KIVI-only knobs (no-op when the RaBitQ module is selected).
export POLAR_KIVI_KERNEL="${POLAR_KIVI_KERNEL:-cuda}"
export POLAR_KIVI_BITS="${POLAR_KIVI_BITS:-4}"

# ── Tunable params ──
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MODEL="${MODEL:-$ROOT/models/Qwen3-4B-Thinking-2507}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-128000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
FINAL_TOPK="${FINAL_TOPK:-100}"

# CPU offload toggle (1=on, 0=off).  Both pipelines support offload; set
# the same value when benchmarking RaBitQ vs KIVI for an apples-to-apples
# comparison.
ENABLE_OFFLOAD="${ENABLE_OFFLOAD:-1}"
# argparse's `type=bool` is broken for "false" — only bool("") is False.
if [[ "$ENABLE_OFFLOAD" == "0" ]]; then
    OFFLOAD_ARG=""
else
    OFFLOAD_ARG="true"
fi

# ── nsys capture control ──
# Default: cudaProfilerApi (code calls cudaProfilerStart/Stop around decode loop)
# NSYS_FULL=1  → profile everything (no capture-range filtering)

TS=$(date +"%Y%m%d_%H%M%S")
NSYS_DIR="${NSYS_DIR:-$ROOT/results/nsys}"
mkdir -p "$NSYS_DIR" "${NSYS_DIR}/tmp"
export TMPDIR="${TMPDIR:-${NSYS_DIR}/tmp}"

MODULE_TAG="${POLAR_MODULE##*.}"
NSYS_OUT="${NSYS_DIR}/nsys_${MODULE_TAG}_bs${BATCH_SIZE}_top${FINAL_TOPK}_gpu${CUDA_DEVICE}_${TS}"
LOG="${NSYS_OUT}.log"

NSYS_FLUSH_INTERVAL="${NSYS_FLUSH_INTERVAL:-10000}"
NSYS_ARGS=(
    --output="$NSYS_OUT"
    --force-overwrite=true
    --sample=none
    --cpuctxsw=none
    --trace=cuda,nvtx
    --cuda-flush-interval="$NSYS_FLUSH_INTERVAL"
)

if [[ "${NSYS_FULL:-0}" == "0" ]]; then
    NSYS_ARGS+=(--capture-range=cudaProfilerApi --cudabacktrace=none)
    MODE="decode-only (cudaProfilerApi)"
else
    MODE="full trace"
fi

echo "==== nsys: ${MODE}  bs=${BATCH_SIZE}  top${FINAL_TOPK}  GPU=${CUDA_DEVICE} ====" | tee "$LOG"
echo "     module=${POLAR_MODULE}" | tee -a "$LOG"
echo "     kivi_kernel=${POLAR_KIVI_KERNEL}  kivi_bits=${POLAR_KIVI_BITS}" | tee -a "$LOG"
echo "     enable_offload=${ENABLE_OFFLOAD}" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
nsys profile "${NSYS_ARGS[@]}" \
    python -u "$ROOT/tests/test_throughput_longbench.py" \
        --model_type qwen \
        --model_name "$MODEL" \
        --batch_size "$BATCH_SIZE" \
        --max_input_tokens "$MAX_INPUT_TOKENS" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --warmup_steps "$WARMUP_STEPS" \
        --attention_type PolarANN \
        --device cuda:0 \
        --length long \
        --difficulty easy \
        --sink_size 64 \
        --local_size 256 \
        --collision_ratio 0.3 \
        --polar_cache_module "$POLAR_MODULE" \
        --final_topk "$FINAL_TOPK" \
        --enable_offload "$OFFLOAD_ARG" \
        2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "==== DONE: ${NSYS_OUT}.nsys-rep ====" | tee -a "$LOG"
echo "  nsys stats --report nvtx_sum ${NSYS_OUT}.nsys-rep" | tee -a "$LOG"
