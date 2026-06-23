#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# RULER 131072 evaluation for Llama-3.1-8B-Instruct with the KIVI separate-kernel
# PolarANN cache (cache_hub.polar_cache_kivi_quest_density_seperate_kernel).
#
# Pariskv2 config requested for this run:
#   - large chunk ratio = 0.8   -> POLAR_QUEST_LARGE_RATIO
#   - small chunk ratio = 0.5   -> POLAR_QUEST_SMALL_RATIO
#   - dense ratio       = 0.4   -> POLAR_CDS_DENSE_RATIO
#   - large chunk topk  = 16    -> POLAR_CDS_DENSE_TOPK   (dense/high-score chunks)
#   - small chunk topk  = 8     -> POLAR_CDS_SPARSE_TOPK  (sparse/low-score chunks)
#
# Scoring / generation are aligned with ShadowKV (data/dataset.py, data/metrics.py):
#   - per-task decode length handled inside run/test_ruler.py
#     niah=128, vt=30, cwe=120, fwe=50, qa=32  (MAX_GEN left empty => per-task)
#   - per-task metric (needle_score / multi_number / multi_words / string_match_part)
#     is computed and written to each row as "score".
#   - Llama-3.1-8B native context is 131072, so NO YaRN scaling is applied
#     (ROPE_FACTOR=none).
#
# Usage:
#   CUDA_DEVICE=0 bash run_ruler/run_ruler_llama31_8b_kivi_quest_density.sh
#   # smoke (first N samples / task):
#   DATA_IDX_TO=2 bash run_ruler/run_ruler_llama31_8b_kivi_quest_density.sh
#   # resume an interrupted run into the same OUT_DIR:
#   OUT_DIR=/path/to/out RESUME=1 bash run_ruler/run_ruler_llama31_8b_kivi_quest_density.sh
# ============================================================================

# ---------- optional conda env ----------
if [ -n "${ZOOMKV_CONDA_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV:-}" != "$ZOOMKV_CONDA_ENV" ]; then
    conda activate "$ZOOMKV_CONDA_ENV"
fi

# ---------- optional glibc stub for some prebuilt flash_attn wheels ----------
GLIBC232_STUB="${CONDA_PREFIX:-}/glibc232_stub.so"
if [ -f "$GLIBC232_STUB" ]; then
    case ":${LD_PRELOAD:-}:" in
        *":$GLIBC232_STUB:"*) ;;
        *) export LD_PRELOAD="$GLIBC232_STUB${LD_PRELOAD:+:$LD_PRELOAD}" ;;
    esac
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

# ---------- basic params ----------
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MODEL_NAME="${MODEL_NAME:-$ROOT/models/Llama-3.1-8B-Instruct}"
DATA_DIR="${DATA_DIR:-$ROOT/data/ruler/llama-3/131072}"
TASKS="${TASKS:-niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2}"
DATA_IDX_TO="${DATA_IDX_TO-}"          # empty => all samples; set N for smoke

ATTENTION_TYPE="PolarANN"
DTYPE="${DTYPE:-bf16}"
# MAX_GEN empty => run/test_ruler.py uses ShadowKV per-task decode lengths.
MAX_GEN="${MAX_GEN-}"
SINK_SIZE="${SINK_SIZE:-64}"
LOCAL_SIZE="${LOCAL_SIZE:-256}"
FINAL_TOPK="${FINAL_TOPK:-2048}"
CANDIDATE_RATIO="${CANDIDATE_RATIO:-0.1}"
COLLISION_RATIO="${COLLISION_RATIO:-0.3}"
FORCE_CAND_TOKENS="${POLAR_FORCE_CAND_TOKENS:-2048}"
# Llama-3.1-8B native context == 131072 == data length => no YaRN.
ROPE_FACTOR="${ROPE_FACTOR:-none}"
ENABLE_OFFLOAD="${ENABLE_OFFLOAD:-1}"

# ---------- cache module ----------
export POLAR_CACHE_MODULE="${POLAR_CACHE_MODULE:-cache_hub.polar_cache_kivi_quest_density_seperate_kernel}"
export POLAR_PURE_RABITQ="${POLAR_PURE_RABITQ:-0}"
export POLAR_KIVI_BITS="${POLAR_KIVI_BITS:-4}"

# ---------- Quest stage-1 / hierarchical chunk selection ----------
export POLAR_QUEST_STAGE1="${POLAR_QUEST_STAGE1:-1}"
export POLAR_QUEST_CHUNK="${POLAR_QUEST_CHUNK:-16}"
export POLAR_QUEST_MODE="${POLAR_QUEST_MODE:-minmax_raw_cached}"
export POLAR_QUEST_CAND_RATIO="${POLAR_QUEST_CAND_RATIO:-$CANDIDATE_RATIO}"
export POLAR_QUEST_HIERARCHICAL="${POLAR_QUEST_HIERARCHICAL:-1}"
export POLAR_QUEST_LARGE_CHUNK="${POLAR_QUEST_LARGE_CHUNK:-256}"
export POLAR_QUEST_LARGE_RATIO="${POLAR_QUEST_LARGE_RATIO:-0.8}"   # large chunk ratio
export POLAR_QUEST_SMALL_RATIO="${POLAR_QUEST_SMALL_RATIO:-0.5}"   # small chunk ratio
export POLAR_FORCE_CAND_TOKENS="$FORCE_CAND_TOKENS"

# ---------- CDS dense/sparse stage-2 ----------
export POLAR_CDS_ENABLED="${POLAR_CDS_ENABLED:-1}"
export POLAR_CDS_DENSE_RATIO="${POLAR_CDS_DENSE_RATIO:-0.4}"       # dense ratio
export POLAR_CDS_DENSE_TOPK="${POLAR_CDS_DENSE_TOPK:-16}"          # large chunk topk
export POLAR_CDS_SPARSE_TOPK="${POLAR_CDS_SPARSE_TOPK:-8}"         # small chunk topk
export POLAR_TOPK_BACKEND="${POLAR_TOPK_BACKEND:-float}"
export POLAR_CDS_GLOBAL_TOPK_BACKEND="${POLAR_CDS_GLOBAL_TOPK_BACKEND:-torch}"

export POLAR_PIPELINE_RECALL="${POLAR_PIPELINE_RECALL:-0}"
export POLAR_CDS_RECALL_CHECK="${POLAR_CDS_RECALL_CHECK:-0}"
export POLAR_PIPELINE_RABITQ_ABLATION="${POLAR_PIPELINE_RABITQ_ABLATION:-0}"

# ---------- sanity ----------
if [ ! -d "$MODEL_NAME" ]; then
    echo "[ERROR] MODEL_NAME not found: $MODEL_NAME" >&2
    exit 2
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "[ERROR] RULER DATA_DIR not found: $DATA_DIR" >&2
    echo "        Expect task subdirs like niah_single_1/validation.jsonl under it." >&2
    exit 2
fi

DATA_LEN="$(basename "$DATA_DIR")"
TIMESTAMP="${SHARED_TS:-$(date +"%Y%m%d_%H%M%S")}"
SMOKE_TAG="full"
DATA_ARGS=()
if [ -n "$DATA_IDX_TO" ]; then
    SMOKE_TAG="first${DATA_IDX_TO}"
    DATA_ARGS=(--data_idx_to "$DATA_IDX_TO")
fi

GEN_ARGS=()
if [ -n "$MAX_GEN" ]; then
    GEN_ARGS=(--max_gen "$MAX_GEN")
fi

RESUME="${RESUME:-0}"
if [ "$RESUME" = "1" ]; then
    DATA_ARGS+=(--resume)
fi

OUT_DIR="${OUT_DIR:-$ROOT/results/ruler_kivi_llama31_8b/llama31_8b_kivi_sk_topk${FINAL_TOPK}_${DATA_LEN}_tasks13_${SMOKE_TAG}_${TIMESTAMP}}"
LOG_DIR="$ROOT/results/ruler_kivi_llama31_8b"
LOG_FILE="${LOG_FILE:-$LOG_DIR/ruler_llama31_8b_kivi_sk_topk${FINAL_TOPK}_tasks13_${SMOKE_TAG}_${TIMESTAMP}.log}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "=========================================="
echo "  RULER kivi separate-kernel (Llama-3.1-8B)"
echo "=========================================="
echo "module:       ${POLAR_CACHE_MODULE}"
echo "model:        ${MODEL_NAME}"
echo "data:         ${DATA_DIR}"
echo "tasks:        ${TASKS}"
echo "samples:      ${SMOKE_TAG}"
echo "GPU:          cuda:${CUDA_DEVICE}"
echo "YaRN:         factor=${ROPE_FACTOR}  (Llama native 131072; expect no scaling)"
echo "max_gen:      ${MAX_GEN:-<per-task ShadowKV>}"
echo "final_topk:   ${FINAL_TOPK}"
echo "force budget: ${POLAR_FORCE_CAND_TOKENS}"
echo "candidate:    ${CANDIDATE_RATIO}"
echo "offload:      ${ENABLE_OFFLOAD}"
echo "large/small:  ${POLAR_QUEST_LARGE_CHUNK}@${POLAR_QUEST_LARGE_RATIO} / ${POLAR_QUEST_CHUNK}@${POLAR_QUEST_SMALL_RATIO}"
echo "CDS:          dense_ratio=${POLAR_CDS_DENSE_RATIO} d${POLAR_CDS_DENSE_TOPK}/s${POLAR_CDS_SPARSE_TOPK}"
echo "output:       ${OUT_DIR}"
echo "log:          ${LOG_FILE}"
echo "=========================================="

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
python run/test_ruler.py \
    --model_name "${MODEL_NAME}" \
    --data_dir "${DATA_DIR}" \
    --tasks "${TASKS}" \
    --attention_type "${ATTENTION_TYPE}" \
    --sink_size "${SINK_SIZE}" \
    --local_size "${LOCAL_SIZE}" \
    --final_topk "${FINAL_TOPK}" \
    --collision_ratio "${COLLISION_RATIO}" \
    --candidate_ratio "${CANDIDATE_RATIO}" \
    --enable_offload "${ENABLE_OFFLOAD}" \
    --rope_factor "${ROPE_FACTOR}" \
    --device cuda:0 \
    --dtype "${DTYPE}" \
    --out_dir "${OUT_DIR}" \
    --polar_cache_module "${POLAR_CACHE_MODULE}" \
    "${GEN_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "=========================================="
echo "  Done. Summarize per-task / macro-avg score:"
echo "    python run_ruler/summarize_ruler.py ${OUT_DIR}"
echo "=========================================="
echo "output: ${OUT_DIR}"
echo "log:    ${LOG_FILE}"

python run_ruler/summarize_ruler.py "${OUT_DIR}" || true
