#!/usr/bin/env bash
# Minimal LongBench-v2 entry point for the ZoomKV KIVI backend.
#
# Example:
#   MODEL_NAME=/path/to/Qwen3-4B-Thinking-2507 \
#   CUDA_DEVICE=0 NUM_SAMPLES=8 FINAL_TOPK=100 \
#   bash run/run_longbench_v2_zoomkv.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_NAME="${MODEL_NAME:-$ROOT/models/Qwen3-4B-Thinking-2507}"
DATASET_NAME="${DATASET_NAME:-zai-org/LongBench-v2}"
SPLIT_NAME="${SPLIT_NAME:-train}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
FINAL_TOPK="${FINAL_TOPK:-100}"
DTYPE="${DTYPE:-bf16}"

# LongBench-v2 in the paper uses two-stage CoT decoding: 1024 reasoning
# tokens followed by a 512-token answer phase.
COT_MAX_NEW_TOKENS="${COT_MAX_NEW_TOKENS:-1024}"
ANSWER_MAX_NEW_TOKENS="${ANSWER_MAX_NEW_TOKENS:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

export POLAR_CACHE_MODULE="${POLAR_CACHE_MODULE:-cache_hub.polar_cache_kivi_quest_density_seperate_kernel}"

# Stage 1: hierarchical Quest chunk filtering.
export POLAR_QUEST_HIERARCHICAL="${POLAR_QUEST_HIERARCHICAL:-1}"
export POLAR_QUEST_CHUNK="${POLAR_QUEST_CHUNK:-16}"
export POLAR_QUEST_LARGE_CHUNK="${POLAR_QUEST_LARGE_CHUNK:-256}"
export POLAR_QUEST_LARGE_RATIO="${POLAR_QUEST_LARGE_RATIO:-0.8}"
export POLAR_QUEST_SMALL_RATIO="${POLAR_QUEST_SMALL_RATIO:-0.5}"
export POLAR_QUEST_CAND_RATIO="${POLAR_QUEST_CAND_RATIO:-0.1}"

# Stage 2: dense/sparse chunk-local KIVI reranking.
export POLAR_CDS_DENSE_RATIO="${POLAR_CDS_DENSE_RATIO:-0.4}"
export POLAR_CDS_DENSE_TOPK="${POLAR_CDS_DENSE_TOPK:-16}"
export POLAR_CDS_SPARSE_TOPK="${POLAR_CDS_SPARSE_TOPK:-8}"

export POLAR_PIPELINE_RECALL="${POLAR_PIPELINE_RECALL:-0}"
export POLAR_CDS_RECALL_CHECK="${POLAR_CDS_RECALL_CHECK:-0}"

OUT_DIR="${OUT_DIR:-$ROOT/results/longbench_v2_zoomkv}"
mkdir -p "$OUT_DIR"
STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_FILE="$OUT_DIR/lbv2_zoomkv_top${FINAL_TOPK}_n${NUM_SAMPLES}_${STAMP}.jsonl"

echo "[ZoomKV] model      : $MODEL_NAME"
echo "[ZoomKV] dataset    : $DATASET_NAME/$SPLIT_NAME"
echo "[ZoomKV] samples    : $NUM_SAMPLES"
echo "[ZoomKV] final_topk : $FINAL_TOPK"
echo "[ZoomKV] gpu        : cuda:$CUDA_DEVICE"
echo "[ZoomKV] module     : $POLAR_CACHE_MODULE"
echo "[ZoomKV] output     : $OUT_FILE"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
python run/test_longbench_v2.py \
  --model_name "$MODEL_NAME" \
  --dataset_name "$DATASET_NAME" \
  --split_name "$SPLIT_NAME" \
  --num_samples "$NUM_SAMPLES" \
  --attention_type PolarANN \
  --polar_cache_module "$POLAR_CACHE_MODULE" \
  --candidate_ratio "$POLAR_QUEST_CAND_RATIO" \
  --final_topk "$FINAL_TOPK" \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --device cuda:0 \
  --dtype "$DTYPE" \
  --output_file "$OUT_FILE" \
  --force_process \
  --sink_size 64 \
  --local_size 256 \
  --enable_cot \
  --cot_max_new_tokens "$COT_MAX_NEW_TOKENS" \
  --answer_max_new_tokens "$ANSWER_MAX_NEW_TOKENS"
