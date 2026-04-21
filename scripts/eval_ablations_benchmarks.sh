#!/bin/bash
# scripts/eval_ablations_benchmarks.sh
# 在统一 benchmark 上评估全部消融 checkpoint

set -e

ABLATIONS=(
  "A1_no_env_anchor"
  "A2_no_memory_consistency"
  "A3_no_confidence_filter"
  "A4_fixed_trigger"
  "A5_random_distill"
  "A6_no_consolidation"
  "A7_no_curriculum"
)

CUDA_DEVICE=${1:-0}  # 默认 GPU 0, 可通过第一个参数指定

for abl in "${ABLATIONS[@]}"; do
  echo "============================================================"
  echo "Evaluating $abl on LoCoMo + LongMemEval"
  echo "============================================================"

  CHECKPOINT_DIR="results/ablations/$abl/checkpoint"
  OUTPUT_DIR="results/ablations_eval/$abl"
  LOG_FILE="logs/ablation_eval_${abl}.log"

  mkdir -p "$OUTPUT_DIR"

  # Determine appropriate LoRA path
  LORA_PATH="$CHECKPOINT_DIR/lora"
  if [ ! -d "$LORA_PATH" ]; then
      LORA_PATH="outputs/phase1/best"
  fi

  # LoCoMo 评估
  echo "[$(date)] Starting LoCoMo evaluation for $abl"
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/eval_locomo.py \
    --checkpoint "$CHECKPOINT_DIR" \
    --lora_checkpoint "$LORA_PATH" \
    --benchmark locomo \
    --output_dir "$OUTPUT_DIR" \
    --no_qlora \
    2>&1 | tee -a "$LOG_FILE"

  # LongMemEval 评估
  echo "[$(date)] Starting LongMemEval evaluation for $abl"
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/eval_locomo.py \
    --checkpoint "$CHECKPOINT_DIR" \
    --lora_checkpoint "$LORA_PATH" \
    --benchmark longmemeval \
    --output_dir "$OUTPUT_DIR" \
    --no_qlora \
    2>&1 | tee -a "$LOG_FILE"

  echo "[$(date)] $abl evaluation complete"
  echo ""
done

echo "All ablation evaluations complete!"
