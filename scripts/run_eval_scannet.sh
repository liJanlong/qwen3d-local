#!/bin/bash
# Single-GPU Qwen-3D eval on ScanNet, wrapping qwen3d/scripts/main_qwen.sh.
#
#   bash scripts/run_eval_scannet.sh grounding   # ScanRefer val_50, mask grounding + PyViz3D vis
#   bash scripts/run_eval_scannet.sh qa          # ScanQA val, text generation
#   bash scripts/run_eval_scannet.sh grounding MAX_FRAME_NUM 60   # extra cfg overrides go last
set -e
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJ/scripts/env.sh"
cd "$PROJ/qwen3d"
source scripts/setup.sh
configure_local

TASK=${1:-grounding}
shift || true

case "$TASK" in
  grounding)
    TASK_ARGS=(GENERATION False VISUALIZE_REF True USE_AUTO_NOUN_DETECTION False
      DATASETS.TRAIN "('scanrefer_scannet_anchor_train_single',)"
      DATASETS.TEST "('scanrefer_scannet_anchor_val_50_single_batched',)")
    ;;
  qa)
    TASK_ARGS=(GENERATION True
      DATASETS.TRAIN "('scanqa_ref_scannet_train_single',)"
      DATASETS.TEST "('scanqa_ref_scannet_val_single_batched',)")
    ;;
  *) echo "unknown task: $TASK (use grounding|qa)"; exit 1 ;;
esac

BS=1 EVAL_ONLY=1 RESUME=0 NUM_GPUS=1 NUM_DATALOADERS=2 NUM_VAL_DATALOADERS=1 \
CHECKPOINT_PERIOD=8000 EVAL_PERIOD=16000 \
bash scripts/main_qwen.sh \
  MODEL.WEIGHTS "$CKPT_PATH" \
  QWEN_MODEL "$QWEN_MODEL" \
  USE_WANDB False \
  "${TASK_ARGS[@]}" \
  "$@"
