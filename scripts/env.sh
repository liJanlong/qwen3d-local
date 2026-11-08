# Source this before any Qwen-3D command:  source scripts/env.sh
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Conda install location: $CONDA_BASE if set, else ask conda, else ~/anaconda3
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate qwen3d
# The shell PATH hardcodes another env's bin dir ahead of conda's; force this env first
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1

export CUDA_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.9"   # RTX 4090
export PYTHONPATH="$PROJ/qwen3d:${PYTHONPATH:-}"

# Data / checkpoint paths (override the cluster paths hardcoded in qwen3d/scripts/main_qwen.sh)
export CKPTS_PATH="$PROJ/ckpts"
export CKPT_PATH="$PROJ/ckpts/qwen3d_3b.pth"
export QWEN_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
export DETECTRON2_DATASETS="$PROJ/data/SEMSEG_100k"
export DETECTRON2_DATASETS_2D="$PROJ/data/datasets_2d"
export REF_DATASET="$PROJ/data/refer_it_3d"
export CLIP_SAMPLING_PATH="$PROJ/data/refer_it_3d"
export PRECOMPUTED_SCANNET_PATH="$PROJ/data/scannet_meta"
export SCANNET_DATA_DIR="$PROJ/data/mask3d_processed/scannet/train_validation_database.yaml"
export SCANNET200_DATA_DIR="$PROJ/data/mask3d_processed/scannet200/train_validation_database.yaml"
export FEATURE_DIR="$PROJ/data/qwen_vit_features"   # only used if CACHE_QWEN_FEATURES True
export OUTPUT_DIR_PREFIX="$PROJ/outputs"
export NUM_GPUS=1
