"""Build Qwen-3D exactly as `train.py --eval-only` does and load the checkpoint.

Usage (from qwen3d_project/qwen3d):  python ../scripts/smoke_load.py [extra cfg opts...]
"""
import os
import sys

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser

sys.path.insert(0, os.getcwd())
from train import Trainer, setup  # noqa: E402

args = default_argument_parser().parse_args(
    [
        "--config-file", "qwen3d/configs/qwen_3d.yaml",
        "--eval-only",
        "OUTPUT_DIR", os.path.join(os.environ["OUTPUT_DIR_PREFIX"], "smoke"),
        "MODEL.WEIGHTS", os.environ["CKPT_PATH"],
        "QWEN_MODEL", os.environ["QWEN_MODEL"],
        "USE_WANDB", "False",
        *sys.argv[1:],
    ]
)
cfg = setup(args)
model = Trainer.build_model(cfg)
DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)
model.eval()

n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"params: total={n_total / 1e9:.2f}B trainable={n_train / 1e6:.1f}M")
print(f"device={next(model.parameters()).device} "
      f"gpu_mem_alloc={torch.cuda.memory_allocated() / 2**30:.1f} GiB")
