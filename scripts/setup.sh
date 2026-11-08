#!/bin/bash
# One-time setup after cloning: fetch submodules, patch upstream Qwen-3D, download + verify the checkpoint.
# Run from the repo root:  bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

git submodule update --init --recursive

# Make qwen3d/scripts/main_qwen.sh honor the path overrides exported by scripts/env.sh
PATCH=patches/qwen3d_main_qwen_env_overrides.patch
if git -C qwen3d apply --reverse --check "../$PATCH" 2>/dev/null; then
    echo "patch already applied"
else
    git -C qwen3d apply "../$PATCH" && echo "patch applied"
fi

mkdir -p ckpts data outputs my_scenes
CKPT=ckpts/qwen3d_3b.pth
SHA=5bb643fe1979fc538af0c7897284a2f57308a589e3118a684c6a73d109891e84
if [ ! -f "$CKPT" ]; then
    hf download katefgroup/Qwen-3D qwen3d_3b.pth --local-dir ckpts
fi
echo "$SHA  $CKPT" | sha256sum --check
