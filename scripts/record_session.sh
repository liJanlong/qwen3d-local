#!/bin/bash
# Session shown in the recording: each command is echoed with a prompt, then run for real.
cd "$(dirname "$0")/.."
run() { printf '\033[1;32m%s\033[0m:\033[1;34m~/%s\033[0m$ %s\n' "$USER@$(hostname)" "$(basename "$PWD")" "$*"; eval "$@"; echo; }

run "source scripts/env.sh"
run "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"
run "python scripts/demo_custom.py --scene tum_office --query \"the teddy bear\" --video"
run "python scripts/demo_custom.py --scene tum_office --question \"What color is the dice on the desk?\""
sleep 2
