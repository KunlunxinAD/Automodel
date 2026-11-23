#!/usr/bin/env bash
# Full-parameter DSV4.1-Flash MedPix-VQA SFT on 16 Kunlun P800 cards of one
# node, with FSDP2 CPU offload.
#
# Thin wrapper: it selects the full-model offload recipe and hands everything
# else to run_medpix_16xpu_kunlun.sh, so IMAGE, WORK_DIR, RUN_DIR, SOURCE_DIR,
# XPU_IDS, EXTRA_PYTHONPATH and LOG_DIR work the same way here.
#
# Read the recipe header before launching: with AdamW this shape needs roughly
# 6 TB of host RAM on the node, and the optimizer step runs on the CPU, so
# throughput will be far below the on-device runs.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG="${CONFIG:-$here/deepseek_v41_flash_medpix_full_16xpu_offload_kunlun.yaml}"
export RUN_TAG="${RUN_TAG:-medpix-full-16xpu-$(date +%Y%m%d-%H%M%S)}"

exec "$here/run_medpix_16xpu_kunlun.sh" "$@"
