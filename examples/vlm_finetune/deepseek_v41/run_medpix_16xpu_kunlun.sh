#!/usr/bin/env bash
# Launch DSV4.1-Flash MedPix-VQA SFT on 16 Kunlun P800 PCIe cards of one node.
#
# The container image ships xpytorch but not nemo_automodel, so the patched
# source tree is mounted and put on PYTHONPATH, the way the SJ53 runs do it.
# Every path below can be overridden from the environment.
set -euo pipefail

IMAGE="${IMAGE:-lym-ds41-min-sft:xav2608-nemo080}"
# Source tree and MedPix parquet live here; on SJ53 the directory is root-owned,
# so RUN_DIR holds everything the run writes.
WORK_DIR="${WORK_DIR:-/ssd1/liyiming06/deepseek41-nemo-smoke-20261110}"
RUN_DIR="${RUN_DIR:-/home/liyiming06/ds41_16xpu}"
SOURCE_DIR="${SOURCE_DIR:-$WORK_DIR/Automodel-6eed13db}"
CONFIG="${CONFIG:-$RUN_DIR/deepseek_v41_flash_medpix_16xpu_ep16_kunlun.yaml}"
# Host card ids to train on, comma separated. Override to skip a busy card.
XPU_IDS="${XPU_IDS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
IFS=',' read -r -a xpu_list <<<"$XPU_IDS"
NPROC="${NPROC:-${#xpu_list[@]}}"
MASTER_PORT="${MASTER_PORT:-29654}"
RUN_TAG="${RUN_TAG:-medpix-16xpu-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$RUN_DIR/logs}"

for path in "$SOURCE_DIR" "$CONFIG"; do
    if [[ ! -e "$path" ]]; then
        echo "missing: $path" >&2
        exit 1
    fi
done
mkdir -p "$LOG_DIR"

# Map every card on the host and let CUDA_VISIBLE_DEVICES pick the ones to
# train on: XCCL resolves OAM ids from the full device set, and a partial
# mapping makes it fail with "multiple devices are mapped to the same oam id".
# The image also carries CUDA_VISIBLE_DEVICES from the container it was
# committed from, so the visible set has to be passed again regardless.
devices=()
for node in /dev/xpu[0-9]*; do
    devices+=(--device "$node")
done
devices+=(--device /dev/xpuctrl)

echo "image      $IMAGE"
echo "config     $CONFIG"
echo "source     $SOURCE_DIR"
echo "cards      $XPU_IDS"
echo "ranks      $NPROC"
echo "log        $LOG_DIR/$RUN_TAG.log"

docker run --rm --entrypoint /bin/bash \
    --name "ds41-$RUN_TAG" \
    "${devices[@]}" \
    --shm-size 32g \
    --ulimit memlock=-1 \
    -v "$WORK_DIR:$WORK_DIR" \
    -v "$RUN_DIR:$RUN_DIR" \
    -w "$RUN_DIR" \
    -e PYTHONPATH="$SOURCE_DIR" \
    -e CUDA_VISIBLE_DEVICES="$XPU_IDS" \
    -e OMP_NUM_THREADS=2 \
    -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    "$IMAGE" -lc "python -m torch.distributed.run \
        --nnodes=1 --nproc_per_node=$NPROC \
        --master_addr=127.0.0.1 --master_port=$MASTER_PORT \
        -m nemo_automodel.recipes.vlm.finetune -c $CONFIG" \
    2>&1 | tee "$LOG_DIR/$RUN_TAG.log"

status=${PIPESTATUS[0]}
echo "$status" > "$LOG_DIR/$RUN_TAG.exit"
echo "exit $status"
exit "$status"
