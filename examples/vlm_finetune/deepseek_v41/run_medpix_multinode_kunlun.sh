#!/usr/bin/env bash
# Launch a multi-node DSV4.1-Flash MedPix-VQA run on Kunlun P800 cards.
#
# Run it on every node with the same MASTER_ADDR and a distinct NODE_RANK.
# Single-node runs should use run_medpix_16xpu_kunlun.sh instead.
#
# UNVERIFIED: the largest P800 run so far is 8 cards of one node. 16 ranks on
# SJ53 still fail in XCCL with "multiple devices are mapped to the same oam
# id", and the full-parameter recipe needs the routed experts on host memory,
# which this recipe path cannot do. Treat this as the starting point for
# bringing a multi-node run up, not as a working configuration.
set -euo pipefail

IMAGE="${IMAGE:-cx-ds41-runtime:20261120}"
# Dataset and processor live under WORK_DIR; RUN_DIR holds the source tree,
# logs and checkpoints and has to be writable.
WORK_DIR="${WORK_DIR:-/ssd1/liyiming06/deepseek41-nemo-smoke-20261110}"
RUN_DIR="${RUN_DIR:-/home/liyiming06/ds41_run}"
SOURCE_DIR="${SOURCE_DIR:-$RUN_DIR/Automodel}"
CONFIG="${CONFIG:-$SOURCE_DIR/examples/vlm_finetune/deepseek_v41/deepseek_v41_flash_medpix_full_ep64_kunlun.yaml}"
XPU_IDS="${XPU_IDS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
IFS=',' read -r -a xpu_list <<<"$XPU_IDS"
NPROC="${NPROC:-${#xpu_list[@]}}"
NNODES="${NNODES:-4}"
NODE_RANK="${NODE_RANK:?set NODE_RANK to this node index, 0 on the MASTER_ADDR node}"
MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR to the address of the NODE_RANK=0 node}"
MASTER_PORT="${MASTER_PORT:-29654}"
RUN_TAG="${RUN_TAG:-medpix-full-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$RUN_DIR/logs}"

for path in "$SOURCE_DIR" "$CONFIG"; do
    if [[ ! -e "$path" ]]; then
        echo "missing: $path" >&2
        exit 1
    fi
done
mkdir -p "$LOG_DIR"

# Map every card and let CUDA_VISIBLE_DEVICES select, as in the single-node
# launcher: XCCL resolves OAM ids from the full device set.
devices=()
for node in /dev/xpu[0-9]*; do
    devices+=(--device "$node")
done
devices+=(--device /dev/xpuctrl)
# RDMA for inter-node XCCL, when the host exposes it.
if [[ -e /dev/infiniband ]]; then
    devices+=(--device /dev/infiniband)
fi

echo "image      $IMAGE"
echo "config     $CONFIG"
echo "source     $SOURCE_DIR"
echo "cards      $XPU_IDS"
echo "node       $NODE_RANK of $NNODES, master $MASTER_ADDR:$MASTER_PORT"
echo "ranks      $NPROC per node, $((NPROC * NNODES)) total"
echo "log        $LOG_DIR/$RUN_TAG-node$NODE_RANK.log"

# Host networking so ranks reach MASTER_ADDR and each other across nodes.
docker run --rm --entrypoint /bin/bash \
    --name "ds41-$RUN_TAG-node$NODE_RANK" \
    "${devices[@]}" \
    --network host \
    --ipc host \
    --shm-size 32g \
    --ulimit memlock=-1 \
    -v "$WORK_DIR:$WORK_DIR" \
    -v "$RUN_DIR:$RUN_DIR" \
    -w "$RUN_DIR" \
    -e PYTHONPATH="$SOURCE_DIR${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}" \
    -e CUDA_VISIBLE_DEVICES="$XPU_IDS" \
    -e OMP_NUM_THREADS=2 \
    -e PYTORCH_ALLOC_CONF=max_split_size_mb:512 \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    "$IMAGE" -lc "python -m torch.distributed.run \
        --nnodes=$NNODES --node_rank=$NODE_RANK --nproc_per_node=$NPROC \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        -m nemo_automodel.recipes.vlm.finetune -c $CONFIG" \
    2>&1 | tee "$LOG_DIR/$RUN_TAG-node$NODE_RANK.log"

status=${PIPESTATUS[0]}
echo "$status" > "$LOG_DIR/$RUN_TAG-node$NODE_RANK.exit"
echo "exit $status"
exit "$status"
