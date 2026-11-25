#!/usr/bin/env bash
# DSV4.1-Flash MedPix-VQA SFT on Kunlun P800 inside a TI-ONE training pod.
#
# Runs torchrun directly: unlike run_medpix_16xpu_kunlun.sh this does not start
# a container, because a platform job is already inside one.
#
# The recipe carries the tiny model shape but the processor comes from the
# official repo, so the image geometry has to match the official vision config
# (patch_size 14, downsample_ratio 3) and the image token id has to be 129264.
# Those overrides are applied below and are what the 8-card run needed.
#
# Single node, 8 cards:
#   NNODES=1 NPROC=8 CONFIG=.../deepseek_v41_flash_medpix_8xpu_ep8_kunlun.yaml \
#     examples/vlm_finetune/deepseek_v41/run_medpix_tione.sh
#
# Two nodes, 8 cards each (ep_size 16). Run on both, changing NODE_RANK:
#   MASTER_ADDR=9.0.144.4 NODE_RANK=0 examples/.../run_medpix_tione.sh
#   MASTER_ADDR=9.0.144.4 NODE_RANK=1 examples/.../run_medpix_tione.sh
# Or set PEER_HOST on node 0 and it ssh-launches node 1 itself:
#   MASTER_ADDR=9.0.144.4 PEER_HOST=9.0.144.9 examples/.../run_medpix_tione.sh
#
# Known risk on two nodes: the first cross-node collective is an int64
# all_reduce of token counts in base_recipe.py, which has crashed BKCL's
# tree-proxy kernel on this cluster. A hang with the progress bar at 0/N is
# that, not slow startup; the watchdog takes about 20 minutes to abort.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CONFIG="${CONFIG:-$ROOT/examples/vlm_finetune/deepseek_v41/deepseek_v41_flash_medpix_16xpu_ep16_kunlun.yaml}"
DATASET="${DATASET:-/mnt/cfs/tangqiao/codes/GPUS_TEST/data/dpskflash4_1/MedPix-VQA}"
PROCESSOR="${PROCESSOR:-deepseek-ai/DeepSeek-V4.1-Flash}"
PROCESSOR_LOCAL_ONLY="${PROCESSOR_LOCAL_ONLY:-False}"

NPROC="${NPROC:-8}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29654}"
PEER_HOST="${PEER_HOST:-}"

RUN_TAG="${RUN_TAG:-medpix-tione-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
# BKCL needs the RDMA netdev names for anything cross-node; without them
# init_bkcl_context fails with error code 2 and the Python layer only reports
# "error code= 0 ... no error". Default to every rdma* interface in the pod.
BKCL_RDMA_NICS="${BKCL_RDMA_NICS:-$(ls /sys/class/net 2>/dev/null | grep -E '^rdma[0-9]+$' | sort -V | paste -sd, - || true)}"
PYTHON="${PYTHON:-$(command -v python || command -v python3 || true)}"
if [[ -z "$PYTHON" ]]; then
    echo "no python on PATH; set PYTHON to the interpreter of the venv" >&2
    exit 1
fi

for path in "$CONFIG" "$DATASET"; do
    if [[ ! -e "$path" ]]; then
        echo "missing: $path" >&2
        exit 1
    fi
done
mkdir -p "$LOG_DIR"

visible="$(seq -s, 0 $((NPROC - 1)))"

overrides=(
    --dataset.path_or_dataset="$DATASET"
    --processor.pretrained_model_name_or_path="$PROCESSOR"
    --processor.local_files_only="$PROCESSOR_LOCAL_ONLY"
    --model.config.image_token_id=129264
    --model.config.vision_config.patch_size=14
    --model.config.vision_config.downsample_ratio=3
    --model.config.vision_config.max_image_tokens=1024
    --model.config.vision_config.min_pixels=295936
)

# Node 0 can bring the peer up over ssh instead of a second manual launch.
if [[ -n "$PEER_HOST" && "$NODE_RANK" == "0" ]]; then
    echo "launching node 1 on $PEER_HOST, log $LOG_DIR/$RUN_TAG-node1.log"
    ssh "$PEER_HOST" "cd $ROOT && ROOT=$ROOT CONFIG=$CONFIG DATASET=$DATASET \
        PROCESSOR=$PROCESSOR NPROC=$NPROC NNODES=$NNODES NODE_RANK=1 \
        MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT RUN_TAG=$RUN_TAG \
        LOG_DIR=$LOG_DIR PYTHON=$PYTHON BKCL_RDMA_NICS=$BKCL_RDMA_NICS bash $ROOT/examples/vlm_finetune/deepseek_v41/run_medpix_tione.sh" \
        > "$LOG_DIR/$RUN_TAG-node1.log" 2>&1 &
fi

echo "config     $CONFIG"
echo "dataset    $DATASET"
echo "processor  $PROCESSOR"
echo "node       $NODE_RANK of $NNODES, master $MASTER_ADDR:$MASTER_PORT"
echo "ranks      $NPROC per node, $((NPROC * NNODES)) total"
echo "rdma nics  ${BKCL_RDMA_NICS:-<none found, cross-node will fail>}"
echo "log        $LOG_DIR/$RUN_TAG-node$NODE_RANK.log"

cd "$ROOT"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
CUDA_VISIBLE_DEVICES="$visible" \
OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" \
RDMAV_FORK_SAFE="${RDMAV_FORK_SAFE:-1}" \
BKCL_RDMA_VERBS="${BKCL_RDMA_VERBS:-1}" \
BKCL_RDMA_NICS="$BKCL_RDMA_NICS" \
    "$PYTHON" -m torch.distributed.run \
    --nnodes="$NNODES" --node_rank="$NODE_RANK" --nproc_per_node="$NPROC" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    -m nemo_automodel.recipes.vlm.finetune -c "$CONFIG" "${overrides[@]}" \
    2>&1 | tee "$LOG_DIR/$RUN_TAG-node$NODE_RANK.log"

status=${PIPESTATUS[0]}
echo "$status" > "$LOG_DIR/$RUN_TAG-node$NODE_RANK.exit"
echo "exit $status"
exit "$status"
