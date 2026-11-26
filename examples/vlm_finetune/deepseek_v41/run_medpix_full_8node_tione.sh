#!/usr/bin/env bash
# DSV4.1-Flash full-parameter MedPix-VQA SFT on 8 nodes x 8 Kunlun P800 cards
# (64 ranks, EP=64) inside TI-ONE training pods.
#
# Runs torchrun directly: a platform job is already inside a container. Unlike
# run_medpix_tione.sh this passes no image-geometry overrides, because the full
# recipe takes its config from the official repo and already has the official
# values; overriding a nested field of a from_pretrained config would replace
# the whole sub-config with a one-key dict.
#
# From node 0, bringing the other seven up over ssh:
#   MASTER_ADDR=<node0> PEER_HOSTS=<n1>,<n2>,<n3>,<n4>,<n5>,<n6>,<n7> \
#     examples/vlm_finetune/deepseek_v41/run_medpix_full_8node_tione.sh
#
# Or once per node, changing NODE_RANK from 0 to 7:
#   MASTER_ADDR=<node0> NODE_RANK=<n> examples/.../run_medpix_full_8node_tione.sh
#
# Read the recipe header before launching. Two things decide whether this gets
# anywhere: host memory (about 750 GB of offloaded state per node) and the
# int64 cross-node all_reduce that has crashed BKCL's tree-proxy kernel on this
# cluster at two nodes. A progress bar stuck at 0/N is that crash, not startup.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CONFIG="${CONFIG:-$ROOT/examples/vlm_finetune/deepseek_v41/deepseek_v41_flash_medpix_full_ep64_offload_kunlun.yaml}"
DATASET="${DATASET:-/mnt/cfs/tangqiao/codes/GPUS_TEST/data/dpskflash4_1/MedPix-VQA}"

NPROC="${NPROC:-8}"
NNODES="${NNODES:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29654}"
PEER_HOSTS="${PEER_HOSTS:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

RUN_TAG="${RUN_TAG:-medpix-full-8node-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
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

overrides=(--dataset.path_or_dataset="$DATASET")
if [[ -n "$EXTRA_OVERRIDES" ]]; then
    # shellcheck disable=SC2206 # deliberate word splitting of a flag list
    overrides+=($EXTRA_OVERRIDES)
fi

# Node 0 can bring every peer up over ssh instead of one launch per pod. Peer
# order fixes the node ranks: the first host becomes rank 1.
if [[ -n "$PEER_HOSTS" && "$NODE_RANK" == "0" ]]; then
    IFS=',' read -r -a peers <<<"$PEER_HOSTS"
    if (( ${#peers[@]} + 1 != NNODES )); then
        echo "PEER_HOSTS lists ${#peers[@]} peers but NNODES is $NNODES" >&2
        exit 1
    fi
    for index in "${!peers[@]}"; do
        peer_rank=$((index + 1))
        echo "launching node $peer_rank on ${peers[index]}, log $LOG_DIR/$RUN_TAG-node$peer_rank.log"
        ssh "${peers[index]}" "cd $ROOT && ROOT=$ROOT CONFIG=$CONFIG DATASET=$DATASET \
            NPROC=$NPROC NNODES=$NNODES NODE_RANK=$peer_rank \
            MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT RUN_TAG=$RUN_TAG \
            LOG_DIR=$LOG_DIR PYTHON=$PYTHON EXTRA_OVERRIDES='$EXTRA_OVERRIDES' \
            bash $ROOT/examples/vlm_finetune/deepseek_v41/run_medpix_full_8node_tione.sh" \
            > "$LOG_DIR/$RUN_TAG-node$peer_rank.log" 2>&1 &
    done
fi

echo "config     $CONFIG"
echo "dataset    $DATASET"
echo "node       $NODE_RANK of $NNODES, master $MASTER_ADDR:$MASTER_PORT"
echo "ranks      $NPROC per node, $((NPROC * NNODES)) total"
echo "log        $LOG_DIR/$RUN_TAG-node$NODE_RANK.log"

cd "$ROOT"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
CUDA_VISIBLE_DEVICES="$visible" \
OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" \
RDMAV_FORK_SAFE="${RDMAV_FORK_SAFE:-1}" \
BKCL_RDMA_VERBS="${BKCL_RDMA_VERBS:-1}" \
    "$PYTHON" -m torch.distributed.run \
    --nnodes="$NNODES" --node_rank="$NODE_RANK" --nproc_per_node="$NPROC" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    -m nemo_automodel.recipes.vlm.finetune -c "$CONFIG" "${overrides[@]}" \
    2>&1 | tee "$LOG_DIR/$RUN_TAG-node$NODE_RANK.log"

status=${PIPESTATUS[0]}
echo "$status" > "$LOG_DIR/$RUN_TAG-node$NODE_RANK.exit"
echo "exit $status"
exit "$status"
