#!/usr/bin/env python
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Probe XCCL collectives on Kunlun XPU, one dtype and size at a time.

Nothing from the model or the recipe is involved, so a failure here is an
interconnect or runtime problem rather than a training bug. Run it with the
same torchrun invocation as training, first on one node and then across nodes:

    python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \\
        --master_addr=127.0.0.1 --master_port=29700 tools/kunlun/probe_xccl.py

    # node 0 and node 1, changing --node_rank
    python -m torch.distributed.run --nnodes=2 --node_rank=0 --nproc_per_node=8 \\
        --master_addr=<node0> --master_port=29700 tools/kunlun/probe_xccl.py

Each case prints one line from rank 0. A case that hangs rather than failing is
itself the finding: note which dtype and size stalled.
"""

import os
import time

import torch
import torch.distributed as dist

CASES = (
    ("float32", torch.float32, 2),
    ("float32", torch.float32, 1 << 20),
    ("int32", torch.int32, 2),
    ("int32", torch.int32, 1 << 20),
    # The recipe's first cross-node collective is an int64 all_reduce of two
    # token counts, which is the shape that has crashed BKCL's tree proxy.
    ("int64", torch.int64, 2),
    ("int64", torch.int64, 1 << 20),
    ("bfloat16", torch.bfloat16, 1 << 20),
)


def main() -> None:
    """Run every case in order, reporting each result from rank 0."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=os.environ.get("PROBE_BACKEND", "nccl"))
    rank = dist.get_rank()
    world = dist.get_world_size()
    if rank == 0:
        print(f"PROBE world={world} local_rank={local_rank} device={torch.cuda.current_device()}", flush=True)

    for name, dtype, numel in CASES:
        dist.barrier()
        tensor = torch.ones(numel, dtype=dtype, device=f"cuda:{local_rank}")
        started = time.time()
        try:
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            total = tensor[0].item()
            ok = total == world
            if rank == 0:
                print(
                    f"PROBE {'ok  ' if ok else 'WRONG'} all_reduce {name:8} numel={numel:<8}"
                    f" sum={total} expected={world} {time.time() - started:.3f}s",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - the probe reports every failure
            if rank == 0:
                print(f"PROBE FAIL all_reduce {name:8} numel={numel:<8} {type(exc).__name__}: {exc}", flush=True)
            break

    dist.barrier()
    if rank == 0:
        print("PROBE done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
