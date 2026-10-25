import os

import torch
import torch.distributed as dist


def init_distributed():
    """Initialize torch.distributed when launched via torchrun."""
    distributed = False
    rank = 0
    local_rank = 0
    world_size = 1

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        distributed = True
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return distributed, rank, local_rank, world_size


def is_main_process(rank=0):
    return rank == 0


def barrier_if_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.barrier()


def broadcast_object(value, src=0, distributed=False):
    if not distributed or not dist.is_initialized():
        return value
    payload = [value]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()
