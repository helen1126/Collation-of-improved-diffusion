"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.

    此函数用于初始化分布式训练所需的进程组。首先检查分布式环境是否已经初始化，如果已经初始化则直接返回。
    接着根据是否有可用的 CUDA 设备选择合适的后端（gloo 或 nccl），并设置环境变量，如主节点地址、当前进程的排名、世界大小和主节点端口。
    最后使用环境变量初始化分布式进程组。

    Returns:
        None
    """
    if dist.is_initialized():
        return

    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.

    此函数用于确定在分布式训练中使用的设备。如果有可用的 CUDA 设备，则根据当前进程的排名和每个节点的 GPU 数量选择合适的 GPU 设备；
    否则使用 CPU 设备。

    Returns:
        torch.device: 用于分布式训练的设备
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.

     此函数用于在分布式环境中加载 PyTorch 模型的状态字典，避免在多个 MPI 进程中进行冗余的文件读取操作。
    只有排名为 0 的进程会实际读取文件内容，然后将读取的数据广播给其他所有进程。最后将数据加载到 PyTorch 模型中。

    Args:
        path (str): 要加载的 PyTorch 文件的路径
        **kwargs: 传递给 torch.load 函数的其他关键字参数

    Returns:
        dict: 加载的 PyTorch 模型的状态字典

    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data = MPI.COMM_WORLD.bcast(data)
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.

     此函数用于在分布式环境中同步一系列张量。从排名为 0 的进程广播张量到其他所有进程，确保所有进程的张量参数一致。

    Args:
        params (Iterable[torch.Tensor]): 要同步的张量序列

    Returns:
        None

    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    """
    Find a free port on the current machine.

    此函数用于在当前机器上找到一个可用的端口。通过创建一个临时的 TCP 套接字并绑定到一个随机的可用端口，然后返回该端口号。
    最后关闭临时套接字。

    Returns:
        int: 可用的端口号
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
