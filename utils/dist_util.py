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

    �˺������ڳ�ʼ���ֲ�ʽѵ������Ľ����顣���ȼ��ֲ�ʽ�����Ƿ��Ѿ���ʼ��������Ѿ���ʼ����ֱ�ӷ��ء�
    ���Ÿ����Ƿ��п��õ� CUDA �豸ѡ����ʵĺ�ˣ�gloo �� nccl���������û��������������ڵ��ַ����ǰ���̵������������С�����ڵ�˿ڡ�
    ���ʹ�û���������ʼ���ֲ�ʽ�����顣

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

    �˺�������ȷ���ڷֲ�ʽѵ����ʹ�õ��豸������п��õ� CUDA �豸������ݵ�ǰ���̵�������ÿ���ڵ�� GPU ����ѡ����ʵ� GPU �豸��
    ����ʹ�� CPU �豸��

    Returns:
        torch.device: ���ڷֲ�ʽѵ�����豸
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.

     �˺��������ڷֲ�ʽ�����м��� PyTorch ģ�͵�״̬�ֵ䣬�����ڶ�� MPI �����н���������ļ���ȡ������
    ֻ������Ϊ 0 �Ľ��̻�ʵ�ʶ�ȡ�ļ����ݣ�Ȼ�󽫶�ȡ�����ݹ㲥���������н��̡�������ݼ��ص� PyTorch ģ���С�

    Args:
        path (str): Ҫ���ص� PyTorch �ļ���·��
        **kwargs: ���ݸ� torch.load �����������ؼ��ֲ���

    Returns:
        dict: ���ص� PyTorch ģ�͵�״̬�ֵ�

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

     �˺��������ڷֲ�ʽ������ͬ��һϵ��������������Ϊ 0 �Ľ��̹㲥�������������н��̣�ȷ�����н��̵���������һ�¡�

    Args:
        params (Iterable[torch.Tensor]): Ҫͬ������������

    Returns:
        None

    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    """
    Find a free port on the current machine.

    �˺��������ڵ�ǰ�������ҵ�һ�����õĶ˿ڡ�ͨ������һ����ʱ�� TCP �׽��ֲ��󶨵�һ������Ŀ��ö˿ڣ�Ȼ�󷵻ظö˿ںš�
    ���ر���ʱ�׽��֡�

    Returns:
        int: ���õĶ˿ں�
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
