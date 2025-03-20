"""
Helpers to train with 16-bit precision.
"""

import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.

    Args:
         l (torch.nn.Module): 要转换的模块。
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().

    Args:
        l (torch.nn.Module): 要转换的模块。

    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        l.bias.data = l.bias.data.float()


def make_master_params(model_params):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.

    Args:
        model_params (Iterable[torch.Tensor]): 模型的参数列表。

    Returns:
        list[torch.nn.Parameter]: 包含展平后的主参数的列表。
    """
    master_params = _flatten_dense_tensors(
        [param.detach().float() for param in model_params]
    )
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]


def model_grads_to_master_grads(model_params, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().

    Args:
        model_params (Iterable[torch.Tensor]): 模型的参数列表。
        master_params (list[torch.nn.Parameter]): 主参数列表。

    """
    master_params[0].grad = _flatten_dense_tensors(
        [param.grad.data.detach().float() for param in model_params]
    )


def master_params_to_model_params(model_params, master_params):
    """
    Copy the master parameter data back into the model parameters.

    Args:
        model_params (Iterable[torch.Tensor]): 模型的参数列表。
        master_params (list[torch.nn.Parameter]): 主参数列表。

    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    model_params = list(model_params)

    for param, master_param in zip(
        model_params, unflatten_master_params(model_params, master_params)
    ):
        param.detach().copy_(master_param)


def unflatten_master_params(model_params, master_params):
    """
    Unflatten the master parameters to look like model_params.

    Args:
        model_params (Iterable[torch.Tensor]): 模型的参数列表。
        master_params (list[torch.nn.Parameter]): 主参数列表。
    """
    return _unflatten_dense_tensors(master_params[0].detach(), tuple(tensor for tensor in model_params))


def zero_grad(model_params):
    """
    将模型参数的梯度置零。
    该函数会遍历模型参数，如果参数的梯度不为空，则将其梯度从计算图中分离并置零。

    参数:
        model_params (Iterable[torch.Tensor]): 模型的参数列表。
    """
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()
