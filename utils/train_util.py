import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from utils import dist_util

from utils import logger
from fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from ..models.nn import update_ema
from ..data.resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    """
    训练循环类，用于管理模型的训练过程，包括参数加载、优化步骤、日志记录和模型保存等操作。
    """
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        """
        初始化训练循环。

        参数:
            model (torch.nn.Module): 要训练的模型。
            diffusion: 扩散模型对象。
            data: 训练数据加载器。
            batch_size (int): 每个批次的样本数量。
            microbatch (int): 微批次的样本数量，如果小于等于 0 则使用 batch_size。
            lr (float): 学习率。
            ema_rate (float or list): 指数移动平均（EMA）的速率，可以是单个值或逗号分隔的多个值。
            log_interval (int): 日志记录的间隔步数。
            save_interval (int): 模型保存的间隔步数。
            resume_checkpoint (str): 恢复训练的检查点文件路径。
            use_fp16 (bool, optional): 是否使用混合精度训练，默认为 False。
            fp16_scale_growth (float, optional): 混合精度训练中损失缩放的增长速率，默认为 1e-3。
            schedule_sampler: 采样器，用于采样时间步，默认为 UniformSampler。
            weight_decay (float, optional): 权重衰减系数，默认为 0.0。
            lr_anneal_steps (int, optional): 学习率退火的步数，默认为 0。
        """
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        """
        加载并同步模型参数。如果指定了恢复检查点，则从检查点加载模型参数，并在分布式训练中同步参数。
        """
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        """
        加载指数移动平均（EMA）参数。如果存在 EMA 检查点，则从检查点加载 EMA 参数，并在分布式训练中同步参数。

        参数:
            rate (float): EMA 的速率。

        返回:
            list: 加载并同步后的 EMA 参数列表。
        """
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        """
        加载优化器的状态。如果存在优化器检查点，则从检查点加载优化器的状态。
        """
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        """
        设置混合精度训练。将模型参数转换为全精度的主参数，并将模型转换为半精度。
        """
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        """
        运行训练循环。不断从数据加载器中获取批次数据，执行训练步骤，记录日志并保存模型，直到达到学习率退火步数。
        """
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        """
        执行单个训练步骤。包括前向传播、反向传播、优化和日志记录。

        参数:
            batch: 输入的批次数据。
            cond: 条件信息。
        """
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond):
        """
        执行前向传播和反向传播。将批次数据分割为微批次，计算损失并反向传播梯度。

        参数:
            batch: 输入的批次数据。
            cond: 条件信息。
        """
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        """
        在混合精度训练中进行优化。检查梯度是否包含 NaN，如果包含则降低损失缩放因子，否则更新主参数和模型参数，并增加损失缩放因子。
        """
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        """
        在普通精度训练中进行优化。记录梯度范数，退火学习率，更新主参数和 EMA 参数。
        """
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        """
        记录梯度的范数。计算主参数梯度的平方和的平方根，并记录到日志中。
        """
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        """
        退火学习率。如果指定了学习率退火步数，则根据当前步数计算退火后的学习率，并更新优化器的学习率。
        """
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        """
        记录当前训练步骤的信息。包括步数、样本数和损失缩放因子（如果使用混合精度训练）。
        """
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        """
        保存模型和优化器的状态。保存主参数、EMA 参数和优化器的状态到检查点文件。
        """
        def save_checkpoint(rate, params):
            """
            保存指定速率的参数到检查点文件。

            参数:
                rate (float): EMA 的速率，如果为 0 则保存主参数。
                params (list): 要保存的参数列表。
            """
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        """
        将主参数转换为模型的状态字典。如果使用混合精度训练，先将主参数解扁平化，然后更新模型的状态字典。

        参数:
            master_params (list): 主参数列表。

        返回:
            dict: 模型的状态字典。
        """
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        """
        将模型的状态字典转换为主参数。如果使用混合精度训练，将模型参数转换为全精度的主参数。

        参数:
            state_dict (dict): 模型的状态字典。

        返回:
            list: 主参数列表。
        """
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    从文件名中解析恢复步骤的编号。文件名格式应为 path/to/modelNNNNNN.pt，其中 NNNNNN 是检查点的步数。

    参数:
        filename (str): 文件名。

    返回:
        int: 恢复步骤的编号，如果无法解析则返回 0。
    """
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    """
    获取日志存储目录。
    首先尝试从环境变量 DIFFUSION_BLOB_LOGDIR 中获取日志目录，
    如果该环境变量未设置，则使用 logger 模块中当前的日志目录。

    返回:
        str: 日志存储目录
    """
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    """
    查找恢复训练的检查点。
    在当前的实现中，此函数默认返回 None。
    在实际的基础设施中，可以重写此函数以自动发现最新的检查点，
    例如在 blob 存储中查找最新的检查点。

    返回:
        str or None: 恢复训练的检查点文件路径，如果未找到则返回 None
    """
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    """
    查找指数移动平均（EMA）检查点。

    参数:
        main_checkpoint (str): 主检查点文件路径
        step (int): 当前训练步数
        rate (float): EMA 率

    返回:
        str or None: EMA 检查点文件路径，如果未找到则返回 None
    """
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    """
    记录损失字典中的各项损失。
    对于损失字典中的每个键值对，记录该损失的均值，并按时间步的四分位数记录损失值。

    参数:
        diffusion: 扩散模型对象
        ts (torch.Tensor): 时间步张量
        losses (dict): 损失字典，键为损失名称，值为损失张量
    """
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
