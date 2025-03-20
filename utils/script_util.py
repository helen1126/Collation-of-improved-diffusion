import argparse
import inspect

from models import gaussian_diffusion as gd
from models.respace import SpacedDiffusion, space_timesteps
from models.unet import SuperResModel, UNetModel

NUM_CLASSES = 1000


def model_and_diffusion_defaults():
    """
    Defaults for image training.

    获取图像训练的默认参数。

    返回:
        dict: 包含图像训练默认参数的字典

    """
    return dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )


def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    sigma_small,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
):
    """
    根据给定参数创建模型和高斯扩散对象。

    参数:
        image_size (int): 图像的大小。
        class_cond (bool): 是否使用类别条件。
        learn_sigma (bool): 是否学习标准差。
        sigma_small (bool): 是否使用小的标准差。
        num_channels (int): 模型的通道数。
        num_res_blocks (int): 残差块的数量。
        num_heads (int): 注意力头的数量。
        num_heads_upsample (int): 上采样时注意力头的数量。
        attention_resolutions (str): 注意力分辨率。
        dropout (float): 丢弃率。
        diffusion_steps (int): 扩散步数。
        noise_schedule (str): 噪声调度方式。
        timestep_respacing (str): 时间步重新间隔。
        use_kl (bool): 是否使用 KL 散度损失。
        predict_xstart (bool): 是否预测起始图像。
        rescale_timesteps (bool): 是否重新缩放时间步。
        rescale_learned_sigmas (bool): 是否重新缩放学习到的标准差。
        use_checkpoint (bool): 是否使用梯度检查点。
        use_scale_shift_norm (bool): 是否使用缩放平移归一化。

    返回:
        tuple: 包含模型和高斯扩散对象的元组。
    """
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    """
    根据给定参数创建 UNet 模型。

    参数:
        image_size (int): 图像的大小。
        num_channels (int): 模型的通道数。
        num_res_blocks (int): 残差块的数量。
        learn_sigma (bool): 是否学习标准差。
        class_cond (bool): 是否使用类别条件。
        use_checkpoint (bool): 是否使用梯度检查点。
        attention_resolutions (str): 注意力分辨率。
        num_heads (int): 注意力头的数量。
        num_heads_upsample (int): 上采样时注意力头的数量。
        use_scale_shift_norm (bool): 是否使用缩放平移归一化。
        dropout (float): 丢弃率。

    返回:
        UNetModel: 创建好的 UNet 模型。
    """
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def sr_model_and_diffusion_defaults():
    """
    获取超分辨率模型和扩散的默认参数。

    返回:
        dict: 包含超分辨率模型和扩散默认参数的字典。
    """
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
    large_size,
    small_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
):
    """
    根据给定参数创建超分辨率模型和高斯扩散对象。

    参数:
        large_size (int): 高分辨率图像的大小。
        small_size (int): 低分辨率图像的大小。
        class_cond (bool): 是否使用类别条件。
        learn_sigma (bool): 是否学习标准差。
        num_channels (int): 模型的通道数。
        num_res_blocks (int): 残差块的数量。
        num_heads (int): 注意力头的数量。
        num_heads_upsample (int): 上采样时注意力头的数量。
        attention_resolutions (str): 注意力分辨率。
        dropout (float): 丢弃率。
        diffusion_steps (int): 扩散步数。
        noise_schedule (str): 噪声调度方式。
        timestep_respacing (str): 时间步重新间隔。
        use_kl (bool): 是否使用 KL 散度损失。
        predict_xstart (bool): 是否预测起始图像。
        rescale_timesteps (bool): 是否重新缩放时间步。
        rescale_learned_sigmas (bool): 是否重新缩放学习到的标准差。
        use_checkpoint (bool): 是否使用梯度检查点。
        use_scale_shift_norm (bool): 是否使用缩放平移归一化。

    返回:
        tuple: 包含超分辨率模型和高斯扩散对象的元组。
    """
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def sr_create_model(
    large_size,
    small_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    """
    根据给定参数创建超分辨率模型。

    参数:
        large_size (int): 高分辨率图像的大小。
        small_size (int): 低分辨率图像的大小。
        num_channels (int): 模型的通道数。
        num_res_blocks (int): 残差块的数量。
        learn_sigma (bool): 是否学习标准差。
        class_cond (bool): 是否使用类别条件。
        use_checkpoint (bool): 是否使用梯度检查点。
        attention_resolutions (str): 注意力分辨率。
        num_heads (int): 注意力头的数量。
        num_heads_upsample (int): 上采样时注意力头的数量。
        use_scale_shift_norm (bool): 是否使用缩放平移归一化。
        dropout (float): 丢弃率。

    返回:
        SuperResModel: 创建好的超分辨率模型。
    """
    _ = small_size  # hack to prevent unused variable

    if large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return SuperResModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    """
    根据给定参数创建高斯扩散对象。

    参数:
        steps (int, optional): 扩散步数，默认为 1000。
        learn_sigma (bool, optional): 是否学习标准差，默认为 False。
        sigma_small (bool, optional): 是否使用小的标准差，默认为 False。
        noise_schedule (str, optional): 噪声调度方式，默认为 "linear"。
        use_kl (bool, optional): 是否使用 KL 散度损失，默认为 False。
        predict_xstart (bool, optional): 是否预测起始图像，默认为 False。
        rescale_timesteps (bool, optional): 是否重新缩放时间步，默认为 False。
        rescale_learned_sigmas (bool, optional): 是否重新缩放学习到的标准差，默认为 False。
        timestep_respacing (str, optional): 时间步重新间隔，默认为 ""。

    返回:
        SpacedDiffusion: 创建好的高斯扩散对象。
    """
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    """
    将字典中的键值对添加到命令行参数解析器中。

    参数:
        parser (argparse.ArgumentParser): 命令行参数解析器。
        default_dict (dict): 包含默认参数的字典。

    返回:
        None
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    """
    从命令行参数对象中提取指定键的值，组成字典。

    参数:
        args (argparse.Namespace): 命令行参数对象。
        keys (list): 要提取的键的列表。

    返回:
        dict: 包含指定键值对的字典。
    """
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    """
    将字符串转换为布尔值。

    参数:
        v (str or bool): 要转换的字符串或布尔值。

    返回:
        bool: 转换后的布尔值。

    异常:
        argparse.ArgumentTypeError: 如果输入的字符串不是有效的布尔值表示。
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
