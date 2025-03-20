import argparse
import inspect

from models import gaussian_diffusion as gd
from models.respace import SpacedDiffusion, space_timesteps
from models.unet import SuperResModel, UNetModel

NUM_CLASSES = 1000


def model_and_diffusion_defaults():
    """
    Defaults for image training.

    ��ȡͼ��ѵ����Ĭ�ϲ�����

    ����:
        dict: ����ͼ��ѵ��Ĭ�ϲ������ֵ�

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
    ���ݸ�����������ģ�ͺ͸�˹��ɢ����

    ����:
        image_size (int): ͼ��Ĵ�С��
        class_cond (bool): �Ƿ�ʹ�����������
        learn_sigma (bool): �Ƿ�ѧϰ��׼�
        sigma_small (bool): �Ƿ�ʹ��С�ı�׼�
        num_channels (int): ģ�͵�ͨ������
        num_res_blocks (int): �в���������
        num_heads (int): ע����ͷ��������
        num_heads_upsample (int): �ϲ���ʱע����ͷ��������
        attention_resolutions (str): ע�����ֱ��ʡ�
        dropout (float): �����ʡ�
        diffusion_steps (int): ��ɢ������
        noise_schedule (str): �������ȷ�ʽ��
        timestep_respacing (str): ʱ�䲽���¼����
        use_kl (bool): �Ƿ�ʹ�� KL ɢ����ʧ��
        predict_xstart (bool): �Ƿ�Ԥ����ʼͼ��
        rescale_timesteps (bool): �Ƿ���������ʱ�䲽��
        rescale_learned_sigmas (bool): �Ƿ���������ѧϰ���ı�׼�
        use_checkpoint (bool): �Ƿ�ʹ���ݶȼ��㡣
        use_scale_shift_norm (bool): �Ƿ�ʹ������ƽ�ƹ�һ����

    ����:
        tuple: ����ģ�ͺ͸�˹��ɢ�����Ԫ�顣
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
    ���ݸ����������� UNet ģ�͡�

    ����:
        image_size (int): ͼ��Ĵ�С��
        num_channels (int): ģ�͵�ͨ������
        num_res_blocks (int): �в���������
        learn_sigma (bool): �Ƿ�ѧϰ��׼�
        class_cond (bool): �Ƿ�ʹ�����������
        use_checkpoint (bool): �Ƿ�ʹ���ݶȼ��㡣
        attention_resolutions (str): ע�����ֱ��ʡ�
        num_heads (int): ע����ͷ��������
        num_heads_upsample (int): �ϲ���ʱע����ͷ��������
        use_scale_shift_norm (bool): �Ƿ�ʹ������ƽ�ƹ�һ����
        dropout (float): �����ʡ�

    ����:
        UNetModel: �����õ� UNet ģ�͡�
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
    ��ȡ���ֱ���ģ�ͺ���ɢ��Ĭ�ϲ�����

    ����:
        dict: �������ֱ���ģ�ͺ���ɢĬ�ϲ������ֵ䡣
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
    ���ݸ��������������ֱ���ģ�ͺ͸�˹��ɢ����

    ����:
        large_size (int): �߷ֱ���ͼ��Ĵ�С��
        small_size (int): �ͷֱ���ͼ��Ĵ�С��
        class_cond (bool): �Ƿ�ʹ�����������
        learn_sigma (bool): �Ƿ�ѧϰ��׼�
        num_channels (int): ģ�͵�ͨ������
        num_res_blocks (int): �в���������
        num_heads (int): ע����ͷ��������
        num_heads_upsample (int): �ϲ���ʱע����ͷ��������
        attention_resolutions (str): ע�����ֱ��ʡ�
        dropout (float): �����ʡ�
        diffusion_steps (int): ��ɢ������
        noise_schedule (str): �������ȷ�ʽ��
        timestep_respacing (str): ʱ�䲽���¼����
        use_kl (bool): �Ƿ�ʹ�� KL ɢ����ʧ��
        predict_xstart (bool): �Ƿ�Ԥ����ʼͼ��
        rescale_timesteps (bool): �Ƿ���������ʱ�䲽��
        rescale_learned_sigmas (bool): �Ƿ���������ѧϰ���ı�׼�
        use_checkpoint (bool): �Ƿ�ʹ���ݶȼ��㡣
        use_scale_shift_norm (bool): �Ƿ�ʹ������ƽ�ƹ�һ����

    ����:
        tuple: �������ֱ���ģ�ͺ͸�˹��ɢ�����Ԫ�顣
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
    ���ݸ��������������ֱ���ģ�͡�

    ����:
        large_size (int): �߷ֱ���ͼ��Ĵ�С��
        small_size (int): �ͷֱ���ͼ��Ĵ�С��
        num_channels (int): ģ�͵�ͨ������
        num_res_blocks (int): �в���������
        learn_sigma (bool): �Ƿ�ѧϰ��׼�
        class_cond (bool): �Ƿ�ʹ�����������
        use_checkpoint (bool): �Ƿ�ʹ���ݶȼ��㡣
        attention_resolutions (str): ע�����ֱ��ʡ�
        num_heads (int): ע����ͷ��������
        num_heads_upsample (int): �ϲ���ʱע����ͷ��������
        use_scale_shift_norm (bool): �Ƿ�ʹ������ƽ�ƹ�һ����
        dropout (float): �����ʡ�

    ����:
        SuperResModel: �����õĳ��ֱ���ģ�͡�
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
    ���ݸ�������������˹��ɢ����

    ����:
        steps (int, optional): ��ɢ������Ĭ��Ϊ 1000��
        learn_sigma (bool, optional): �Ƿ�ѧϰ��׼�Ĭ��Ϊ False��
        sigma_small (bool, optional): �Ƿ�ʹ��С�ı�׼�Ĭ��Ϊ False��
        noise_schedule (str, optional): �������ȷ�ʽ��Ĭ��Ϊ "linear"��
        use_kl (bool, optional): �Ƿ�ʹ�� KL ɢ����ʧ��Ĭ��Ϊ False��
        predict_xstart (bool, optional): �Ƿ�Ԥ����ʼͼ��Ĭ��Ϊ False��
        rescale_timesteps (bool, optional): �Ƿ���������ʱ�䲽��Ĭ��Ϊ False��
        rescale_learned_sigmas (bool, optional): �Ƿ���������ѧϰ���ı�׼�Ĭ��Ϊ False��
        timestep_respacing (str, optional): ʱ�䲽���¼����Ĭ��Ϊ ""��

    ����:
        SpacedDiffusion: �����õĸ�˹��ɢ����
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
    ���ֵ��еļ�ֵ����ӵ������в����������С�

    ����:
        parser (argparse.ArgumentParser): �����в�����������
        default_dict (dict): ����Ĭ�ϲ������ֵ䡣

    ����:
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
    �������в�����������ȡָ������ֵ������ֵ䡣

    ����:
        args (argparse.Namespace): �����в�������
        keys (list): Ҫ��ȡ�ļ����б�

    ����:
        dict: ����ָ����ֵ�Ե��ֵ䡣
    """
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    """
    ���ַ���ת��Ϊ����ֵ��

    ����:
        v (str or bool): Ҫת�����ַ����򲼶�ֵ��

    ����:
        bool: ת����Ĳ���ֵ��

    �쳣:
        argparse.ArgumentTypeError: ���������ַ���������Ч�Ĳ���ֵ��ʾ��
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
