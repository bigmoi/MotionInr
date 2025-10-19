import importlib

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import MSELoss
import copy


#由配置文件实例化对象
def instantiate_from_config(config, extra_args=None):
    if extra_args is not None:
        full_params = copy.deepcopy(config['params'])
        full_params.update(extra_args)
    else:
        full_params = config.get("params", dict())
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**full_params)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def get_dct_matrix(N, is_torch=True):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    if is_torch:
        dct_m = torch.from_numpy(dct_m)
        idct_m = torch.from_numpy(idct_m)
    return dct_m, idct_m

def generate_pad( t_his, t_pred,padding='LastFrame'):
    zero_index = None
    if padding == 'Zero':
        idx_pad = list(range(t_his)) + [t_his - 1] * t_pred
        zero_index = max(idx_pad)
    elif padding == 'Repeat':
        idx_pad = list(range(t_his)) * int(((t_pred + t_his) / t_his))
        # [0, 1, 2,....,24, 0, 1, 2,....,24, 0, 1, 2,...., 24...]
    elif padding == 'LastFrame':
        idx_pad = list(range(t_his)) + [t_his - 1] * t_pred
        # [0, 1, 2,....,24, 24, 24,.....]
    else:
        raise NotImplementedError(f"unknown padding method: {padding}")
    return idx_pad, zero_index

def padding_traj(traj,  idx_pad, zero_index,padding=None):
    #traj:输入序列 BT3j
    # padding:填充方式
    # idx_pad:填充索引   后两个参数由方法generate_pad生成 三种填充方式，后0，后重复，后最后一帧
    # zero_index:填充为0的索引
    if padding == 'Zero':
        traj_tmp = traj
        traj_tmp[..., zero_index, :] = 0
        traj_pad = traj_tmp[..., idx_pad, :]
    else:
        traj_pad = traj[..., idx_pad, :]

    return traj_pad

def sample_preprocessing(traj,dct_m_all,t_his=25,t_pred=100,n_pre=20, mod_test=1):
    """
    This function is used to preprocess traj for sample_ddim().
    input : traj_seq, cfg, mode
    output: a dict for specific mode,
            traj_dct,
            traj_dct_mod
    """
    n = traj.shape[0]

    mask = torch.zeros([n, t_his + t_pred, traj.shape[-1]]).to(traj.device)
    for i in range(0, t_his):
        mask[:, i, :] = 1 #将前t_his帧设为1 代表可视
    idx_pad, zero_index = generate_pad(t_his, t_pred, padding='LastFrame')
    traj_pad = padding_traj(traj, idx_pad, zero_index) #填充到全长

    traj_dct = torch.matmul(dct_m_all[:n_pre,], traj_pad) #填充序列做做dct变换
    traj_dct_mod = copy.deepcopy(traj_dct)
    if np.random.random() > mod_test: #随机drop条件，这里不使用，后续可以实验
        traj_dct_mod = None

    return {'mask': mask,
            'sample_num': n,
            'mode': 'metrics'}, traj_dct, traj_dct_mod

def map_to_original_frames(orig_fps: float, target_fps: float, orig_len: int) -> np.ndarray:
    """
    将目标帧率映射到原始帧率的帧索引列表
    :param orig_fps: 原始帧率
    :param target_fps: 目标帧率
    :param orig_len: 原始序列总帧数
    :return: 对应的原始帧索引列表
    """
    # 计算目标序列的长度
    target_len = int(orig_len * target_fps / orig_fps)

    # 创建目标帧索引
    target_indices = np.arange(target_len)

    # 计算每个目标帧在原始帧序列中的对应时间戳
    target_times = target_indices / target_fps  # 目标帧的时间戳

    # 将目标时间戳映射到原始时间戳（乘以原始帧率）
    orig_times = target_times * orig_fps

    # 根据原始时间戳计算对应的原始帧索引
    orig_indices = np.clip(np.round(orig_times).astype(int), 0, orig_len - 1)

    return orig_indices

if __name__ == "__main__":
    # 示例使用：假设原始帧率是50fps，目标帧率是100fps，原始序列长度为200帧
    orig_fps = 50  # 原始帧率
    target_fps = 100  # 目标帧率
    orig_len = 200  # 原始序列总帧数

    # 获取目标帧序列对应的原始帧索引
    orig_frame_indices = map_to_original_frames(orig_fps, target_fps, orig_len)
    print("对应的原始帧索引列表：", orig_frame_indices)
