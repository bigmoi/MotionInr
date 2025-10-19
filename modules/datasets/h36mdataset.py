import os
import numpy as np
import torch
from .dataset import Dataset  # 假设你的 Dataset 类已经定义并导入
from .skeleton import Skeleton  # 假设你的 Skeleton 类已经定义并导入


class DatasetH36M(Dataset):
    def __init__(self, mode,data_path, t_his=25, t_pred=100, actions='all', use_vel=False):
        # 初始化父类
        self.use_vel = use_vel
        self.data_path = data_path
        if use_vel:
            self.traj_dim += 3  # 如果使用速度信息，增加维度
        super().__init__(mode, t_his, t_pred, actions)  # 调用父类的 __init__, 这会调用 prepare_data 方法


    def prepare_data(self):
        # 数据文件路径
        # self.data_file = os.path.join('data', 'data_3d_h36m.npz')
        if self.data_path==None:
            self.data_file = 'E:\MyProjects\MotionInr\data\data_3d_h36m.npz' #测试用硬编码
        else :
            self.data_file = self.data_path
        # 设置训练和测试的 subject 列表
        self.subjects_split = {'train': [1, 5, 6, 7, 8],
                               'test': [9, 11]}
        self.subjects = ['S%d' % x for x in self.subjects_split[self.mode]]

        # 初始化骨架（Skeleton）
        self.skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                          16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                                 joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                                 joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])

        # 定义需要删除的关节
        self.removed_joints = {4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}
        self.kept_joints = np.array([x for x in range(32) if x not in self.removed_joints])

        # 去除不需要的关节
        self.skeleton.remove_joints(self.removed_joints)
        self.skeleton._parents[11] = 8
        self.skeleton._parents[14] = 8

        # 处理数据
        self.process_data()

    def process_data(self):
        # 加载 3D 数据
        data_o = np.load(self.data_file, allow_pickle=True)['positions_3d'].item()

        # 从加载的数据中提取相关的关节
        self.S1_skeleton = data_o['S1']['Directions'][:1, self.kept_joints].copy()

        # 按 subject 筛选数据
        data_f = dict(filter(lambda x: x[0] in self.subjects, data_o.items()))

        # 如果指定了 action 列表，过滤出对应的动作
        if self.actions != 'all':
            for key in list(data_f.keys()):
                data_f[key] = dict(filter(lambda x: all([a in x[0] for a in self.actions]), data_f[key].items()))
                if len(data_f[key]) == 0:
                    data_f.pop(key)

        # 处理每个 subject 的动作数据
        for data_s in data_f.values():
            for action in data_s.keys():
                seq = data_s[action][:, self.kept_joints, :]
                if self.use_vel:
                    v = (np.diff(seq[:, :1], axis=0) * 50).clip(-5.0, 5.0)  # 计算速度
                    v = np.append(v, v[[-1]], axis=0)  # 保持最后一帧的速度
                seq[:, 1:] -= seq[:, :1]  # 坐标减去第一个关节（可能是根关节）
                if self.use_vel:
                    seq = np.concatenate((seq, v), axis=1)  # 如果使用速度信息，拼接速度数据
                data_s[action] = seq  # 更新数据
        self.data = data_f  # 保存数据

        # 计算数据集的大小
        self.all_samples = []  # 这个列表保存了所有样本的索引
        for subject, data_s in self.data.items():
            for action, seq in data_s.items():
                seq_len = seq.shape[0] - self.t_total   #指的是序列全长
                self.all_samples.extend([(subject, action, i) for i in range(seq_len)])

    def __len__(self):
        # 返回数据集的大小
        return len(self.all_samples)

    def __getitem__(self, idx):
        # 获取一个样本，保证 idx 不会超出范围
        subject, action, frame_idx = self.all_samples[idx]
        seq = self.data[subject][action]
        traj = seq[frame_idx:frame_idx + self.t_total]  # 截取对应的轨迹片段
        traj=traj[:,1:,:] #去掉根节点
        traj_=traj.reshape(traj.shape[0], -1)
        return torch.tensor(traj_, dtype=torch.float32)  # 返回一个样本


# 测试示例
if __name__ == '__main__':
    # 创建 DatasetH36M 实例
    dataset = DatasetH36M(mode='train', t_his=25, t_pred=100, actions='all', use_vel=False)

    # 查看数据集的长度
    print(f'Dataset size: {len(dataset)}')

    # 获取一个样本
    sample = dataset[0]
    print(f'Sample shape: {sample.shape}')

    # 使用 DataLoader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=8)

    # 打印每个批次的样本形状
    # batch_num=0
    # # for batch in dataloader:
    # #     batch_num+=1
    # #     print(f'Batch shape: {batch.shape}')
    # print(f'Batch size: {batch_num}')