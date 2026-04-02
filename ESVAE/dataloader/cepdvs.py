# -*- coding: utf-8 -*-
"""
CEP-DVS Dataset Loader
CEP-DVS数据集加载器

数据集结构：
- RGB路径: /home/user/kpm/kpm/Dataset/CEP-DVS/data/img/*.jpg
- DVS路径: /home/user/kpm/kpm/Dataset/CEP-DVS/data/MAT/img/*.mat
- Edge路径: /home/user/kpm/kpm/Dataset/CEP-DVS/data/edge/*.pt
- pathFile.csv: /home/user/kpm/kpm/Dataset/CEP-DVS/pathFile.csv
- 样本数量: 10000个样本
- 类别数量: 20类 (CEP-DVS是20个类别)

配置说明：
请在本文件顶部的配置区域修改路径，所有训练脚本会自动使用这些配置。
"""

import os
import torch
import numpy as np
import scipy.io as scio
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .dataloader_utils import DataLoaderX


# ==================== CEP-DVS 配置区域 ====================
# 请根据您的实际路径修改以下配置
CEPDVS_ROOT = '/home/user/kpm/kpm/Dataset/CEP-DVS'  # CEP-DVS根目录（包含pathFile.csv）
CEPDVS_RGB_ROOT = '/home/user/kpm/kpm/Dataset/CEP-DVS/data/img'  # RGB图像目录
CEPDVS_DVS_ROOT = '/home/user/kpm/kpm/Dataset/CEP-DVS/data/MAT/img'  # DVS数据目录（原始.mat文件）
CEPDVS_DVS_PROCESSED_ROOT = '/home/user/kpm/kpm/Dataset/CEP-DVS/data/dvs_processed'  # DVS预处理数据目录（.pt文件）
CEPDVS_EDGE_ROOT = '/home/user/kpm/kpm/Dataset/CEP-DVS/data/edge'  # Edge数据目录
# =========================================================

# CEP-DVS数据集路径（兼容旧代码）
DIR = {
    'CEPDVS_ROOT': CEPDVS_ROOT,
    'CEPDVS_RGB': CEPDVS_RGB_ROOT,
    'CEPDVS_DVS': CEPDVS_DVS_ROOT,
    'CEPDVS_DVS_PROCESSED': CEPDVS_DVS_PROCESSED_ROOT,
    'CEPDVS_EDGE': CEPDVS_EDGE_ROOT
}

# CEP-DVS类别映射（20个类别）
CEPDVS_CLASSES = {
    'aquatic_mammals': 0,
    'fish': 1,
    'flowers': 2,
    'food_containers': 3,
    'fruit_and_vegetables': 4,
    'household_electrical_devices': 5,
    'household_furniture': 6,
    'insects': 7,
    'large_carnivores': 8,
    'large_man-made_outdoor_things': 9,
    'large_natural_outdoor_scenes': 10,
    'large_omnivores_and_herbivores': 11,
    'medium_mammals': 12,
    'non-insect_invertebrates': 13,
    'people': 14,
    'reptiles': 15,
    'small_mammals': 16,
    'trees': 17,
    'vehicles_1': 18,
    'vehicles_2': 19,
}


def load_pathfile_csv(csv_path):
    """
    加载pathFile.csv获取文件名到类别的映射
    
    Args:
        csv_path: pathFile.csv路径
    
    Returns:
        label_dict: {file_index: label} 的字典
    """
    import re
    label_dict = {}
    
    if not os.path.exists(csv_path):
        print(f"警告: pathFile.csv 不存在: {csv_path}")
        return label_dict
    
    record_file = np.genfromtxt(open(csv_path, "rb"), delimiter=",", skip_header=1, dtype='U')
    
    for i, row in enumerate(record_file):
        # 从CSV行中提取类别名称（格式：序号 空格 空格 空格 空格 类别名）
        match = re.search(r'(?:\S*\s){4}(\S+)', row)
        if match:
            label_name = match.group(1)
            if label_name in CEPDVS_CLASSES:
                label_dict[i] = CEPDVS_CLASSES[label_name]
    
    return label_dict


def load_mat_file(file_path):
    """
    加载.mat文件中的DVS事件数据
    
    Args:
        file_path: .mat文件路径
    
    Returns:
        event_dict: 包含 {'x', 'y', 'p', 'ts'} 的字典
    """
    data = scio.loadmat(file_path, verify_compressed_data_integrity=False)
    event_dict = {}
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            event_dict[key] = np.squeeze(data[key].astype(np.int64))
            # 将极性 -1 转换为 0 (标准化为0/1)
            if key == "p":
                event_dict[key] = np.where(event_dict[key] == -1, 0, event_dict[key])
    return event_dict


def events_to_frames(events, height=180, width=240, time_bins=10):
    """
    将DVS事件转换为时间序列帧
    
    Args:
        events: 事件字典，包含 {'x', 'y', 'p', 'ts'}
        height: 图像高度
        width: 图像宽度
        time_bins: 时间分箱数量
    
    Returns:
        frames: (time_bins, 2, H, W) 的张量，2通道分别代表正负极性
    """
    x = events['x']
    y = events['y']
    p = events['p']
    ts = events['ts']
    
    # 归一化时间戳到[0, time_bins-1]
    if len(ts) > 0:
        ts_min = ts.min()
        ts_max = ts.max()
        if ts_max > ts_min:
            ts_norm = ((ts - ts_min) / (ts_max - ts_min) * (time_bins - 1)).astype(np.int32)
        else:
            ts_norm = np.zeros_like(ts, dtype=np.int32)
    else:
        # 空事件：返回全零张量
        return torch.zeros((time_bins, 2, height, width), dtype=torch.float32)
    
    # 初始化帧
    frames = np.zeros((time_bins, 2, height, width), dtype=np.float32)
    
    # 填充事件到对应的时间帧和极性通道
    for i in range(len(x)):
        xi, yi, pi, ti = x[i], y[i], p[i], ts_norm[i]
        if 0 <= xi < width and 0 <= yi < height and 0 <= ti < time_bins:
            # pi=0 -> 通道0 (负极性), pi=1 -> 通道1 (正极性)
            frames[ti, pi, yi, xi] += 1
    
    # 归一化
    frames = frames / (frames.max() + 1e-8)
    
    return torch.from_numpy(frames).float()


class CEPDVSRGBDataset(Dataset):
    """
    CEP-DVS RGB数据集（使用pathFile.csv获取正确标签）
    """
    def __init__(self, root, csv_root=None, transform=None, sample_ratio=1.0):
        self.root = root
        self.transform = transform
        
        # 加载pathFile.csv获取标签映射
        if csv_root is None:
            csv_root = DIR['CEPDVS_ROOT']
        csv_path = os.path.join(csv_root, 'pathFile.csv')
        self.label_dict = load_pathfile_csv(csv_path)
        
        # 加载所有.jpg文件
        self.samples = []
        if os.path.exists(root):
            files = sorted([f for f in os.listdir(root) if f.endswith('.jpg')])
            
            # 使用pathFile.csv中的标签
            for f in files:
                file_path = os.path.join(root, f)
                # 从文件名提取索引（如：01891.jpg -> 1891）
                file_idx = int(f.split('.')[0])
                
                # 从label_dict获取标签
                if file_idx in self.label_dict:
                    label = self.label_dict[file_idx]
                    self.samples.append((file_path, label))
                else:
                    print(f"警告: 文件 {f} (索引={file_idx}) 在pathFile.csv中未找到标签，跳过")
        
        # 采样
        if sample_ratio < 1.0:
            num_samples = int(len(self.samples) * sample_ratio)
            self.samples = self.samples[:num_samples]
        
        print(f"✓ CEP-DVS RGB: {len(self.samples)} 样本，20类")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        file_path, label = self.samples[index]
        img = Image.open(file_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class CEPDVSDVSDataset(Dataset):
    """
    CEP-DVS DVS数据集（从.mat文件加载，使用pathFile.csv获取正确标签）
    """
    def __init__(self, root, csv_root=None, img_size=48, time_bins=10, sample_ratio=1.0):
        self.root = root
        self.img_size = img_size
        self.time_bins = time_bins
        
        # 加载pathFile.csv获取标签映射
        if csv_root is None:
            csv_root = DIR['CEPDVS_ROOT']
        csv_path = os.path.join(csv_root, 'pathFile.csv')
        self.label_dict = load_pathfile_csv(csv_path)
        
        # 加载所有.mat文件
        self.samples = []
        if os.path.exists(root):
            files = sorted([f for f in os.listdir(root) if f.endswith('.mat')])
            
            # 使用pathFile.csv中的标签
            for f in files:
                file_path = os.path.join(root, f)
                # 从文件名提取索引（如：01891.mat -> 1891）
                file_idx = int(f.split('.')[0])
                
                # 从label_dict获取标签
                if file_idx in self.label_dict:
                    label = self.label_dict[file_idx]
                    self.samples.append((file_path, label))
                else:
                    print(f"警告: 文件 {f} (索引={file_idx}) 在pathFile.csv中未找到标签，跳过")
        
        # 采样
        if sample_ratio < 1.0:
            num_samples = int(len(self.samples) * sample_ratio)
            self.samples = self.samples[:num_samples]
        
        print(f"✓ CEP-DVS DVS: {len(self.samples)} 样本，20类")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        file_path, label = self.samples[index]
        
        # 加载.mat文件
        events = load_mat_file(file_path)
        
        # 转换为帧序列 (T, 2, H, W)
        frames = events_to_frames(events, height=180, width=240, time_bins=self.time_bins)
        
        # Resize到目标尺寸
        if self.img_size != 180:
            # frames: (T, 2, 180, 240) -> (T, 2, img_size, img_size)
            T, C, H, W = frames.shape
            frames_resized = torch.zeros((T, C, self.img_size, self.img_size))
            for t in range(T):
                for c in range(C):
                    frame_pil = transforms.ToPILImage()(frames[t, c].unsqueeze(0))
                    frame_resized = transforms.Resize((self.img_size, self.img_size))(frame_pil)
                    frames_resized[t, c] = transforms.ToTensor()(frame_resized).squeeze(0)
            frames = frames_resized
        
        return frames, label


class CEPDVSDVSProcessedDataset(Dataset):
    """
    CEP-DVS预处理DVS数据集（加载预处理的.pt文件，速度快）
    """
    def __init__(self, root, csv_root=None, sample_ratio=1.0):
        self.root = root
        
        # 加载pathFile.csv获取标签映射（用于验证）
        if csv_root is None:
            csv_root = DIR['CEPDVS_ROOT']
        csv_path = os.path.join(csv_root, 'pathFile.csv')
        self.label_dict = load_pathfile_csv(csv_path)
        
        # 加载所有.pt文件
        self.samples = []
        if os.path.exists(root):
            files = sorted([f for f in os.listdir(root) if f.endswith('.pt')])
            
            for f in files:
                file_path = os.path.join(root, f)
                # 从文件名提取索引（如：01891.pt -> 1891）
                file_idx = int(f.split('.')[0])
                
                # 从label_dict获取标签（用于验证）
                if file_idx in self.label_dict:
                    label = self.label_dict[file_idx]
                    self.samples.append((file_path, label))
                else:
                    print(f"警告: 文件 {f} (索引={file_idx}) 在pathFile.csv中未找到标签，跳过")
        
        # 采样
        if sample_ratio < 1.0:
            num_samples = int(len(self.samples) * sample_ratio)
            self.samples = self.samples[:num_samples]
        
        print(f"✓ CEP-DVS DVS (预处理): {len(self.samples)} 样本，20类")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        file_path, label = self.samples[index]
        # 直接加载预处理的帧数据
        frames, _ = torch.load(file_path, weights_only=True)
        return frames, label


class TLCEPDVSDataset(Dataset):
    """
    CEP-DVS RGB到DVS迁移学习数据集
    """
    def __init__(self, rgb_dataset, dvs_dataset):
        self.rgb_dataset = rgb_dataset
        self.dvs_dataset = dvs_dataset
    
    def __len__(self):
        return max(len(self.rgb_dataset), len(self.dvs_dataset))
    
    def __getitem__(self, index):
        # RGB数据
        rgb_idx = index % len(self.rgb_dataset)
        rgb, rgb_label = self.rgb_dataset[rgb_idx]
        
        # DVS数据
        dvs_idx = index % len(self.dvs_dataset)
        dvs, dvs_label = self.dvs_dataset[dvs_idx]
        
        return (rgb, dvs), (rgb_label, dvs_label)


def get_tl_cepdvs(batch_size, train_set_ratio=1.0, dvs_train_set_ratio=1.0, 
                  num_workers=8, img_size=48, split_ratio=0.9, time_bins=10):
    """
    获取RGB到DVS迁移学习的CEP-DVS数据加载器
    
    Args:
        batch_size: 批次大小
        train_set_ratio: RGB训练集使用比例
        dvs_train_set_ratio: DVS训练集使用比例
        num_workers: 数据加载线程数
        img_size: 图像尺寸
        split_ratio: 训练/测试划分比例
        time_bins: DVS时间分箱数量
    
    Returns:
        train_loader: 训练数据加载器（返回RGB和DVS配对）
        test_loader: 测试数据加载器（仅DVS）
    """
    print("\n加载CEP-DVS RGB到DVS迁移学习数据集...")
    
    # RGB数据变换
    rgb_trans_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # 加载完整数据集
    rgb_full_dataset = CEPDVSRGBDataset(DIR['CEPDVS_RGB'], transform=rgb_trans_train, sample_ratio=1.0)
    dvs_full_dataset = CEPDVSDVSDataset(DIR['CEPDVS_DVS'], img_size=img_size, time_bins=time_bins, sample_ratio=1.0)
    
    # 划分训练/测试集
    total_size = len(rgb_full_dataset)
    train_size = int(total_size * split_ratio)
    test_size = total_size - train_size
    
    indices = list(range(total_size))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # RGB训练集采样
    if train_set_ratio < 1.0:
        num_train = int(len(train_indices) * train_set_ratio)
        train_indices = train_indices[:num_train]
    
    # DVS训练集采样
    dvs_train_indices = train_indices
    if dvs_train_set_ratio < 1.0:
        num_dvs_train = int(len(train_indices) * dvs_train_set_ratio)
        dvs_train_indices = train_indices[:num_dvs_train]
    
    print(f"  RGB训练集: {len(train_indices)} 样本")
    print(f"  DVS训练集: {len(dvs_train_indices)} 样本")
    print(f"  DVS测试集: {len(test_indices)} 样本")
    
    # 创建子集
    from torch.utils.data import Subset
    rgb_train_dataset = Subset(rgb_full_dataset, train_indices)
    dvs_train_dataset = Subset(dvs_full_dataset, dvs_train_indices)
    dvs_test_dataset = Subset(dvs_full_dataset, test_indices)
    
    # 创建迁移学习数据集
    train_dataset = TLCEPDVSDataset(rgb_train_dataset, dvs_train_dataset)
    
    # 自定义collate_fn处理不同形状的数据
    def collate_fn(batch):
        rgb_list, dvs_list, rgb_labels, dvs_labels = [], [], [], []
        for (rgb, dvs), (rgb_label, dvs_label) in batch:
            rgb_list.append(rgb)
            dvs_list.append(dvs)
            rgb_labels.append(rgb_label)
            dvs_labels.append(dvs_label)
        
        rgb_batch = torch.stack(rgb_list)
        dvs_batch = torch.stack(dvs_list)
        rgb_labels_batch = torch.tensor(rgb_labels)
        dvs_labels_batch = torch.tensor(dvs_labels)
        
        return (rgb_batch, dvs_batch), (rgb_labels_batch, dvs_labels_batch)
    
    # 训练数据加载器
    train_loader = DataLoaderX(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # 测试数据加载器（仅DVS）
    test_loader = DataLoaderX(
        dvs_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True
    )
    
    return train_loader, test_loader


class EdgeDatasetCEPDVS(Dataset):
    """
    加载预处理的CEP-DVS Edge数据（扁平结构，无类别目录，使用pathFile.csv获取正确标签）
    """
    def __init__(self, root, csv_root=None, sample_ratio=1.0):
        self.root = root
        
        # 加载pathFile.csv获取标签映射
        if csv_root is None:
            csv_root = DIR['CEPDVS_ROOT']
        csv_path = os.path.join(csv_root, 'pathFile.csv')
        self.label_dict = load_pathfile_csv(csv_path)
        
        self.samples = []
        
        if os.path.exists(root):
            files = sorted([f for f in os.listdir(root) if f.endswith('.pt')])
            
            # 使用pathFile.csv中的标签
            for f in files:
                file_path = os.path.join(root, f)
                # 从文件名提取索引（如：01891.pt -> 1891）
                file_idx = int(f.split('.')[0])
                
                # 从label_dict获取标签
                if file_idx in self.label_dict:
                    label = self.label_dict[file_idx]
                    self.samples.append((file_path, label))
                else:
                    print(f"警告: 文件 {f} (索引={file_idx}) 在pathFile.csv中未找到标签，跳过")
        
        # 采样
        if sample_ratio < 1.0:
            num_samples = int(len(self.samples) * sample_ratio)
            self.samples = self.samples[:num_samples]
        
        print(f"✓ CEP-DVS Edge: {len(self.samples)} 样本，20类")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        file_path, label = self.samples[index]
        # 加载预处理的边缘数据
        edge, _ = torch.load(file_path, weights_only=True)
        
        # 归一化处理：对Edge数据进行标准化
        if edge.dtype != torch.float32:
            edge = edge.float()
        
        # 对每个通道进行归一化（2通道：水平和垂直边缘）
        mean = torch.tensor([0.5, 0.5]).view(2, 1, 1)
        std = torch.tensor([0.5, 0.5]).view(2, 1, 1)
        edge = (edge - mean) / std
        
        return edge, label


class TLEdge2DVSDatasetCEPDVS(Dataset):
    """
    Edge到DVS迁移学习数据集（CEP-DVS）
    
    Args:
        edge_dataset: Edge数据集（可以是完整数据集或Subset）
        dvs_dataset: DVS数据集（可以是完整数据集或Subset）
    """
    def __init__(self, edge_dataset, dvs_dataset):
        self.edge_dataset = edge_dataset
        self.dvs_dataset = dvs_dataset
    
    def __len__(self):
        return max(len(self.edge_dataset), len(self.dvs_dataset))
    
    def __getitem__(self, index):
        # Edge数据（循环采样）
        edge_idx = index % len(self.edge_dataset)
        edge, edge_label = self.edge_dataset[edge_idx]
        
        # DVS数据（循环采样）
        dvs_idx = index % len(self.dvs_dataset)
        dvs_data, dvs_label = self.dvs_dataset[dvs_idx]
        
        return (edge, dvs_data), (edge_label, dvs_label)
    
    def get_len(self):
        return [len(self.edge_dataset), len(self.dvs_dataset)]


def get_edge2dvs_cepdvs(batch_size, edge_root, dvs_root, 
                        edge_ratio=1.0, dvs_ratio=1.0, 
                        num_workers=8, img_size=48, split_ratio=0.9, time_bins=10, use_processed=True):
    """
    获取Edge到DVS迁移学习的CEP-DVS数据加载器
    
    Args:
        batch_size: 批次大小
        edge_root: Edge数据根目录
        dvs_root: DVS数据根目录（可以是.mat文件目录或.pt文件目录）
        edge_ratio: Edge数据使用比例
        dvs_ratio: DVS数据使用比例
        num_workers: 数据加载线程数
        img_size: 图像尺寸
        split_ratio: 训练/测试划分比例
        time_bins: DVS时间分箱数量
        use_processed: 是否使用预处理数据（推荐True，速度快）
    
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    print("\n加载CEP-DVS Edge到DVS迁移学习数据集...")
    
    # 加载Edge数据集
    edge_full_dataset = EdgeDatasetCEPDVS(edge_root, csv_root=None, sample_ratio=1.0)
    
    # 加载DVS数据集（优先使用预处理数据）
    # 检查dvs_root是否指向预处理数据（.pt文件）
    if use_processed or (os.path.exists(dvs_root) and any(f.endswith('.pt') for f in os.listdir(dvs_root))):
        print(f"✓ 使用预处理DVS数据（.pt文件，速度快）")
        dvs_full_dataset = CEPDVSDVSProcessedDataset(dvs_root, csv_root=None, sample_ratio=1.0)
    else:
        print(f"⚠️  使用原始DVS数据（.mat文件，速度慢）")
        print(f"   建议运行: python preprocess_cepdvs_dvs.py")
        dvs_full_dataset = CEPDVSDVSDataset(dvs_root, img_size=img_size, time_bins=time_bins, sample_ratio=1.0)
    
    # 划分训练/测试集
    total_size = len(edge_full_dataset)
    train_size = int(total_size * split_ratio)
    test_size = total_size - train_size
    
    indices = list(range(total_size))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Edge训练集采样
    if edge_ratio < 1.0:
        num_train = int(len(train_indices) * edge_ratio)
        train_indices = train_indices[:num_train]
    
    # DVS训练集采样
    dvs_train_indices = train_indices
    if dvs_ratio < 1.0:
        num_dvs_train = int(len(train_indices) * dvs_ratio)
        dvs_train_indices = train_indices[:num_dvs_train]
    
    print(f"  Edge训练集: {len(train_indices)} 样本")
    print(f"  DVS训练集: {len(dvs_train_indices)} 样本")
    print(f"  DVS测试集: {len(test_indices)} 样本")
    
    # 创建子集
    from torch.utils.data import Subset
    edge_train_dataset = Subset(edge_full_dataset, train_indices)
    dvs_train_dataset = Subset(dvs_full_dataset, dvs_train_indices)
    dvs_test_dataset = Subset(dvs_full_dataset, test_indices)
    
    # 创建Edge2DVS训练数据集（直接传入已创建的数据集）
    train_dataset = TLEdge2DVSDatasetCEPDVS(edge_train_dataset, dvs_train_dataset)
    
    # 自定义collate_fn处理不同形状的数据
    def collate_fn(batch):
        edge_list, dvs_list, edge_labels, dvs_labels = [], [], [], []
        for (edge, dvs), (edge_label, dvs_label) in batch:
            edge_list.append(edge)
            dvs_list.append(dvs)
            edge_labels.append(edge_label)
            dvs_labels.append(dvs_label)
        
        edge_batch = torch.stack(edge_list)
        dvs_batch = torch.stack(dvs_list)
        edge_labels_batch = torch.tensor(edge_labels)
        dvs_labels_batch = torch.tensor(dvs_labels)
        
        return (edge_batch, dvs_batch), (edge_labels_batch, dvs_labels_batch)
    
    # 训练数据加载器
    train_loader = DataLoaderX(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # 测试数据加载器（仅DVS）
    test_loader = DataLoaderX(
        dvs_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True
    )
    
    print(f"\n最终数据集配对:")
    edge_len, dvs_len = train_dataset.get_len()
    print(f"  Edge样本: {edge_len}")
    print(f"  DVS样本: {dvs_len}")
    if edge_len != dvs_len:
        print(f"  ⚠️  样本数不一致，训练时使用循环采样（取模）策略")
    
    return train_loader, test_loader


def get_cepdvs(batch_size, train_set_ratio=1.0, img_size=48):
    """
    获取CEP-DVS RGB数据集（仅用于RGB->Edge预训练）
    
    Args:
        batch_size: 批次大小
        train_set_ratio: 训练集使用比例
        img_size: 图像尺寸
    
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    print("\n加载CEP-DVS RGB数据集（用于RGB->Edge预训练）...")
    
    # RGB数据变换
    rgb_trans = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # 加载完整数据集
    full_dataset = CEPDVSRGBDataset(DIR['CEPDVS_RGB'], transform=rgb_trans, sample_ratio=1.0)
    
    # 手动划分train/test (90% train, 10% test)
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    test_size = total_size - train_size
    
    # 使用固定种子确保可复现
    torch.manual_seed(1000)
    from torch.utils.data import random_split
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(1000)
    )
    
    print(f"  训练集: {len(train_dataset)} 样本 (90%)")
    print(f"  测试集: {len(test_dataset)} 样本 (10%)")
    
    # 根据train_set_ratio进一步采样训练集
    if train_set_ratio < 1.0:
        sampled_size = int(len(train_dataset) * train_set_ratio)
        train_indices = torch.randperm(len(train_dataset))[:sampled_size]
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, train_indices)
        print(f"  采样后训练集: {len(train_dataset)} 样本 (ratio={train_set_ratio})")
    
    # 创建数据加载器
    train_loader = DataLoaderX(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )
    
    test_loader = DataLoaderX(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_cepdvs_dvs(batch_size, train_set_ratio=1.0, img_size=48, time_bins=10, split_ratio=0.9, use_processed=True):
    """
    获取CEP-DVS DVS数据集（用于Baseline训练）
    
    Args:
        batch_size: 批次大小
        train_set_ratio: 训练集使用比例
        img_size: 图像尺寸
        time_bins: DVS时间分箱数量
        split_ratio: 训练/测试划分比例
        use_processed: 是否使用预处理数据（推荐True，速度快）
    
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    print("\n加载CEP-DVS DVS数据集（用于Baseline训练）...")
    
    # 优先使用预处理数据
    if use_processed and os.path.exists(DIR['CEPDVS_DVS_PROCESSED']):
        print(f"✓ 使用预处理DVS数据（速度快）: {DIR['CEPDVS_DVS_PROCESSED']}")
        full_dataset = CEPDVSDVSProcessedDataset(DIR['CEPDVS_DVS_PROCESSED'], sample_ratio=1.0)
    else:
        if use_processed:
            print(f"⚠️  预处理数据不存在，使用原始.mat文件（速度慢）")
            print(f"   建议运行: python preprocess_cepdvs_dvs.py")
        else:
            print(f"使用原始.mat文件（速度慢）")
        full_dataset = CEPDVSDVSDataset(DIR['CEPDVS_DVS'], img_size=img_size, time_bins=time_bins, sample_ratio=1.0)
    
    # 划分训练/测试集
    total_size = len(full_dataset)
    train_size = int(total_size * split_ratio)
    test_size = total_size - train_size
    
    indices = list(range(total_size))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # 训练集采样
    if train_set_ratio < 1.0:
        num_train = int(len(train_indices) * train_set_ratio)
        train_indices = train_indices[:num_train]
    
    print(f"  训练集: {len(train_indices)} 样本")
    print(f"  测试集: {len(test_indices)} 样本")
    
    # 创建子集
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # 创建数据加载器
    train_loader = DataLoaderX(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )
    
    test_loader = DataLoaderX(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        pin_memory=True
    )
    
    return train_loader, test_loader

