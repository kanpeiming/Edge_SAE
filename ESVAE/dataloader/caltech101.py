import os
import bisect
import torch
import random
from collections import Counter
from .dataloader_utils import (
    DataLoaderX, DVSResize, DVSAugment, DVSAugmentCaltech101, split_to_train_test_set
)
from torchvision import datasets, transforms
from spikingjelly.datasets import n_caltech101
from torch.utils.data import Dataset, random_split
from typing import Any, Callable, Optional, Tuple, Union, List
from torch.utils.data.sampler import SubsetRandomSampler

# your own data dir
USER_NAME = 'zhan'
DIR = {'Caltech101': f'/home/user/kpm/kpm/Dataset/Caltech101/caltech101/101_ObjectCategories',
       'Caltech101DVS': f'/home/user/kpm/kpm/Dataset/Caltech101/NCALTECH101/NCALTECH101/Caltech101',
       'Caltech101DVS_CATCH': f'/data/{USER_NAME}/Event_Camera_Datasets/Caltech101/NCaltech101_dst_cache'
       }


def get_tl_caltech101(batch_size, train_set_ratio=1.0, dvs_train_set_ratio=1.0, 
                      num_workers=8, img_size=48, split_ratio=0.9):
    """
    获取RGB到DVS迁移学习的数据加载器（现代版本）
    
    N-Caltech101标准: 保留BACKGROUND_Google，移除Faces/Faces_easy类
    最终101类 = 100个物体类 + 1个背景类(BACKGROUND_Google)
    
    重要修复:
    1. 使用自定义collate_fn处理RGB(3,H,W)和DVS(T,C,H,W)不同形状的批处理
    2. 过滤Faces/Faces_easy类并重新映射标签到[0,100]范围
    3. 确保RGB和DVS数据集的类别和标签一致
    
    Args:
        batch_size: 批次大小
        train_set_ratio: RGB训练集使用比例
        dvs_train_set_ratio: DVS训练集使用比例
        num_workers: 数据加载线程数
        img_size: 图像尺寸
        split_ratio: DVS数据自动划分比例（如果没有train/test目录）
    
    Returns:
        train_loader: 训练数据加载器（返回RGB和DVS配对）
        test_loader: 测试数据加载器（仅DVS）
    """
    import random
    from torch.utils.data.sampler import SubsetRandomSampler
    
    print("\n加载Caltech101 RGB到DVS迁移学习数据集...")
    
    # 1. 加载RGB数据集（自动过滤Faces类）
    print("  加载RGB数据...")
    rgb_trans_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5429, 0.5263, 0.4994), (0.2422, 0.2392, 0.2406)),
    ])
    
    try:
        # 尝试使用Caltech101数据集
        caltech101_root = os.path.dirname(DIR['Caltech101'])
        rgb_dataset = datasets.Caltech101(caltech101_root, transform=rgb_trans_train, download=False)
        categories = rgb_dataset.categories
        
        # N-Caltech101标准：移除Faces类
        if 'Faces' in categories:
            faces_idx = categories.index('Faces')
            # 过滤掉Faces类的样本
            from torch.utils.data import Subset
            indices = [i for i, (_, label) in enumerate(rgb_dataset) if label != faces_idx]
            rgb_dataset = Subset(rgb_dataset, indices)
            print(f"  ✓ RGB: 移除Faces类，保留{len(categories)-1}类，{len(rgb_dataset)}样本")
        else:
            print(f"  ✓ RGB: {len(categories)}类，{len(rgb_dataset)}样本")
            
    except Exception as e:
        # 回退到ImageFolder
        rgb_dataset_raw = datasets.ImageFolder(DIR['Caltech101'], transform=rgb_trans_train)
        
        # N-Caltech101标准：保留BACKGROUND_Google，只移除Faces类（与Edge→DVS保持一致）
        # 最终应该是101类（100个物体类 + 1个BACKGROUND_Google）
        classes_to_remove = ['Faces']
        
        # 找出需要保留的类别并按名称排序（与DVS数据集保持一致）
        valid_classes = sorted([cls for cls in rgb_dataset_raw.classes if cls not in classes_to_remove])
        
        # 过滤样本并重新映射标签
        from torch.utils.data import Subset
        valid_indices = []
        label_mapping = {}  # 旧标签 -> 新标签
        
        # 按类别名称排序后重新分配标签（与DVS的sorted(subdirs)保持一致）
        for new_idx, class_name in enumerate(valid_classes):
            old_idx = rgb_dataset_raw.class_to_idx[class_name]
            label_mapping[old_idx] = new_idx
        
        for idx, (path, label) in enumerate(rgb_dataset_raw.samples):
            if label in label_mapping:
                valid_indices.append(idx)
        
        # 创建包装类来重新映射标签
        class RemappedDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, indices, label_mapping):
                self.dataset = dataset
                self.indices = indices
                self.label_mapping = label_mapping
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                real_idx = self.indices[idx]
                img, old_label = self.dataset[real_idx]
                new_label = self.label_mapping[old_label]
                return img, new_label
        
        rgb_dataset = RemappedDataset(rgb_dataset_raw, valid_indices, label_mapping)
        print(f"  ✓ RGB (ImageFolder): 移除Faces类，保留{len(valid_classes)}类（含BACKGROUND_Google），{len(rgb_dataset)}样本")
        print(f"  RGB标签映射: 按类别名称排序，标签范围 [0, {len(valid_classes)-1}]")
    
    # RGB采样
    if train_set_ratio < 1.0:
        n_rgb = len(rgb_dataset)
        rgb_indices = list(range(n_rgb))
        random.shuffle(rgb_indices)
        rgb_dataset = Subset(rgb_dataset, rgb_indices[:int(n_rgb * train_set_ratio)])
        print(f"  采样后RGB: {len(rgb_dataset)}样本")
    
    # 2. 加载DVS数据集（使用NCaltech101Dataset，参考baseline）
    print("  加载DVS数据...")
    dvs_root = DIR['Caltech101DVS']
    train_path = os.path.join(dvs_root, 'train')
    test_path = os.path.join(dvs_root, 'test')
    
    has_split = os.path.exists(train_path) and os.path.exists(test_path)
    
    if has_split:
        # 使用预先划分的train/test
        # 注意：DVS数据不过滤Faces类，保持原始101类结构
        dvs_train_dataset = NCaltech101Dataset(train_path, transform=True, img_size=img_size,
                                               use_nda=False, use_eventrpg=False, filter_faces=False)
        dvs_test_dataset = NCaltech101Dataset(test_path, transform=False, img_size=img_size,
                                              use_nda=False, use_eventrpg=False, filter_faces=False)
        print(f"  ✓ DVS训练集: {len(dvs_train_dataset)}样本")
        print(f"  ✓ DVS测试集: {len(dvs_test_dataset)}样本")
        
        # DVS训练集采样
        if dvs_train_set_ratio < 1.0:
            n_dvs = len(dvs_train_dataset)
            dvs_indices = list(range(n_dvs))
            random.shuffle(dvs_indices)
            dvs_train_indices = dvs_indices[:int(n_dvs * dvs_train_set_ratio)]
            # 创建采样后的子集
            from torch.utils.data import Subset
            dvs_train_dataset = Subset(dvs_train_dataset, dvs_train_indices)
            print(f"  采样后DVS训练集: {len(dvs_train_dataset)}样本")
    else:
        # 自动划分train/test
        print(f"  未找到train/test划分，自动划分（{split_ratio*100:.0f}%训练，{(1-split_ratio)*100:.0f}%测试）")
        # 注意：DVS数据不过滤Faces类，保持原始101类结构
        full_dataset = NCaltech101Dataset(dvs_root, transform=False, img_size=img_size,
                                         use_nda=False, use_eventrpg=False, filter_faces=False)
        
        # 按类别划分
        if hasattr(full_dataset, 'file_labels') and full_dataset.file_labels is not None:
            from collections import defaultdict
            samples_by_class = defaultdict(list)
            for idx, label in enumerate(full_dataset.file_labels):
                samples_by_class[label].append(idx)
            
            train_indices = []
            test_indices = []
            
            for label, indices in samples_by_class.items():
                random.shuffle(indices)
                split_point = int(len(indices) * split_ratio)
                train_indices.extend(indices[:split_point])
                test_indices.extend(indices[split_point:])
            
            random.shuffle(train_indices)
            random.shuffle(test_indices)
        else:
            all_indices = list(range(len(full_dataset)))
            random.shuffle(all_indices)
            split_point = int(len(all_indices) * split_ratio)
            train_indices = all_indices[:split_point]
            test_indices = all_indices[split_point:]
        
        # DVS采样
        if dvs_train_set_ratio < 1.0:
            n_use = int(len(train_indices) * dvs_train_set_ratio)
            train_indices = train_indices[:n_use]
        
        print(f"  ✓ DVS训练集: {len(train_indices)}样本")
        print(f"  ✓ DVS测试集: {len(test_indices)}样本")
        
        # 创建数据集
        # 注意：DVS数据不过滤Faces类，保持原始101类结构
        dvs_train_dataset = NCaltech101Dataset(dvs_root, transform=True, img_size=img_size,
                                               use_nda=False, use_eventrpg=False, filter_faces=False)
        dvs_test_dataset = NCaltech101Dataset(dvs_root, transform=False, img_size=img_size,
                                              use_nda=False, use_eventrpg=False, filter_faces=False)
        
        # 应用采样器
        from torch.utils.data import Subset
        dvs_train_dataset = Subset(dvs_train_dataset, train_indices)
        dvs_test_dataset = Subset(dvs_test_dataset, test_indices)
    
    # 3. 创建RGB2DVS训练数据集（使用与Edge2DVS相同的配对方式）
    class TLRGB2DVSDataset(Dataset):
        """RGB到DVS迁移学习数据集 - 使用取模配对（与Edge2DVS一致）"""
        def __init__(self, rgb_dataset, dvs_dataset):
            self.rgb_dataset = rgb_dataset
            self.dvs_dataset = dvs_dataset
        
        def __len__(self):
            return max(len(self.rgb_dataset), len(self.dvs_dataset))
        
        def __getitem__(self, index):
            # RGB数据 - 使用取模循环
            rgb_idx = index % len(self.rgb_dataset)
            rgb_data, rgb_label = self.rgb_dataset[rgb_idx]
            
            # DVS数据 - 使用取模循环
            dvs_idx = index % len(self.dvs_dataset)
            dvs_data, dvs_label = self.dvs_dataset[dvs_idx]
            
            return (rgb_data, dvs_data), (rgb_label, dvs_label)
        
        def get_len(self):
            return [len(self.rgb_dataset), len(self.dvs_dataset)]
    
    train_dataset = TLRGB2DVSDataset(rgb_dataset, dvs_train_dataset)
    
    # 自定义collate函数，处理RGB和DVS数据的批处理
    def tl_collate_fn(batch):
        """
        自定义collate函数，处理RGB-DVS配对数据
        batch: list of ((rgb_data, dvs_data), (rgb_label, dvs_label))
        返回格式与Edge2DVS一致：((rgb_batch, dvs_batch), (rgb_labels, dvs_labels))
        """
        rgb_batch = []
        dvs_batch = []
        rgb_label_batch = []
        dvs_label_batch = []
        
        for (rgb_data, dvs_data), (rgb_label, dvs_label) in batch:
            rgb_batch.append(rgb_data)
            dvs_batch.append(dvs_data)
            
            # 处理RGB标签
            if isinstance(rgb_label, torch.Tensor):
                rgb_label_batch.append(rgb_label.item() if rgb_label.numel() == 1 else rgb_label)
            else:
                rgb_label_batch.append(rgb_label)
            
            # 处理DVS标签
            if isinstance(dvs_label, torch.Tensor):
                dvs_label_batch.append(dvs_label.item() if dvs_label.numel() == 1 else dvs_label)
            else:
                dvs_label_batch.append(dvs_label)
        
        # 堆叠成batch
        rgb_batch = torch.stack(rgb_batch, dim=0)  # (B, 3, H, W)
        dvs_batch = torch.stack(dvs_batch, dim=0)  # (B, T, C, H, W)
        
        # 处理RGB标签
        if isinstance(rgb_label_batch[0], torch.Tensor) and rgb_label_batch[0].numel() > 1:
            rgb_label_batch = torch.stack(rgb_label_batch, dim=0)
        else:
            rgb_label_batch = torch.tensor(rgb_label_batch)  # (B,)
        
        # 处理DVS标签
        if isinstance(dvs_label_batch[0], torch.Tensor) and dvs_label_batch[0].numel() > 1:
            dvs_label_batch = torch.stack(dvs_label_batch, dim=0)
        else:
            dvs_label_batch = torch.tensor(dvs_label_batch)  # (B,)
        
        return (rgb_batch, dvs_batch), (rgb_label_batch, dvs_label_batch)
    
    # 4. 创建数据加载器
    train_loader = DataLoaderX(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=tl_collate_fn
    )
    
    test_loader = DataLoaderX(
        dvs_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True
    )
    
    # 显示配对信息
    print(f"\n最终数据集配对:")
    print(f"  RGB样本: {train_dataset.get_len()[0]}")
    print(f"  DVS样本: {train_dataset.get_len()[1]}")
    if train_dataset.get_len()[0] != train_dataset.get_len()[1]:
        print(f"  ⚠️  样本数不一致，训练时使用循环采样（取模）策略")
    print()
    
    return train_loader, test_loader


def get_caltech101(batch_size, train_set_ratio=1.0, img_size=48):
    """
    获取Caltech101 RGB数据加载器
    
    N-Caltech101标准: 保留BACKGROUND_Google，移除Faces类
    最终101类 = 100个物体类 + 1个背景类(BACKGROUND_Google)
    
    Args:
        batch_size: 批次大小
        train_set_ratio: 训练集使用比例
        img_size: 图像尺寸 (默认48)
    """
    # 自定义变换类来处理灰度图像转RGB
    class GrayscaleToRGB:
        def __call__(self, img):
            if img.mode == 'L':  # 灰度图像
                img = img.convert('RGB')  # 转换为RGB
            return img
    
    trans_train = transforms.Compose([
                                      GrayscaleToRGB(),  # 确保所有图像都是RGB格式
                                      # transforms.Resize((56, 56)),
                                      # transforms.RandomHorizontalFlip(p=0.5), # 概率50%水平翻转
                                      # transforms.RandomRotation((-15,15)), # 随机旋转，角度范围为 -15° – 15°
                                      # transforms.ColorJitter(), # 随机的颜色调整
                                      transforms.Resize((img_size, img_size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5429, 0.5263, 0.4994), (0.2422, 0.2392, 0.2406)),  # RGB归一化
                                      ])
    trans_test = transforms.Compose([GrayscaleToRGB(),  # 确保所有图像都是RGB格式
                                     transforms.Resize((img_size, img_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) 

    # 使用本地准备好的数据集
    caltech101_root = os.path.dirname(DIR['Caltech101'])  # /home/user/kpm/kpm/Dataset/Caltech101/caltech101
    
    # 检查本地数据集是否存在
    if not os.path.exists(DIR['Caltech101']):
        raise FileNotFoundError(f"数据集目录不存在: {DIR['Caltech101']}")
    
    # 检查并解压 Annotations.tar（torchvision 需要）
    annotations_dir = os.path.join(caltech101_root, 'Annotations')
    annotations_tar = os.path.join(caltech101_root, 'Annotations.tar')
    
    if not os.path.exists(annotations_dir) and os.path.exists(annotations_tar):
        # 静默解压 Annotations.tar
        import tarfile
        with tarfile.open(annotations_tar, 'r') as tar:
            tar.extractall(path=caltech101_root)
    
    # 优先尝试使用 torchvision 自带的 Caltech101 类
    # 如果因为 MD5 校验或其他原因被认为"损坏"，则回退到 ImageFolder 方式加载
    try:
        train_data = datasets.Caltech101(caltech101_root, transform=trans_train, download=False)
        test_data = datasets.Caltech101(caltech101_root, transform=trans_test, download=False)
    except RuntimeError:
        # 静默回退到 ImageFolder 加载方式
        # N-Caltech101标准: 保留BACKGROUND_Google，移除Faces类
        from torch.utils.data import Subset
        
        full_train_data = datasets.ImageFolder(DIR['Caltech101'], transform=trans_train)
        full_test_data = datasets.ImageFolder(DIR['Caltech101'], transform=trans_test)
        
        # 查找需要移除的Faces类索引（N-Caltech101标准）
        faces_idx = full_train_data.class_to_idx.get('Faces', 
                    full_train_data.class_to_idx.get('faces', -1))
        
        if faces_idx != -1:
            # 过滤掉Faces类别的样本（保留BACKGROUND_Google）
            train_indices = [i for i, (_, label) in enumerate(full_train_data.samples) if label != faces_idx]
            test_indices = [i for i, (_, label) in enumerate(full_test_data.samples) if label != faces_idx]
            
            train_data = Subset(full_train_data, train_indices)
            test_data = Subset(full_test_data, test_indices)
            
            # 重新映射标签：移除Faces类后重新编号
            class LabelRemapDataset(torch.utils.data.Dataset):
                def __init__(self, subset, removed_idx):
                    self.subset = subset
                    self.removed_idx = removed_idx
                
                def __getitem__(self, idx):
                    img, label = self.subset[idx]
                    # 如果标签大于removed_idx，减1以填补空缺
                    if label > self.removed_idx:
                        label = label - 1
                    return img, label
                
                def __len__(self):
                    return len(self.subset)
            
            train_data = LabelRemapDataset(train_data, faces_idx)
            test_data = LabelRemapDataset(test_data, faces_idx)
            print(f"✓ [N-Caltech101标准] 移除Faces类，保留BACKGROUND_Google，共101类")
        else:
            train_data = full_train_data
            test_data = full_test_data
            print(f"✓ [get_caltech101] 使用所有{len(full_train_data.classes)}个类别")

    # take train set by train_set_ratio
    n_train = len(train_data)
    split = int(n_train * train_set_ratio)
    indices = list(range(n_train))
    random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[:split])

    if train_set_ratio < 1.0:
        train_dataloader = DataLoaderX(train_data, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True,
                                       sampler=train_sampler, pin_memory=True)
    else:
        train_dataloader = DataLoaderX(train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True,
                                       pin_memory=True)
    test_dataloader = DataLoaderX(test_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False,
                                  pin_memory=True)

    return train_dataloader, test_dataloader


def get_caltech101_gray(batch_size, train_set_ratio=1.0):
    """
    获取Caltech101灰度图数据加载器（用于消融实验2）
    将RGB图像转换为灰度图，然后扩展为3通道（复制灰度值）以适配模型
    
    N-Caltech101标准: 保留BACKGROUND_Google，移除Faces类
    最终101类 = 100个物体类 + 1个背景类
    
    Args:
        batch_size: 批次大小
        train_set_ratio: 训练集使用比例
    
    Returns:
        train_dataloader, test_dataloader
    """
    # 自定义变换类：将图像转为灰度图，然后扩展为3通道
    class GrayscaleToThreeChannel:
        def __call__(self, img):
            # 转换为灰度图
            if img.mode != 'L':
                img = img.convert('L')  # 转换为灰度图
            # 扩展为3通道（复制灰度值）
            img = img.convert('RGB')  # 将灰度图扩展为3通道RGB
            return img
    
    trans_train = transforms.Compose([
                                      GrayscaleToThreeChannel(),  # 转为灰度图并扩展为3通道
                                      transforms.Resize((56, 56)),
                                      transforms.RandomHorizontalFlip(p=0.5), # 概率50%水平翻转
                                      transforms.RandomRotation((-15,15)), # 随机旋转，角度范围为 -15° – 15°
                                      transforms.Resize((48, 48)),
                                      transforms.ToTensor(),
                                      # 使用灰度图的归一化参数（3个通道使用相同的值）
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
                                      ])
    trans_test = transforms.Compose([GrayscaleToThreeChannel(),  # 转为灰度图并扩展为3通道
                                     transforms.Resize((48, 48)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))]) 

    # 使用本地准备好的数据集
    caltech101_root = os.path.dirname(DIR['Caltech101'])  # /home/user/kpm/kpm/Dataset/Caltech101/caltech101
    
    # 检查本地数据集是否存在
    if not os.path.exists(DIR['Caltech101']):
        raise FileNotFoundError(f"数据集目录不存在: {DIR['Caltech101']}")
    
    # 检查并解压 Annotations.tar（torchvision 需要）
    annotations_dir = os.path.join(caltech101_root, 'Annotations')
    annotations_tar = os.path.join(caltech101_root, 'Annotations.tar')
    
    if not os.path.exists(annotations_dir) and os.path.exists(annotations_tar):
        # 静默解压 Annotations.tar
        import tarfile
        with tarfile.open(annotations_tar, 'r') as tar:
            tar.extractall(path=caltech101_root)
    
    # 优先尝试使用 torchvision 自带的 Caltech101 类
    # 如果因为 MD5 校验或其他原因被认为"损坏"，则回退到 ImageFolder 方式加载
    try:
        train_data = datasets.Caltech101(caltech101_root, transform=trans_train, download=False)
        test_data = datasets.Caltech101(caltech101_root, transform=trans_test, download=False)
    except RuntimeError:
        # 静默回退到 ImageFolder 加载方式
        # N-Caltech101标准: 保留BACKGROUND_Google，移除Faces类
        from torch.utils.data import Subset
        
        full_train_data = datasets.ImageFolder(DIR['Caltech101'], transform=trans_train)
        full_test_data = datasets.ImageFolder(DIR['Caltech101'], transform=trans_test)
        
        # 查找需要移除的Faces类索引（N-Caltech101标准）
        faces_idx = full_train_data.class_to_idx.get('Faces', 
                    full_train_data.class_to_idx.get('faces', -1))
        
        if faces_idx != -1:
            # 过滤掉Faces类别的样本（保留BACKGROUND_Google）
            train_indices = [i for i, (_, label) in enumerate(full_train_data.samples) if label != faces_idx]
            test_indices = [i for i, (_, label) in enumerate(full_test_data.samples) if label != faces_idx]
            
            train_data = Subset(full_train_data, train_indices)
            test_data = Subset(full_test_data, test_indices)
            
            # 重新映射标签：移除Faces类后重新编号
            class LabelRemapDataset(torch.utils.data.Dataset):
                def __init__(self, subset, removed_idx):
                    self.subset = subset
                    self.removed_idx = removed_idx
                
                def __getitem__(self, idx):
                    img, label = self.subset[idx]
                    # 如果标签大于removed_idx，减1以填补空缺
                    if label > self.removed_idx:
                        label = label - 1
                    return img, label
                
                def __len__(self):
                    return len(self.subset)
            
            train_data = LabelRemapDataset(train_data, faces_idx)
            test_data = LabelRemapDataset(test_data, faces_idx)
            print(f"✓ [N-Caltech101标准-灰度] 移除Faces类，保留BACKGROUND_Google，共101类")
        else:
            train_data = full_train_data
            test_data = full_test_data
            print(f"✓ [get_caltech101_gray] 使用所有{len(full_train_data.classes)}个类别") 

    # take train set by train_set_ratio
    n_train = len(train_data)
    split = int(n_train * train_set_ratio)
    indices = list(range(n_train))
    random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[:split])

    if train_set_ratio < 1.0:
        train_dataloader = DataLoaderX(train_data, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True,
                                       sampler=train_sampler, pin_memory=True)
    else:
        train_dataloader = DataLoaderX(train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True,
                                       pin_memory=True)
    test_dataloader = DataLoaderX(test_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False,
                                  pin_memory=True)

    return train_dataloader, test_dataloader


def get_n_caltech101(batch_size,T,split_ratio=0.9,train_set_ratio=1.0,size=224,encode_type='TET',use_eventrpg=False,eventrpg_mix_prob=0.5):
    if encode_type is "spikingjelly":

        trans = DVSResize((size, size), T)

        train_set_pth = os.path.join(DIR['Caltech101DVS_CATCH'], f'train_set_{T}_{split_ratio}_{size}.pt')
        test_set_pth = os.path.join(DIR['Caltech101DVS_CATCH'], f'test_set_{T}_{split_ratio}_{size}.pt')

        if os.path.exists(train_set_pth) and os.path.exists(test_set_pth):
            train_set = torch.load(train_set_pth)
            test_set = torch.load(test_set_pth)
        else:
            origin_set = n_caltech101.NCaltech101(root=DIR['Caltech101'], data_type='frame', frames_number=T,
                                                split_by='number', transform=trans)

            train_set, test_set = split_to_train_test_set(split_ratio, origin_set, 101 )
            if not os.path.exists(DIR['Caltech101DVS_CATCH']):
                os.makedirs(DIR['Caltech101DVS_CATCH'])
            torch.save(train_set, train_set_pth)
            torch.save(test_set, test_set_pth)
    elif encode_type is "TET":
        path = '/home/user/kpm/kpm/Dataset/Caltech101/n-caltech101'
        train_path = path + '/train'
        test_path = path + '/test'
        train_set = NCaltech101(root=train_path, train=True, transform=True, use_eventrpg=use_eventrpg, eventrpg_mix_prob=eventrpg_mix_prob)
        test_set = NCaltech101(root=test_path, train=False, transform=False, use_eventrpg=False)
    elif encode_type is "3_channel":
        pass

    # take train set by train_set_ratio
    n_train = len(train_set)
    split = int(n_train * train_set_ratio)
    indices = list(range(n_train))
    random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[:split])
    # valid_sampler = SubsetRandomSampler(indices[split:])

    # generate dataloader
    # train_data_loader = DataLoaderX(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True,
    #                                 num_workers=8, pin_memory=True)
    train_data_loader = DataLoaderX(dataset=train_set, batch_size=batch_size, shuffle=False, drop_last=False,
                                    sampler=train_sampler, num_workers=8,
                                    pin_memory=True)  # SubsetRandomSampler 自带shuffle，不能重复使用
    test_data_loader = DataLoaderX(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=False,
                                   num_workers=8, pin_memory=True)

    return train_data_loader, test_data_loader


class NCaltech101(Dataset):
    # This code is form https://github.com/Gus-Lab/temporal_efficient_training
    def __init__(self, root, train=True, transform=True, target_transform=None, use_eventrpg=False, eventrpg_mix_prob=0.5):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.use_eventrpg = use_eventrpg
        self.resize = transforms.Resize(size=(48, 48))  # 128 128
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()
        
        # 初始化EventRPG增强器
        if self.use_eventrpg:
            from .eventrpg_augment import EventRPGAugment
            self.eventrpg_augment = EventRPGAugment(img_size=48, mix_prob=eventrpg_mix_prob)

    def __getitem__(self, index):
        
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}_np.pt'.format(index))
        
        # 优化：批量resize，避免Tensor→PIL→Tensor转换
        # 直接使用torch.nn.functional.interpolate一次性处理所有时间步
        if data.shape[2] != 48 or data.shape[3] != 48:
            data = torch.nn.functional.interpolate(
                data.float(),  # (T, C, H, W)
                size=(48, 48),
                mode='bilinear',
                align_corners=False
            )
        
        if self.transform:
            if self.use_eventrpg:
                # 使用EventRPG的增强方法
                data = self.eventrpg_augment(data)
            else:
                # 使用传统增强方法
                flip = random.random() > 0.5
                if flip:
                    data = torch.flip(data, dims=(3,))
                off1 = random.randint(-5, 5)
                off2 = random.randint(-5, 5)
                data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
    
        if self.target_transform is not None:
            target = self.target_transform(target)
       
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))


class TLCaltech101(Dataset):
    """
    自定义Caltech101数据集类，支持RGB和DVS数据的配对加载
    不依赖torchvision的自动下载功能，直接从本地目录加载数据
    """

    def __init__(
            self,
            root: str,
            dvs_root: str,
            train: bool = True,
            dvs_train_set_ratio: float = 1.0,
            target_type: Union[List[str], str] = "category",
            transform: Optional[Callable] = None,
            dvs_transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,  # 保持兼容性，但不使用
    ) -> None:
        self.root = root
        self.train = train
        self.dvs_train_set_ratio = dvs_train_set_ratio
        self.dvs_transform = dvs_transform
        self.transform = transform
        self.target_transform = target_transform
        self.target_type = target_type if isinstance(target_type, list) else [target_type]
        self.imgx = transforms.ToPILImage()
        
        # DVS数据根路径
        self.dvs_base_root = dvs_root
        
        # 设置DVS数据路径（根据train/test模式）
        if self.train:
            self.dvs_root = os.path.join(dvs_root, 'train')
        else:
            self.dvs_root = os.path.join(dvs_root, 'test')

        # 初始化RGB数据（仅在训练模式下加载，测试模式只用DVS）
        if self.train:
            self._load_rgb_data()
        
        # 初始化DVS数据（根据train/test模式加载对应的DVS数据）
        self._load_dvs_data()

    def _load_rgb_data(self):
        """
        加载RGB数据
        注意：RGB数据没有train/test划分，无论在训练还是测试模式都加载全部RGB数据
        在训练模式下，RGB数据会与DVS的train数据配对
        在测试模式下，只使用DVS的test数据（不使用RGB数据）
        
        N-Caltech101标准: 保留BACKGROUND_Google，移除Faces类
        最终101类 = 100个物体类 + 1个背景类
        """
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"RGB数据目录不存在: {self.root}")
        
        # 获取所有类别目录
        all_dirs = [d for d in os.listdir(self.root) 
                   if os.path.isdir(os.path.join(self.root, d))]
        
        # N-Caltech101标准: 保留BACKGROUND_Google，移除Faces类
        # 原始Caltech101有102类，N-Caltech101保留101类（包含BACKGROUND_Google，移除Faces）
        self.categories = sorted([d for d in all_dirs 
                                if d != 'Faces' and d != 'faces'])
        
        if len(self.categories) == 0:
            raise FileNotFoundError(f"在 {self.root} 中未找到有效类别目录")
        
        self.rgb_data = []
        self.y = []
        self.index = []
        
        for class_idx, category in enumerate(self.categories):
            category_path = os.path.join(self.root, category)
            # 查找image_开头的jpg文件
            image_files = sorted([f for f in os.listdir(category_path) 
                                if f.startswith('image_') and f.lower().endswith(('.jpg', '.jpeg'))])
            
            for img_idx, img_file in enumerate(image_files, 1):
                self.rgb_data.append(os.path.join(category_path, img_file))
                self.y.append(class_idx)
                self.index.append(img_idx)
        
        print(f"✓ RGB [N-Caltech101标准]: {len(self.categories)} 类 (保留BACKGROUND_Google，移除Faces), {len(self.rgb_data)} 样本")
        self.cumulative_sizes = self.cumsum(self.y)

    def _load_dvs_data(self):
        """
        加载DVS数据
        DVS数据结构：train/test目录下直接是数字命名的.pt文件
        - 训练模式：加载 dvs_root/train/ 下的数据
        - 测试模式：加载 dvs_root/test/ 下的数据
        
        注意：DVS文件没有按类别分目录，需要从文件内容或文件名推断类别
        """
        if not os.path.exists(self.dvs_root):
            raise FileNotFoundError(f"DVS数据目录不存在: {self.dvs_root}")
        
        mode = "训练" if self.train else "测试"
        
        # 直接从train/test目录加载所有.pt文件
        pt_files = [f for f in os.listdir(self.dvs_root) if f.endswith('.pt')]
        
        # 按数字排序
        def extract_number(filename):
            try:
                return int(filename.split('.')[0])
            except ValueError:
                return 0
        
        pt_files = sorted(pt_files, key=extract_number)
        
        self.dvs_data = []
        self.dvs_targets = []
        
        # 加载DVS文件并从文件内容获取标签
        for i, file_name in enumerate(pt_files):
            file_path = os.path.join(self.dvs_root, file_name)
            try:
                # 加载DVS数据，获取标签
                data, target = torch.load(file_path, weights_only=True)
                self.dvs_data.append(file_path)
                # 确保标签是整数
                if isinstance(target, torch.Tensor):
                    target = target.item() if target.numel() == 1 else target[0].item()
                self.dvs_targets.append(int(target))
                
            except Exception as e:
                print(f"警告: 无法加载DVS文件 {file_name}: {e}")
                continue
        
        print(f"✓ DVS ({mode}): {len(self.dvs_data)} 样本")
        
        # 构建DVS的累积大小（按类别）
        self.dvs_cumulative_sizes = self.cumsum(self.dvs_targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        if self.train:
            # 加载RGB图像
            from PIL import Image
            img = Image.open(self.rgb_data[index])
            
            # 确保图像是RGB模式（3通道）
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 构建目标标签
            target: Any = []
            for t in self.target_type:
                if t == "category":
                    target.append(self.y[index])
                elif t == "annotation":
                    # 注释功能暂不支持，返回空列表
                    target.append([])
            target = tuple(target) if len(target) > 1 else target[0]

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            # 获取对应的DVS图像
            rgb_class = self.y[index]  # RGB图像的类别索引
            
            # 在DVS数据中找到相同类别的样本
            same_class_dvs_indices = [i for i, dvs_class in enumerate(self.dvs_targets) 
                                    if dvs_class == rgb_class]
            
            if same_class_dvs_indices:
                # 如果有同类别的DVS样本，循环选择
                dvs_choice_idx = index % len(same_class_dvs_indices)
                dvs_index = same_class_dvs_indices[dvs_choice_idx]
                
                # 加载DVS图像
                dvs_data, _ = torch.load(self.dvs_data[dvs_index], weights_only=True)
                if self.dvs_transform is not None:
                    dvs_img = self.dvs_trans(dvs_data)
                else:
                    dvs_img = dvs_data
            else:
                # 如果没有同类别的DVS数据，随机选择一个DVS样本
                if len(self.dvs_data) > 0:
                    dvs_index = index % len(self.dvs_data)
                    dvs_data, _ = torch.load(self.dvs_data[dvs_index], weights_only=True)
                    if self.dvs_transform is not None:
                        dvs_img = self.dvs_trans(dvs_data)
                    else:
                        dvs_img = dvs_data
                else:
                    # 创建默认的DVS tensor
                    dvs_img = torch.zeros(10, 2, 48, 48)

            return (img, dvs_img), target
        else:
            # 测试模式：只返回DVS图像
            if index < len(self.dvs_data):
                dvs_data, _ = torch.load(self.dvs_data[index], weights_only=True)
                if self.dvs_transform is not None:
                    dvs_img = self.dvs_trans(dvs_data)
                else:
                    dvs_img = dvs_data
                target = self.dvs_targets[index]
            else:
                # 如果索引超出范围，返回默认值
                dvs_img = torch.zeros(10, 2, 48, 48)
                target = 0

            return dvs_img, target

    def __len__(self) -> int:
        if self.train:
            return len(self.rgb_data)
        else:
            return len(self.dvs_data)

    def dvs_trans(self, dvs_img):
        """
        DVS数据变换
        DVS数据可能是 (C, H, W, T) 或 (T, C, H, W) 格式
        需要统一转换为 (T, C, H, W) 格式，其中 C=2 (正负极性)
        """
        # 检查DVS数据形状并进行必要的重塑
        original_shape = dvs_img.shape
        
        # 处理不同的输入格式
        if len(original_shape) == 4:
            # 判断数据格式：(C, H, W, T) 还是 (T, C, H, W)
            if original_shape[0] == 2 and original_shape[3] >= 10:
                # 格式是 (C=2, H, W, T)，需要转换为 (T, C, H, W)
                C, H, W, T = original_shape
                dvs_img = dvs_img.permute(3, 0, 1, 2)  # (C, H, W, T) -> (T, C, H, W)
                # print(f"转换DVS数据格式: {original_shape} -> {dvs_img.shape}")
            elif original_shape[1] == 2:
                # 格式已经是 (T, C=2, H, W)，无需转换
                T, C, H, W = original_shape
            else:
                # 通道数异常，尝试重新整形
                print(f"警告: DVS数据形状异常 {original_shape}，尝试重塑...")
                total_elements = dvs_img.numel()
                T_target = 10
                C_target = 2
                remaining = total_elements // (T_target * C_target)
                H_target = W_target = int(remaining ** 0.5)
                
                try:
                    dvs_img = dvs_img.view(T_target, C_target, H_target, W_target)
                    print(f"重塑DVS数据: {original_shape} -> {dvs_img.shape}")
                except:
                    print(f"错误: 无法重塑DVS数据 {original_shape}，使用默认数据")
                    dvs_img = torch.zeros(10, 2, 48, 48)
        
        # 现在应该是正确的形状 (T, C, H, W)
        T, C, H, W = dvs_img.shape
        
        # 对每个时间步进行变换
        transformed_dvs_img = []
        for t in range(T):
            frame = dvs_img[t]  # (2, H, W)
            
            # 直接使用tensor操作进行resize，避免PIL转换
            if H != 48 or W != 48:
                frame_resized = torch.nn.functional.interpolate(
                    frame.unsqueeze(0), 
                    size=(48, 48), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            else:
                frame_resized = frame
            
            # 转换为tensor (如果还不是)
            if not isinstance(frame_resized, torch.Tensor):
                frame_resized = torch.tensor(frame_resized, dtype=torch.float32)
            
            transformed_dvs_img.append(frame_resized)
        
        dvs_img = torch.stack(transformed_dvs_img, dim=0)

        # 数据增强（训练时）
        if self.train:
            flip = random.random() > 0.5
            if flip:
                dvs_img = torch.flip(dvs_img, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            dvs_img = torch.roll(dvs_img, shifts=(off1, off2), dims=(2, 3))
        
        return dvs_img

    @staticmethod
    def cumsum(targets):
        result = Counter(targets)
        r, s = [0], 0
        for e in range(len(result)):
            l = result[e]
            r.append(l + s)
            s += l
        return r

    def get_len(self):
        return len(self.rgb_data), len(self.dvs_data)


# ============================================================================
# 优化的N-Caltech101数据集类和数据加载器
# ============================================================================

class NCaltech101Dataset(Dataset):
    """
    优化的N-Caltech101数据集类
    支持灵活的文件命名格式和数据增强
    """
    def __init__(self, root, transform=True, img_size=224, use_nda=False, use_eventrpg=False, eventrpg_mix_prob=0.5, 
                 filter_faces=False):
        """
        Args:
            root: 数据根目录
            transform: 是否使用数据增强
            img_size: 图像尺寸
            use_nda: 是否使用NDA_SNN的数据增强方法
            use_eventrpg: 是否使用EventRPG的数据增强方法
            eventrpg_mix_prob: EventRPG的RPGMix概率
            filter_faces: 是否过滤Faces/Faces_easy类（用于RGB2DVS迁移学习）
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.img_size = img_size
        self.use_nda = use_nda
        self.use_eventrpg = use_eventrpg
        self.filter_faces = filter_faces
        self.resize = transforms.Resize(size=(img_size, img_size))
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        
        # 构建文件列表并排序
        self.files = self._build_file_list()
        print(f"Loaded {len(self.files)} samples from {root}")
        
        # 初始化NDA增强器
        if self.use_nda:
            self.nda_augment = DVSAugmentCaltech101(apply_prob=1.0)
        
        # 初始化EventRPG增强器
        if self.use_eventrpg:
            from .eventrpg_augment import EventRPGAugment
            self.eventrpg_augment = EventRPGAugment(img_size=img_size, mix_prob=eventrpg_mix_prob)

    def _build_file_list(self):
        """构建并排序文件列表（支持.pt和.bin格式）"""
        if not os.path.exists(self.root):
            return []
        
        # 检查是否为类别目录结构（如 /wild_cat/image_0016.bin）
        subdirs = [d for d in os.listdir(self.root) 
                   if os.path.isdir(os.path.join(self.root, d))]
        
        if subdirs:
            # 类别目录结构：遍历所有类别目录
            print(f"检测到类别目录结构，共 {len(subdirs)} 个类别")
            
            # 如果需要过滤Faces类
            if self.filter_faces:
                original_count = len(subdirs)
                subdirs = [d for d in subdirs if d.lower() not in ['faces', 'faces_easy']]
                filtered_count = original_count - len(subdirs)
                if filtered_count > 0:
                    print(f"  过滤掉 {filtered_count} 个Faces类（N-Caltech101标准）")
            
            files = []
            self.category_labels = {}  # 类别名到标签的映射
            self.file_labels = []  # 每个文件对应的标签
            
            # 对类别目录排序以确保一致的标签分配
            subdirs = sorted(subdirs)
            
            for label_idx, category_dir in enumerate(subdirs):
                self.category_labels[category_dir] = label_idx
                category_path = os.path.join(self.root, category_dir)
                
                # 查找该类别下的所有.bin或.pt文件
                category_files = [f for f in os.listdir(category_path) 
                                if f.endswith('.bin') or f.endswith('.pt')]
                
                for f in sorted(category_files):
                    files.append(os.path.join(category_dir, f))
                    self.file_labels.append(label_idx)
            
            print(f"从类别目录加载了 {len(files)} 个文件")
            if self.filter_faces and self.file_labels:
                print(f"  标签范围: [0, {max(self.file_labels)}]，共 {len(set(self.file_labels))} 个类别")
            return files
        else:
            # 扁平目录结构：直接在root下查找.pt文件
            files = [f for f in os.listdir(self.root) if f.endswith('.pt')]
            self.file_labels = None  # 标签从文件内容中读取
            
            # 智能排序：提取数字部分
            def extract_number(filename):
                basename = filename.split('.')[0].replace('_np', '')
                try:
                    return int(basename)
                except ValueError:
                    return 0
            
            return sorted(files, key=extract_number)

    def _load_bin_file(self, file_path):
        """
        加载.bin格式的事件数据
        N-Caltech101 .bin文件格式（官方标准）：每个事件占40 bits (5 bytes)
        
        Bit layout (40 bits):
        - bit 39-32: X address (8 bits, 0-255 pixels)
        - bit 31-24: Y address (8 bits, 0-255 pixels)
        - bit 23:    Polarity (1 bit, 0 for OFF, 1 for ON)
        - bit 22-0:  Timestamp (23 bits, in microseconds)
        
        Byte layout (5 bytes):
        - byte 0:    X address (8 bits)
        - byte 1:    Y address (8 bits)
        - byte 2:    Polarity (bit 7) + Timestamp[22:16] (bits 6-0)
        - byte 3:    Timestamp[15:8]
        - byte 4:    Timestamp[7:0]
        """
        import numpy as np
        
        # 读取二进制数据
        with open(file_path, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.uint8)
        
        # 检查数据长度是否为5的倍数
        if len(raw_data) % 5 != 0:
            print(f"警告: {file_path} 的大小 ({len(raw_data)} bytes) 不是5的倍数，可能存在损坏")
            raw_data = raw_data[:-(len(raw_data) % 5)]  # 截断到5的倍数
        
        num_events = len(raw_data) // 5
        
        if num_events == 0:
            print(f"警告: {file_path} 没有有效事件数据")
            # 返回空的事件帧
            return torch.zeros(10, 2, 180, 240, dtype=torch.float32)
        
        # 重塑为 (num_events, 5) 数组
        events = raw_data.reshape(-1, 5)
        
        # 解析坐标（8位，0-255）
        x = events[:, 0].astype(np.int32)  # X address (byte 0)
        y = events[:, 1].astype(np.int32)  # Y address (byte 1)
        
        # 解析极性和时间戳
        # byte 2: [polarity(1bit) | timestamp[22:16](7bits)]
        polarity = (events[:, 2] >> 7) & 1  # 取最高位作为极性
        
        # 重建23位时间戳
        # timestamp = byte2[6:0] << 16 | byte3 << 8 | byte4
        timestamp = (
            ((events[:, 2] & 0x7F).astype(np.int32) << 16) |  # 低7位左移16位
            (events[:, 3].astype(np.int32) << 8) |            # byte 3左移8位
            events[:, 4].astype(np.int32)                      # byte 4
        )
        
        # N-Caltech101 的分辨率是 240x180（宽x高）
        H_orig, W_orig = 180, 240
        T = 10  # 时间步数
        
        # 创建帧缓冲区 (T, C=2, H, W)
        frames = np.zeros((T, 2, H_orig, W_orig), dtype=np.float32)
        
        # 将时间戳归一化到[0, T-1]
        t_min, t_max = timestamp.min(), timestamp.max()
        if t_max > t_min:
            t_normalized = ((timestamp - t_min) / (t_max - t_min) * (T - 1)).astype(np.int32)
            t_normalized = np.clip(t_normalized, 0, T - 1)
        else:
            # 所有事件时间戳相同，放在第一帧
            t_normalized = np.zeros(num_events, dtype=np.int32)
        
        # 过滤无效坐标（超出分辨率范围的事件）
        valid_mask = (x >= 0) & (x < W_orig) & (y >= 0) & (y < H_orig)
        x = x[valid_mask]
        y = y[valid_mask]
        polarity = polarity[valid_mask]
        t_normalized = t_normalized[valid_mask]
        
        # 将事件累积到对应的帧和通道中
        # 使用向量化操作提高效率
        for t_idx in range(T):
            mask_t = (t_normalized == t_idx)
            if mask_t.any():
                x_t = x[mask_t]
                y_t = y[mask_t]
                pol_t = polarity[mask_t]
                
                # 分别累积正极性和负极性事件
                for pol in [0, 1]:
                    mask_pol = (pol_t == pol)
                    if mask_pol.any():
                        x_pol = x_t[mask_pol]
                        y_pol = y_t[mask_pol]
                        # 累积事件计数
                        np.add.at(frames[t_idx, pol], (y_pol, x_pol), 1.0)
        
        # 转换为torch tensor
        data = torch.from_numpy(frames).float()
        
        return data
    
    def __getitem__(self, index):
        file_path = os.path.join(self.root, self.files[index])
        
        # 根据文件扩展名选择加载方法
        if file_path.endswith('.bin'):
            # 加载.bin文件
            data = self._load_bin_file(file_path)
            
            # 如果使用类别目录结构，从file_labels获取标签
            if self.file_labels is not None:
                target = torch.tensor(self.file_labels[index])
            else:
                # 尝试从文件名推断类别（不推荐）
                print(f"警告: .bin文件缺少标签信息，使用默认标签0")
                target = torch.tensor(0)
        else:
            # 加载.pt文件
            data, target = torch.load(file_path, weights_only=True)
        
        # 调试信息：打印数据形状
        if index == 0:  # 只在第一个样本时打印
            print(f"Original data shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
        
        # 数据预处理：处理不同的数据格式
        if len(data.shape) == 4:  # 可能是 (T, C, H, W) 或 (C, H, W, T)
            # 检查哪个维度最可能是时间步
            if data.shape[3] <= 20:  # 最后一个维度较小，可能是时间步 (C, H, W, T)
                C, H, W, T = data.shape
                if index == 0:
                    print(f"4D data (C,H,W,T): C={C}, H={H}, W={W}, T={T}")
                # 转换为 (T, C, H, W)
                data = data.permute(3, 0, 1, 2)  # (C,H,W,T) -> (T,C,H,W)
                T, C, H, W = data.shape
            else:  # 标准格式 (T, C, H, W)
                T, C, H, W = data.shape
                if index == 0:
                    print(f"4D data (T,C,H,W): T={T}, C={C}, H={H}, W={W}")
            
            # 检查通道数是否合理
            if C <= 4:  # 正常的图像通道数
                # 优化：批量resize所有时间步，避免逐帧处理
                data = torch.nn.functional.interpolate(
                    data,  # (T, C, H, W)
                        size=(self.img_size, self.img_size), 
                        mode='bilinear', 
                        align_corners=False
                )
            else:
                # 可能是错误的维度排列，尝试重新整形
                if index == 0:
                    print(f"Unusual channel count: {C}, trying to reshape...")
                # 假设数据实际上是 (T*C, H, W) 或其他格式
                # 尝试将其重新整形为合理的格式
                total_elements = data.numel()
                # 假设目标格式是 (T, 2, H, W)，其中T=10, C=2
                T_target = 10
                C_target = 2
                H_target = int((total_elements / (T_target * C_target)) ** 0.5)
                W_target = H_target
                
                try:
                    data = data.view(T_target, C_target, H_target, W_target)
                    if index == 0:
                        print(f"Reshaped to: {data.shape}")
                    
                    # 优化：批量resize所有时间步
                    data = torch.nn.functional.interpolate(
                        data,  # (T, C, H, W)
                            size=(self.img_size, self.img_size), 
                            mode='bilinear', 
                            align_corners=False
                    )
                except:
                    # 如果重新整形失败，创建默认数据
                    if index == 0:
                        print("Reshape failed, using default data")
                    data = torch.zeros(10, 2, self.img_size, self.img_size)
        
        elif len(data.shape) == 3:  # (C, H, W) - 单帧
            if index == 0:
                print(f"3D data: C={data.shape[0]}, H={data.shape[1]}, W={data.shape[2]}")
            # 扩展为时间序列
            frame_resized = torch.nn.functional.interpolate(
                data.unsqueeze(0), 
                size=(self.img_size, self.img_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            # 复制为多个时间步
            data = frame_resized.unsqueeze(0).repeat(10, 1, 1, 1)
        
        else:
            if index == 0:
                print(f"Unexpected data shape: {data.shape}")
            # 创建默认数据
            data = torch.zeros(10, 2, self.img_size, self.img_size)
        
        # 确保数据类型正确
        data = data.float()
        
        # 数据增强
        if self.transform:
            if self.use_eventrpg:
                # 使用EventRPG的增强方法 (几何增强+RPGMix)
                data = self.eventrpg_augment(data)
            elif self.use_nda:
                # 使用NDA_SNN的增强方法
                data = self.nda_augment(data)
            else:
                # 使用传统增强方法
                if random.random() > 0.5:  # 随机水平翻转
                    data = torch.flip(data, dims=(3,))
                # 随机平移
                off_x, off_y = random.randint(-5, 5), random.randint(-5, 5)
                data = torch.roll(data, shifts=(off_x, off_y), dims=(2, 3))
        
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(self.files)


def create_caltech101_dataloaders(data_path, batch_size, train_ratio=1.0, num_workers=8, img_size=224, use_nda=False, use_eventrpg=False, eventrpg_mix_prob=0.5, split_ratio=0.9):
    """
    创建N-Caltech101数据加载器
    
    Args:
        data_path: 数据集根路径（包含train和test文件夹，或类别目录）
        batch_size: 批次大小
        train_ratio: 训练集使用比例
        num_workers: 数据加载线程数
        img_size: 图像尺寸
        use_nda: 是否使用NDA_SNN的数据增强方法
        use_eventrpg: 是否使用EventRPG的数据增强方法
        eventrpg_mix_prob: EventRPG的RPGMix概率
        split_ratio: 当没有train/test划分时，自动划分的训练集比例（默认0.9）
    
    Returns:
        train_loader, test_loader
    """
    import random
    from torch.utils.data.sampler import SubsetRandomSampler
    
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    
    # 检查是否存在train/test划分
    has_split = os.path.exists(train_path) and os.path.exists(test_path)
    
    if has_split:
        # 使用预先划分的train/test
        train_dataset = NCaltech101Dataset(train_path, transform=True, img_size=img_size, use_nda=use_nda, use_eventrpg=use_eventrpg, eventrpg_mix_prob=eventrpg_mix_prob)
        test_dataset = NCaltech101Dataset(test_path, transform=False, img_size=img_size, use_nda=False, use_eventrpg=False)
        
        print(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
        
        # 训练集采样
        if train_ratio < 1.0:
            n_train = len(train_dataset)
            indices = list(range(n_train))
            random.shuffle(indices)
            train_indices = indices[:int(n_train * train_ratio)]
            train_sampler = SubsetRandomSampler(train_indices)
            shuffle = False
        else:
            train_sampler = None
            shuffle = True
        
        test_sampler = None
    else:
        # 没有train/test划分，自动划分
        print(f"No train/test split found, auto-splitting with ratio {split_ratio}")
        
        # 加载整个数据集
        full_dataset = NCaltech101Dataset(data_path, transform=False, img_size=img_size, use_nda=False, use_eventrpg=False)
        
        # 按类别划分
        if hasattr(full_dataset, 'file_labels') and full_dataset.file_labels is not None:
            from collections import defaultdict
            samples_by_class = defaultdict(list)
            for idx, label in enumerate(full_dataset.file_labels):
                samples_by_class[label].append(idx)
            
            train_indices = []
            test_indices = []
            
            for label, indices in samples_by_class.items():
                random.shuffle(indices)
                split_point = int(len(indices) * split_ratio)
                train_indices.extend(indices[:split_point])
                test_indices.extend(indices[split_point:])
            
            random.shuffle(train_indices)
            random.shuffle(test_indices)
        else:
            # 没有类别信息，按顺序划分
            all_indices = list(range(len(full_dataset)))
            random.shuffle(all_indices)
            split_point = int(len(all_indices) * split_ratio)
            train_indices = all_indices[:split_point]
            test_indices = all_indices[split_point:]
        
        print(f"Dataset loaded: {len(train_indices)} train, {len(test_indices)} test samples")
        
        # 创建带数据增强的训练集和不带数据增强的测试集
        train_dataset = NCaltech101Dataset(data_path, transform=True, img_size=img_size, use_nda=use_nda, use_eventrpg=use_eventrpg, eventrpg_mix_prob=eventrpg_mix_prob)
        test_dataset = NCaltech101Dataset(data_path, transform=False, img_size=img_size, use_nda=False, use_eventrpg=False)
        
        # 使用采样器
        if train_ratio < 1.0:
            n_use = int(len(train_indices) * train_ratio)
            train_indices = train_indices[:n_use]
        
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        shuffle = False
    
    # 创建数据加载器
    # 优化配置：添加prefetch_factor和persistent_workers以减少数据加载停顿
    train_loader = DataLoaderX(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,  # 每个worker预取4个batch，减少等待时间
        persistent_workers=True if num_workers > 0 else False  # 保持worker进程，避免重复创建
    )
    
    test_loader = DataLoaderX(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=4,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, test_loader


# ============================================================================
# Edge to DVS Transfer Learning Dataset
# ============================================================================

class EdgeDataset(Dataset):
    """
    加载预处理的Edge数据（按类别组织）
    
    注意：遵循N-Caltech101标准，过滤掉Faces类并重新映射标签（与DVS数据集保持一致）
    
    标签映射说明：
    - Edge数据保存时使用原始Caltech101标签（可能包含Faces类）
    - N-Caltech101标准移除了Faces类
    - 需要重新映射标签以匹配DVS数据集
    """
    def __init__(self, root, sample_ratio=1.0):
        self.root = root
        self.samples = []
        self.class_to_idx = {}
        self.label_mapping = {}  # 原始标签 -> 新标签的映射
        
        # 第一遍：收集所有类别和对应的原始标签
        class_dirs = sorted(os.listdir(root))
        original_labels = []
        valid_classes = []
        
        for class_dir in class_dirs:
            # N-Caltech101标准：跳过Faces类
            if class_dir.lower() == 'faces':
                print(f"  跳过Faces类（N-Caltech101标准）")
                continue
            
            class_path = os.path.join(root, class_dir)
            if os.path.isdir(class_path):
                # 读取一个样本以获取原始标签
                pt_files = [f for f in os.listdir(class_path) if f.endswith('.pt')]
                if pt_files:
                    sample_file = os.path.join(class_path, pt_files[0])
                    _, original_label = torch.load(sample_file, weights_only=True)
                    if isinstance(original_label, torch.Tensor):
                        original_label = original_label.item()
                    
                    original_labels.append(original_label)
                    valid_classes.append(class_dir)
        
        # 创建标签映射：按原始标签排序后重新分配连续标签
        sorted_indices = sorted(range(len(original_labels)), key=lambda i: original_labels[i])
        for new_label, idx in enumerate(sorted_indices):
            old_label = original_labels[idx]
            self.label_mapping[old_label] = new_label
            self.class_to_idx[valid_classes[idx]] = new_label
        
        print(f"  标签重映射: {len(self.label_mapping)} 个类别")
        print(f"  标签范围: {min(self.label_mapping.values())} - {max(self.label_mapping.values())}")
        
        # 检查标签是否连续
        expected_labels = set(range(len(self.label_mapping)))
        actual_labels = set(self.label_mapping.values())
        if expected_labels != actual_labels:
            missing = expected_labels - actual_labels
            print(f"  ⚠️  警告: 标签不连续，缺失标签: {sorted(missing)}")
        
        # 第二遍：收集所有样本
        for class_dir in valid_classes:
            class_path = os.path.join(root, class_dir)
            pt_files = sorted([f for f in os.listdir(class_path) if f.endswith('.pt')])
            for filename in pt_files:
                filepath = os.path.join(class_path, filename)
                self.samples.append(filepath)
        
        # 采样
        if sample_ratio < 1.0:
            n_samples = int(len(self.samples) * sample_ratio)
            self.samples = self.samples[:n_samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        filepath = self.samples[index]
        # 加载原始数据和标签
        edge, original_label = torch.load(filepath, weights_only=True)  # (2, H, W), label
        
        # 确保标签是标量
        if isinstance(original_label, torch.Tensor):
            original_label = original_label.item()
        
        # 重新映射标签以匹配N-Caltech101标准
        remapped_label = self.label_mapping.get(original_label, original_label)
        
        return edge, remapped_label


class TLEdge2DVSDataset(Dataset):
    """Edge到DVS迁移学习数据集"""
    def __init__(self, edge_root, dvs_dataset, edge_ratio=1.0):
        self.edge_dataset = EdgeDataset(edge_root, edge_ratio)
        self.dvs_dataset = dvs_dataset
    
    def __len__(self):
        return max(len(self.edge_dataset), len(self.dvs_dataset))
    
    def __getitem__(self, index):
        # Edge数据
        edge_idx = index % len(self.edge_dataset)
        edge, edge_label = self.edge_dataset[edge_idx]
        
        # DVS数据
        dvs_idx = index % len(self.dvs_dataset)
        dvs_data, dvs_label = self.dvs_dataset[dvs_idx]
        
        return (edge, dvs_data), (edge_label, dvs_label)
    
    def get_len(self):
        return [len(self.edge_dataset), len(self.dvs_dataset)]


def get_edge2dvs_caltech101(batch_size, edge_root, dvs_root, 
                            edge_ratio=1.0, dvs_ratio=1.0, 
                            num_workers=8, img_size=48, split_ratio=0.9):
    """
    获取Edge到DVS迁移学习的数据加载器
    
    Args:
        batch_size: 批次大小
        edge_root: Edge数据根目录（按类别组织）
        dvs_root: DVS数据根目录（使用NCaltech101Dataset加载方式）
        edge_ratio: Edge数据使用比例
        dvs_ratio: DVS数据使用比例
        num_workers: 数据加载线程数
        img_size: 图像尺寸
        split_ratio: 当没有train/test划分时的训练集比例
    
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    import random
    from torch.utils.data.sampler import SubsetRandomSampler
    
    # 加载Edge数据集
    edge_dataset = EdgeDataset(edge_root, edge_ratio)
    print(f"✓ Edge训练集: {len(edge_dataset)} 样本")
    
    # 检查Edge数据集的标签范围
    if len(edge_dataset) > 0:
        sample_edge, sample_label = edge_dataset[0]
        print(f"  Edge数据形状示例: {sample_edge.shape}")
        print(f"  Edge标签示例: {sample_label}")
        
        # 采样检查标签范围 - 均匀采样以覆盖所有类别
        labels = []
        check_samples = min(100, len(edge_dataset))
        # 均匀采样整个数据集，而不是只采样前100个
        step = max(1, len(edge_dataset) // check_samples)
        for i in range(0, len(edge_dataset), step):
            _, label = edge_dataset[i]
            labels.append(label)
            if len(labels) >= check_samples:
                break
        print(f"  Edge标签范围（采样检查）: [{min(labels)}, {max(labels)}]")
        print(f"  Edge唯一标签数: {len(set(labels))}")
    
    # 加载DVS数据集（使用与baseline相同的方式）
    train_path = os.path.join(dvs_root, 'train')
    test_path = os.path.join(dvs_root, 'test')
    
    has_split = os.path.exists(train_path) and os.path.exists(test_path)
    
    if has_split:
        # 使用预先划分的train/test
        dvs_train_dataset = NCaltech101Dataset(train_path, transform=True, img_size=img_size, 
                                               use_nda=False, use_eventrpg=False)
        dvs_test_dataset = NCaltech101Dataset(test_path, transform=False, img_size=img_size, 
                                              use_nda=False, use_eventrpg=False)
        
        print(f"✓ DVS训练集: {len(dvs_train_dataset)} 样本")
        print(f"✓ DVS测试集: {len(dvs_test_dataset)} 样本")
        
        # DVS训练集采样
        if dvs_ratio < 1.0:
            n_dvs = len(dvs_train_dataset)
            indices = list(range(n_dvs))
            random.shuffle(indices)
            dvs_train_indices = indices[:int(n_dvs * dvs_ratio)]
            dvs_train_sampler = SubsetRandomSampler(dvs_train_indices)
        else:
            dvs_train_sampler = None
        
        # 创建Edge2DVS训练数据集
        train_dataset = TLEdge2DVSDataset(edge_root, dvs_train_dataset, edge_ratio)
        
        # 训练数据加载器
        train_loader = DataLoaderX(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=(dvs_train_sampler is None),
            sampler=None,  # TLEdge2DVSDataset自己处理采样
            num_workers=num_workers, 
            drop_last=True, 
            pin_memory=True
        )
        
        # 测试数据加载器（只用DVS数据）
        test_loader = DataLoaderX(
            dvs_test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers, 
            drop_last=False, 
            pin_memory=True
        )
    else:
        # 没有train/test划分，自动划分
        print(f"No train/test split found, auto-splitting with ratio {split_ratio}")
        
        full_dataset = NCaltech101Dataset(dvs_root, transform=False, img_size=img_size, 
                                         use_nda=False, use_eventrpg=False)
        
        # 按类别划分
        if hasattr(full_dataset, 'file_labels') and full_dataset.file_labels is not None:
            from collections import defaultdict
            samples_by_class = defaultdict(list)
            for idx, label in enumerate(full_dataset.file_labels):
                samples_by_class[label].append(idx)
            
            train_indices = []
            test_indices = []
            
            for label, indices in samples_by_class.items():
                random.shuffle(indices)
                split_point = int(len(indices) * split_ratio)
                train_indices.extend(indices[:split_point])
                test_indices.extend(indices[split_point:])
            
            random.shuffle(train_indices)
            random.shuffle(test_indices)
        else:
            all_indices = list(range(len(full_dataset)))
            random.shuffle(all_indices)
            split_point = int(len(all_indices) * split_ratio)
            train_indices = all_indices[:split_point]
            test_indices = all_indices[split_point:]
        
        # DVS采样
        if dvs_ratio < 1.0:
            n_use = int(len(train_indices) * dvs_ratio)
            train_indices = train_indices[:n_use]
        
        print(f"✓ DVS训练集: {len(train_indices)} 样本")
        print(f"✓ DVS测试集: {len(test_indices)} 样本")
        
        # 创建数据集
        dvs_train_dataset = NCaltech101Dataset(dvs_root, transform=True, img_size=img_size, 
                                               use_nda=False, use_eventrpg=False)
        dvs_test_dataset = NCaltech101Dataset(dvs_root, transform=False, img_size=img_size, 
                                              use_nda=False, use_eventrpg=False)
        
        # 创建Edge2DVS训练数据集
        train_dataset = TLEdge2DVSDataset(edge_root, dvs_train_dataset, edge_ratio)
        
        # 训练数据加载器
        train_loader = DataLoaderX(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers, 
            drop_last=True, 
            pin_memory=True
        )
        
        # 测试数据加载器
        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = DataLoaderX(
            dvs_test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            sampler=test_sampler,
            num_workers=num_workers, 
            drop_last=False, 
            pin_memory=True
        )
    
    # 显示最终配对信息
    print(f"\n最终数据集配对:")
    edge_len, dvs_len = train_loader.dataset.get_len()
    print(f"  Edge样本: {edge_len}")
    print(f"  DVS样本: {dvs_len}")
    if edge_len != dvs_len:
        print(f"  ⚠️  样本数不一致，训练时使用循环采样（取模）策略")
    
    return train_loader, test_loader


# ============================================================================
# (Stage 1 Bridge components removed)
# ============================================================================