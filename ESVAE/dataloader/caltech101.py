import os
import bisect
import torch
import random
from collections import Counter
from dataloader.dataloader_utils import *
from torchvision import datasets, transforms
from spikingjelly.datasets import n_caltech101
from torch.utils.data import Dataset, random_split
from typing import Any, Callable, Optional, Tuple, Union, List
from torch.utils.data.sampler import SubsetRandomSampler

# your own data dir
USER_NAME = 'zhan'
DIR = {'Caltech101': f'/home/user/kpm/kpm/Dataset/Caltech101/caltech101',
       'Caltech101DVS': f'/home/user/kpm/kpm/Dataset/Caltech101/n-caltech101',
       'Caltech101DVS_CATCH': f'/data/{USER_NAME}/Event_Camera_Datasets/Caltech101/NCaltech101_dst_cache'
       }


def get_tl_caltech101(batch_size, train_set_ratio=1.0, dvs_train_set_ratio=1.0):
    """
    get the train loader which yield rgb_img and dvs_img with label
    and test loader which yield dvs_img with label of caltech101.
    :return: train_loader, test_loader
    """
    rgb_trans_train = transforms.Compose([
                                          transforms.Resize((52, 52)),  # 先resize到稍大尺寸
                                          transforms.RandomCrop(48, padding=2),  # 随机裁剪到48x48
                                          transforms.RandomHorizontalFlip(p=0.5), # 概率50%水平翻转
                                          transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 随机的颜色调整
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet标准归一化
                                      ])
    # rgb_trans_test = transforms.Compose([transforms.Resize(48, 48),
    #                                      transforms.ToTensor(),
    #                                      #transforms.Lambda(lambda x: x.repeat(3,1,1)),
    #                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #                                      ])
    dvs_trans = transforms.Compose([transforms.Resize((48, 48)),
                                    # transforms.RandomCrop(48, padding=4),
                                    # transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                    transforms.ToTensor(),
                                   ])

    tl_train_data = TLCaltech101(DIR['Caltech101'], DIR['Caltech101DVS'], train=True, dvs_train_set_ratio=dvs_train_set_ratio,
                                 transform=rgb_trans_train, dvs_transform=dvs_trans, download=False)
    dvs_test_data = TLCaltech101(DIR['Caltech101'], DIR['Caltech101DVS'], train=False, dvs_train_set_ratio=1.0,
                                 dvs_transform=dvs_trans, download=False)

    # take train set by train_set_ratio
    if train_set_ratio < 1.0:
        n_train = len(tl_train_data)  # 60000
        split = int(n_train * train_set_ratio)  # 60000*0.9 = 54000
        tl_train_data, _ = random_split(tl_train_data, [split, n_train-split], generator=torch.Generator().manual_seed(1000))

    train_dataloader = DataLoaderX(tl_train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True,
                                   pin_memory=True)
    test_dataloader = DataLoaderX(dvs_test_data, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False,
                                  pin_memory=True)

    return train_dataloader, test_dataloader


def get_caltech101(batch_size, train_set_ratio=1.0):
    # 自定义变换类来处理灰度图像转RGB
    class GrayscaleToRGB:
        def __call__(self, img):
            if img.mode == 'L':  # 灰度图像
                img = img.convert('RGB')  # 转换为RGB
            return img
    
    trans_train = transforms.Compose([
                                      GrayscaleToRGB(),  # 确保所有图像都是RGB格式
                                      transforms.Resize((56, 56)),
                                      transforms.RandomHorizontalFlip(p=0.5), # 概率50%水平翻转
                                      transforms.RandomRotation((-15,15)), # 随机旋转，角度范围为 -15° – 15°
                                      transforms.ColorJitter(), # 随机的颜色调整
                                      transforms.Resize((48, 48)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5429, 0.5263, 0.4994), (0.2422, 0.2392, 0.2406)),  # RGB归一化
                                      ])
    trans_test = transforms.Compose([GrayscaleToRGB(),  # 确保所有图像都是RGB格式
                                     transforms.Resize((48, 48)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) 

    # 使用本地准备好的数据集
    # 数据集路径: /home/user/kpm/kpm/Dataset/Caltech101/caltech-101
    # PyTorch期望的结构: root/101_ObjectCategories/类别名/图片
    caltech101_root = os.path.dirname(DIR['Caltech101'])  # /home/user/kpm/kpm/Dataset/Caltech101
    
    # 检查本地数据集是否存在
    if not os.path.exists(DIR['Caltech101']):
        raise FileNotFoundError(f"数据集目录不存在: {DIR['Caltech101']}")
    
    # 使用本地数据集，不下载
    train_data = datasets.Caltech101(caltech101_root, transform=trans_train, download=False)
    test_data = datasets.Caltech101(caltech101_root, transform=trans_test, download=False) 

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

def get_n_caltech101(batch_size,T,split_ratio=0.9,train_set_ratio=1.0,size=224,encode_type='TET'):
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
        train_set = NCaltech101(root=train_path)
        test_set = NCaltech101(root=test_path)
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
    train_data_loader = DataLoaderX(dataset=train_set, batch_size=batch_size, shuffle=False, drop_last=True,
                                    sampler=train_sampler, num_workers=8,
                                    pin_memory=True)  # SubsetRandomSampler 自带shuffle，不能重复使用
    test_data_loader = DataLoaderX(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=False,
                                   num_workers=8, pin_memory=True)

    return train_data_loader, test_data_loader


class NCaltech101(Dataset):
    # This code is form https://github.com/Gus-Lab/temporal_efficient_training
    def __init__(self, root, train=True, transform=True, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(48, 48))  # 128 128
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()

    def __getitem__(self, index):
        
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}_np.pt'.format(index))
        # if self.train:
        new_data = []
        for t in range(data.size(0)):
            new_data.append(self.tensorx(self.resize(self.imgx(data[t, ...]))))
        
        data = torch.stack(new_data, dim=0)
        
        if self.transform:
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

        # 初始化RGB数据（RGB数据没有train/test划分，总是加载全部）
        self._load_rgb_data()
        
        # 初始化DVS数据（根据train/test模式加载对应的DVS数据）
        self._load_dvs_data()

    def _load_rgb_data(self):
        """
        加载RGB数据
        注意：RGB数据没有train/test划分，无论在训练还是测试模式都加载全部RGB数据
        在训练模式下，RGB数据会与DVS的train数据配对
        在测试模式下，只使用DVS的test数据（不使用RGB数据）
        """
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"RGB数据目录不存在: {self.root}")
        
        # 获取所有类别目录，排除BACKGROUND_Google
        all_dirs = [d for d in os.listdir(self.root) 
                   if os.path.isdir(os.path.join(self.root, d))]
        
        # 过滤掉BACKGROUND_Google类别（Caltech101标准做法）
        self.categories = sorted([d for d in all_dirs 
                                if d != 'BACKGROUND_Google'])
        
        if len(self.categories) == 0:
            raise FileNotFoundError(f"在 {self.root} 中未找到有效类别目录")
        
        print(f"发现 {len(all_dirs)} 个目录，过滤后保留 {len(self.categories)} 个类别")
        
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
        
        print(f"✓ 加载RGB数据: {len(self.categories)} 个类别，{len(self.rgb_data)} 张图像")
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
                
                # 调试：打印前几个文件的数据形状
                if i < 3:
                    print(f"  DVS样本 {i}: 形状={data.shape}, 标签={target}, 数据类型={data.dtype}")
                    
            except Exception as e:
                print(f"警告: 无法加载DVS文件 {file_name}: {e}")
                continue
        
        print(f"✓ 加载DVS数据 ({mode}): {len(self.dvs_data)} 个样本")
        
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
        DVS数据应该是2通道 (正负极性)，形状为 (T, 2, H, W)
        """
        # 检查DVS数据形状并进行必要的重塑
        original_shape = dvs_img.shape
        
        # 如果形状不符合预期，尝试重塑
        if len(original_shape) == 4:
            T, C, H, W = original_shape
            # 如果通道数异常，可能需要重新整形
            if C > 4:  # 通道数异常大
                # 尝试重新整形为 (T, 2, H, W)
                total_elements = dvs_img.numel()
                # 假设T=10, C=2 (DVS标准)
                T_target = 10
                C_target = 2
                # 计算H和W
                remaining = total_elements // (T_target * C_target)
                H_target = W_target = int(remaining ** 0.5)
                
                try:
                    dvs_img = dvs_img.view(T_target, C_target, H_target, W_target)
                    # print(f"重塑DVS数据: {original_shape} -> {dvs_img.shape}")
                except:
                    # 如果重塑失败，创建默认的DVS数据
                    # print(f"警告: 无法重塑DVS数据 {original_shape}，使用默认数据")
                    dvs_img = torch.zeros(10, 2, 48, 48)
        
        # 现在应该是正确的形状 (T, 2, H, W)
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
    def __init__(self, root, transform=True, img_size=224):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.img_size = img_size
        self.resize = transforms.Resize(size=(img_size, img_size))
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        
        # 构建文件列表并排序
        self.files = self._build_file_list()
        print(f"Loaded {len(self.files)} samples from {root}")

    def _build_file_list(self):
        """构建并排序文件列表"""
        if not os.path.exists(self.root):
            return []
        
        files = [f for f in os.listdir(self.root) if f.endswith('.pt')]
        
        # 智能排序：提取数字部分
        def extract_number(filename):
            basename = filename.split('.')[0].replace('_np', '')
            try:
                return int(basename)
            except ValueError:
                return 0
        
        return sorted(files, key=extract_number)

    def __getitem__(self, index):
        file_path = os.path.join(self.root, self.files[index])
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
                processed_data = []
                for t in range(T):
                    frame = data[t]  # (C, H, W)
                    # 直接resize，不需要转换为PIL
                    frame_resized = torch.nn.functional.interpolate(
                        frame.unsqueeze(0), 
                        size=(self.img_size, self.img_size), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                    processed_data.append(frame_resized)
                data = torch.stack(processed_data, dim=0)
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
                    
                    # 然后resize到目标尺寸
                    processed_data = []
                    for t in range(T_target):
                        frame_resized = torch.nn.functional.interpolate(
                            data[t].unsqueeze(0), 
                            size=(self.img_size, self.img_size), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0)
                        processed_data.append(frame_resized)
                    data = torch.stack(processed_data, dim=0)
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
            if random.random() > 0.5:  # 随机水平翻转
                data = torch.flip(data, dims=(3,))
            # 随机平移
            off_x, off_y = random.randint(-5, 5), random.randint(-5, 5)
            data = torch.roll(data, shifts=(off_x, off_y), dims=(2, 3))
        
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(self.files)


def create_caltech101_dataloaders(data_path, batch_size, train_ratio=1.0, num_workers=8, img_size=224):
    """
    创建N-Caltech101数据加载器
    
    Args:
        data_path: 数据集根路径（包含train和test文件夹）
        batch_size: 批次大小
        train_ratio: 训练集使用比例
        num_workers: 数据加载线程数
        img_size: 图像尺寸
    
    Returns:
        train_loader, test_loader
    """
    import random
    from torch.utils.data.sampler import SubsetRandomSampler
    
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    
    # 验证路径
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(f"train/test directories not found in {data_path}")
    
    # 创建数据集
    train_dataset = NCaltech101Dataset(train_path, transform=True, img_size=img_size)
    test_dataset = NCaltech101Dataset(test_path, transform=False, img_size=img_size)
    
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
    
    # 创建数据加载器
    train_loader = DataLoaderX(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoaderX(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, test_loader