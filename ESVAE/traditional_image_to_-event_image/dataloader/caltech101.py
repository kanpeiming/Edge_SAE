import os
import bisect
from collections import Counter
from dataloader.dataloader_utils import *
from torchvision import datasets, transforms
from spikingjelly.datasets import n_caltech101
from torch.utils.data import Dataset, random_split
from typing import Any, Callable, Optional, Tuple, Union, List
from torch.utils.data.sampler import SubsetRandomSampler

# your own data dir
USER_NAME = 'tr'
DIR = {'Caltech101': f'/home/user/{USER_NAME}/Dataset/Event_Camera_Datasets/Caltech101',
       'Caltech101DVS': f'/home/user/{USER_NAME}/Dataset/Event_Camera_Datasets/Caltech101/NCaltech101/temporal_effecient_training_0.9_np',
       'Caltech101DVS_CATCH': f'/home/user/{USER_NAME}/Dataset/Event_Camera_Datasets/Caltech101/NCaltech101_dst_cache'
       }


def get_tl_caltech101(batch_size, train_set_ratio=1.0, dvs_train_set_ratio=1.0):
    """
    get the train loader which yield rgb_img and dvs_img with label
    and test loader which yield dvs_img with label of caltech101.
    :return: train_loader, test_loader
    """
    rgb_trans_train = transforms.Compose([
                                          # transforms.Resize((56, 56)),
                                          # transforms.RandomHorizontalFlip(p=0.5), # 概率50%水平翻转
                                          # transforms.RandomRotation((-15,15)), # 随机旋转，角度范围为 -15° – 15°
                                          # transforms.ColorJitter(), # 随机的颜色调整
                                          transforms.Resize((48, 48)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5429, 0.5263, 0.4994), (0.2422, 0.2392, 0.2406)),  # 归一化
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
                                 transform=rgb_trans_train, dvs_transform=dvs_trans, download=True)
    dvs_test_data = TLCaltech101(DIR['Caltech101'], DIR['Caltech101DVS'], train=False, dvs_train_set_ratio=1.0,
                                 dvs_transform=dvs_trans, download=True)

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


def get_caltech101(batch_size,train_set_ratio=1.0):
    trans_train = transforms.Compose([transforms.Resize((56, 56)),
                                      transforms.RandomHorizontalFlip(p=0.5), # 概率50%水平翻转
                                      transforms.RandomRotation((-15,15)), # 随机旋转，角度范围为 -15° – 15°
                                      transforms.ColorJitter(), # 随机的颜色调整
                                      transforms.Resize((48, 48)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5429, 0.5263, 0.4994), (0.2422, 0.2392, 0.2406)),  # 归一化
                                      ])
    trans_test = transforms.Compose([transforms.Resize(48, 48),
                                     transforms.ToTensor(),
                                     #transforms.Lambda(lambda x: x.repeat(3,1,1)), 
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) 

    train_data = datasets.Caltech101(DIR['Caltech101'], transform=trans_train, download=True)
    test_data = datasets.Caltech101(DIR['Caltech101'], transform=trans_test, download=True) 

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
        path = '/data/zhan/Event_Camera_Datasets/Caltech101/NCaltech101/temporal_effecient_training_0.9_np'
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


class TLCaltech101(datasets.Caltech101):
    """`Caltech 101 <http://www.vision.caltech.edu/Image_Datasets/Caltech101/>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``caltech101`` exists or will be saved to if download is set to True.
        target_type (string or list, optional): Type of target to use, ``category`` or
        ``annotation``. Can also be a list to output a tuple with all specified target types.
        ``category`` represents the target class, and ``annotation`` is a list of points
        from a hand-generated outline. Defaults to ``category``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
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
            download: bool = False,
    ) -> None:
        super(TLCaltech101, self).__init__(root, target_type,
                                           transform=transform,
                                           target_transform=target_transform,
                                           download=download)
        self.train = train  # training set or test set
        self.dvs_train_set_ratio = dvs_train_set_ratio
        self.dvs_transform = dvs_transform
        self.imgx = transforms.ToPILImage()
        if self.train:
            self.dvs_root = os.path.join(dvs_root, 'train')
        else:
            self.dvs_root = os.path.join(dvs_root, 'test')

        """
        准备RGB数据
        """
        self.cumulative_sizes = self.cumsum(self.y)

        """
        准备DVS数据
        """
        self.dvs_data = []
        self.dvs_targets = []
        for i, dvs_class in enumerate(self.categories):
            dvs_class_path = os.path.join(self.dvs_root, dvs_class)
            file_list = sorted(os.listdir(dvs_class_path))
            for file_name in file_list:
                self.dvs_data.append(os.path.join(dvs_class_path, file_name))
                self.dvs_targets.append(i)

        self.dvs_cumulative_sizes = self.cumsum(self.dvs_targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        if self.train:
            import scipy.io

            img = Image.open(os.path.join(self.root,
                                          "101_ObjectCategories",
                                          self.categories[self.y[index]],
                                          "image_{:04d}.jpg".format(self.index[index])))

            target: Any = []
            for t in self.target_type:
                if t == "category":
                    target.append(self.y[index])
                elif t == "annotation":
                    data = scipy.io.loadmat(os.path.join(self.root,
                                                         "Annotations",
                                                         self.annotation_categories[self.y[index]],
                                                         "annotation_{:04d}.mat".format(self.index[index])))
                    target.append(data["obj_contour"])
            target = tuple(target) if len(target) > 1 else target[0]

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            # 获取dvs图像的索引
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)  # 输出索引对应rgb图像的类别+1
            dvs_index_start = self.dvs_cumulative_sizes[dataset_idx - 1]  # 得到该类别对应dvs图像的开始索引
            dvs_index_end = self.dvs_cumulative_sizes[dataset_idx]  # 得到该类别对应dvs图像的结束索引
            dvs_index = dvs_index_start + (index - self.cumulative_sizes[dataset_idx - 1]) % (
                int((dvs_index_end - dvs_index_start) * self.dvs_train_set_ratio))  # 利用求余，得到在该类别循环0次或多次后的最终索引，self.dvs_train_set_ratio可控制选取dvs图像的比例

            # dvs图像的transform
            dvs_img = torch.load(self.dvs_data[dvs_index])
            if self.dvs_transform is not None:
                dvs_img = self.dvs_trans(dvs_img)

            return (img, dvs_img), target
        else:
            # dvs图像的transform
            dvs_img = torch.load(self.dvs_data[index])
            if self.dvs_transform is not None:
                dvs_img = self.dvs_trans(dvs_img)
            target = self.dvs_targets[index]  # 输入索引对应dvs图像的类别

            return dvs_img, target

    def __len__(self) -> int:
        if self.train:
            return len(self.index)
        else:
            return len(self.dvs_data)

    def dvs_trans(self, dvs_img):
        transformed_dvs_img = []
        for t in range(dvs_img.size(0)):
            data = self.imgx(dvs_img[t, ...])
            transformed_dvs_img.append(self.dvs_transform(data))
        dvs_img = torch.stack(transformed_dvs_img, dim=0)

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
        return len(self.index), len(self.dvs_data)
