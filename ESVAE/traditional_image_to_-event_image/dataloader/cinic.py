# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: cinic.py
@time: 2023/11/13 19:28
"""

import os
import cv2
import bisect
from collections import Counter
from torch._utils import _accumulate
from dataloader.dataloader_utils import *
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from spikingjelly.datasets import cifar10_dvs
from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Dataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler


# your own data dir
USER_NAME = 'zhan'
DIR = {'CINIC10': f'/data/{USER_NAME}/Event_Camera_Datasets/CINIC10/cinic10_without_cifar10',
       'CIFAR10DVS': f'/data/{USER_NAME}/Event_Camera_Datasets/CIFAR10/CIFAR10DVS/temporal_effecient_training_0.9_mat',
       'CIFAR10DVS_CATCH': f'/data/{USER_NAME}/Event_Camera_Datasets/CIFAR10/CIFAR10DVS_dst_cache',
       }


def get_tl_cinic10_wo_cifar10(batch_size, train_set_ratio=1.0, dvs_train_set_ratio=1.0):
    """
    get the train loader which yield rgb_img and dvs_img with label
    and test loader which yield dvs_img with label of cifar10.
    :return: train_loader, test_loader
    """
    rgb_trans_train = transforms.Compose([transforms.Resize(48),
                                          transforms.RandomCrop(48, padding=4),
                                          transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                          CIFAR10Policy(),  # TODO: 待注释
                                          transforms.ToTensor(),
                                          # transforms.RandomGrayscale(),  # 随机变为灰度图
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 归一化
                                          # transforms.Normalize((0., 0., 0.), (1, 1, 1)),
                                          # Cutout(n_holes=1, length=16)  # 随机挖n_holes个length * length的洞
                                          ])
    # rgb_trans_test = transforms.Compose([transforms.Resize(48),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dvs_trans = transforms.Compose([transforms.Resize((48, 48)),
                                    # transforms.RandomCrop(48, padding=4),
                                    # transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                    transforms.ToTensor(),
                                   ])

    tl_train_data = TLCINIC10_WO_CIFAR10(DIR['CINIC10'], DIR['CIFAR10DVS'], train=True, dvs_train_set_ratio=dvs_train_set_ratio,
                                         transform=rgb_trans_train, dvs_transform=dvs_trans, download=False)
    dvs_test_data = TLCINIC10_WO_CIFAR10(DIR['CINIC10'], DIR['CIFAR10DVS'], train=False, dvs_train_set_ratio=1.0,
                                         dvs_transform=dvs_trans, download=False)

    # take train set by train_set_ratio
    if train_set_ratio < 1.0:
        n_train = len(tl_train_data)  # 60000
        split = int(n_train * train_set_ratio)  # 60000*0.9 = 54000
        print(n_train, split)
        tl_train_data, _ = my_random_split(tl_train_data, [split, n_train-split], generator=torch.Generator().manual_seed(1000))

    train_dataloader = DataLoaderX(tl_train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True,
                                   pin_memory=True)
    test_dataloader = DataLoaderX(dvs_test_data, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False,
                                  pin_memory=True)

    return train_dataloader, test_dataloader


class TLCINIC10_WO_CIFAR10(datasets.VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(
            self,
            root: str,
            dvs_root: str,
            train: bool = True,
            rgb_train_set_ratio: float = 1.0,
            dvs_train_set_ratio: float = 1.0,
            transform: Optional[Callable] = None,
            dvs_transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(TLCINIC10_WO_CIFAR10, self).__init__(root=root, transform=transform,
                                                   target_transform=target_transform)

        self.train = train  # training set or test set
        self.rgb_train_set_ratio = rgb_train_set_ratio
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
        self.class_to_idx = {}
        self.categories = sorted(os.listdir(os.path.join(self.root, "train")))
        self.rgb_data = []
        self.targets = []

        for (i, c) in enumerate(self.categories):
            # print(i, c)
            file_list = sorted(os.listdir(os.path.join(self.root, "train", c)))
            file_range = int(self.rgb_train_set_ratio * len(file_list))
            for file_name in file_list[: file_range]:
                self.rgb_data.append(os.path.join(self.root, "train", c, file_name))
            self.targets.extend(file_range * [i])
            self.class_to_idx[c] = i

        self.cumulative_sizes = self.cumsum(self.targets)

        """
        准备DVS数据
        """
        dvs_class_list = sorted(os.listdir(self.dvs_root))
        self.dvs_data = []
        self.dvs_targets = []

        for dvs_class in self.categories:
            # print(dvs_class)
            dvs_class_path = os.path.join(self.dvs_root, dvs_class)
            file_list = sorted(os.listdir(dvs_class_path))
            file_range = int(self.dvs_train_set_ratio * len(file_list))
            for file_name in file_list[: file_range]:
                self.dvs_data.append(os.path.join(dvs_class_path, file_name))
            self.dvs_targets.extend([self.class_to_idx[dvs_class]] * file_range)

        # print("准备DVS数据", len(self.dvs_data), len(self.dvs_targets))
        self.dvs_cumulative_sizes = self.cumsum(self.dvs_targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img_path, target = self.rgb_data[index], self.targets[index]

            img = Image.open(img_path)
            if len(img.split()) != 3:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                cv2.imwrite(img_path, img)
                img = Image.open(img_path)

            if self.transform is not None:
                try:
                    img = self.transform(img)
                except:
                    print("!!!!!!!!!, ", img_path)

            if self.target_transform is not None:
                target = self.target_transform(target)

            # 获取dvs图像的索引
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)  # 输出索引对应rgb图像的类别+1
            dvs_index_start = self.dvs_cumulative_sizes[dataset_idx - 1]  # 得到该类别对应dvs图像的开始索引
            dvs_index_end = self.dvs_cumulative_sizes[dataset_idx]  # 得到该类别对应dvs图像的结束索引
            dvs_index = dvs_index_start + (index - self.cumulative_sizes[dataset_idx - 1]) % (
                int((dvs_index_end - dvs_index_start)))  # 利用求余，得到在该类别循环0次或多次后的最终索引，self.dvs_train_set_ratio可控制选取dvs图像的比例 * self.dvs_train_set_ratio

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
            return len(self.rgb_data)
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
        return len(self.rgb_data), len(self.dvs_data)


class MySubset(Subset):
    def get_len(self):
        return len(self.indices), len(self.dataset.dvs_data)


def my_random_split(dataset, lengths, generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = torch.randperm(sum(lengths), generator=generator).tolist()
    return [MySubset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]