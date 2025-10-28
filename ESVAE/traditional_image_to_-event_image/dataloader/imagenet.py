# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: cifar.py
@time: 2022/4/19 11:19
"""
import multiprocessing
import os
import time
import psutil
import bisect
from collections import Counter
from torch._utils import _accumulate
from dataloader.dataloader_utils import *
from torch.utils.data.dataset import Subset
import torchvision
from torch import Tensor
from torchvision import datasets
from torchvision.transforms import transforms
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union
from torch.utils.data import Dataset, random_split
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
try:
    from torchvision.transforms import autoaugment
    from torchvision.transforms.functional import InterpolationMode
    from spikingjelly.activation_based.model.tv_ref_classify import transforms as jelly_transforms
except:
    pass


# your own data dir
USER_NAME = 'tr'
DIR = {'ImageNet100': f'/home/user/{USER_NAME}/Dataset/Event_Camera_Datasets/ImageNet100',
       'ImageNet100DVS': f'/home/user/{USER_NAME}/Dataset/Event_Camera_Datasets/ImageNet100/NImageNet100/temporal_effecient_training_1.0_np_10_timestep',
       'ImageNet100DVS_CATCH': f'/home/user/{USER_NAME}/Dataset/Event_Camera_Datasets/ImageNet100/NImageNet100_dst_cache'
       }


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_tl_imagenet100(batch_size, train_set_ratio=1.0, dvs_train_set_ratio=1.0, seed=2000, num_classes=100):
    """
    get the train loader which yield rgb_img and dvs_img with label
    and test loader which yield dvs_img with label of cifar10.
    :return: train_loader, test_loader
    """
    rgb_trans_train = transforms.Compose([
                                          transforms.Resize((48, 48)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                          # transforms.RandomErasing(p=0.1),

                                          ])

    dvs_trans = transforms.Compose([transforms.Resize((48, 48)),
                                    # transforms.RandomCrop(48, padding=4),
                                    # transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                    transforms.ToTensor(),
                                   ])
    dvs_test_trans = transforms.Compose([transforms.Resize((48, 48)),
                                        # transforms.RandomCrop(48, padding=4),
                                        # transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                        transforms.ToTensor(),
                                        ])
    # rgb_trans_train = transforms.Compose([
    #                                       transforms.RandomResizedCrop(176, interpolation=InterpolationMode.BILINEAR),
    #                                       transforms.RandomHorizontalFlip(0.5),
    #                                       autoaugment.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
    #                                       transforms.PILToTensor(),
    #                                       transforms.ConvertImageDtype(torch.float),
    #                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #                                       transforms.RandomErasing(p=0.1),
    #
    #                                       ])
    # mixup_transforms = [MyRandomMixup(num_classes=100, p=1.0, alpha=0.2),
    #                     MyRandomCutmix(num_classes=100, p=1.0, alpha=1.0)]
    # mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
    # collate_fn = lambda batch: mixupcutmix(*default_collate(batch))
    # dvs_trans = transforms.Compose([transforms.Resize((176, 176)),
    #                                 # transforms.RandomCrop(48, padding=4),
    #                                 # transforms.RandomHorizontalFlip(),  # 随机水平翻转
    #                                 transforms.ToTensor(),
    #                                ])
    # dvs_test_trans = transforms.Compose([transforms.Resize((224, 224)),
    #                                     # transforms.RandomCrop(48, padding=4),
    #                                     # transforms.RandomHorizontalFlip(),  # 随机水平翻转
    #                                     transforms.ToTensor(),
    #                                     ])

    tl_train_data = TLImageNet100(DIR['ImageNet100'], DIR['ImageNet100DVS'], train=True,
                                  rgb_train_set_ratio=train_set_ratio, dvs_train_set_ratio=dvs_train_set_ratio, num_classes=num_classes,
                                  transform=rgb_trans_train, dvs_transform=dvs_trans, download=True)
    tl_val_data = TLImageNet100(DIR['ImageNet100'], DIR['ImageNet100DVS'], train=False,
                                dvs_train_set_ratio=train_set_ratio, num_classes=num_classes,
                                dvs_transform=dvs_test_trans, download=True, test_id=0)
    dvs_test_data_list = [TLImageNet100(DIR['ImageNet100'], DIR['ImageNet100DVS'], train=False,
                                        dvs_train_set_ratio=train_set_ratio, num_classes=num_classes,
                                        dvs_transform=dvs_trans, download=True, test_id=i) for i in range(1, 10)]

    # take train set by train_set_ratio
    # if train_set_ratio < 1.0:
    #     n_train = len(tl_train_data)  # 60000
    #     split = int(n_train * train_set_ratio)  # 60000*0.9 = 54000
    #     print(n_train, split)
    #     tl_train_data, _ = my_random_split(tl_train_data, [split, n_train-split], generator=torch.Generator().manual_seed(1000))

    train_dataloader = DataLoaderX(tl_train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True,
                                   pin_memory=True, worker_init_fn=seed_worker)  # collate_fn=collate_fn,
    val_dataloader = DataLoaderX(tl_val_data, batch_size=batch_size*4, shuffle=False, num_workers=8, drop_last=False,
                                 pin_memory=True, worker_init_fn=seed_worker)
    test_dataloader_list = [DataLoaderX(dvs_test_data, batch_size=batch_size*4, shuffle=False, num_workers=8, drop_last=False,
                                        pin_memory=True, worker_init_fn=seed_worker) for dvs_test_data in dvs_test_data_list]

    return train_dataloader, val_dataloader, test_dataloader_list


class TLImageNet100(datasets.VisionDataset):
    def __init__(
            self,
            root: str,
            dvs_root: str,
            train: bool = True,
            rgb_train_set_ratio: float = 1.0,
            dvs_train_set_ratio: float = 1.0,
            num_classes: int = 100,
            transform: Optional[Callable] = None,
            dvs_transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            test_id: int = None,
    ) -> None:

        super(TLImageNet100, self).__init__(root=root, transform=transform,
                                            target_transform=target_transform)

        self.train = train  # training set or test set
        self.rgb_train_set_ratio = rgb_train_set_ratio
        self.dvs_train_set_ratio = dvs_train_set_ratio
        self.dvs_transform = dvs_transform
        self.imgx = transforms.ToPILImage()
        if self.train:
            self.dvs_root = os.path.join(dvs_root, 'train')
        else:
            if test_id == 0:
                self.dvs_root = os.path.join(dvs_root, 'val')
            else:
                self.dvs_root = os.path.join(dvs_root, f'test{test_id}')

        """
        准备RGB数据
        """
        self.class_to_idx = {}
        self.categories = sorted(os.listdir(os.path.join(self.root, "imagenet100")))[: num_classes]
        self.rgb_data = []
        self.targets = []

        for (i, c) in enumerate(self.categories):
            # print(i, c)
            file_list = sorted(os.listdir(os.path.join(self.root, "imagenet100", c)))
            file_range = int(self.rgb_train_set_ratio * len(file_list))
            for file_name in file_list[: file_range]:
                self.rgb_data.append(os.path.join(self.root, "imagenet100", c, file_name))
            self.targets.extend(file_range * [i])
            self.class_to_idx[c] = i
        # print("准备RGB数据", len(self.rgb_data), len(self.targets))

        # 对rgb数据按label排序
        # sort_idx = sorted(range(len(self.targets)), key=lambda k: self.targets[k])
        # self.data = list(np.array(self.data)[sort_idx])
        # self.targets = list(np.array(self.targets)[sort_idx])

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

    def _read_rgb_data(self, category, file_name):
        fimg = open(os.path.join(self.root, "imagenet100", category, file_name), 'rb')
        img = Image.open(fimg)
        if self.transform is not None:
            img = self.transform(img)
        self.rgb_data.append(img)
        fimg.close()

    def _read_dvs_data(self, dvs_class_path, file_name):
        self.dvs_data.append(torch.load(os.path.join(dvs_class_path, file_name)))

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

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            # img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

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
            assert self.class_to_idx[self.dvs_data[dvs_index].split('/')[-2]] == target
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

#
# class MyRandomMixup(jelly_transforms.RandomMixup):
#     """Randomly apply Mixup to the provided batch and targets.
#     The class implements the data augmentations as described in the paper
#     `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.
#
#     Args:
#         num_classes (int): number of classes used for one-hot encoding.
#         p (float): probability of the batch being transformed. Default value is 0.5.
#         alpha (float): hyperparameter of the Beta distribution used for mixup.
#             Default value is 1.0.
#         inplace (bool): boolean to make this transform inplace. Default set to False.
#     """
#
#     def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
#         super(MyRandomMixup, self).__init__(num_classes, p, alpha, inplace)
#
#     def forward(self, batch: Tensor, target: Tensor) -> Tuple[list, Union[Tensor, Any]]:
#         """
#         Args:
#             batch (Tensor): Float tensor of size (B, C, H, W)
#             target (Tensor): Integer tensor of size (B, )
#
#         Returns:
#             Tensor: Randomly transformed batch.
#         """
#         batch, batch_dvs = batch
#         if batch.ndim != 4:
#             raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
#         if target.ndim != 1:
#             raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
#         if not batch.is_floating_point():
#             raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
#         if target.dtype != torch.int64:
#             raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")
#
#         if not self.inplace:
#             batch = batch.clone()
#             target = target.clone()
#
#         if target.ndim == 1:
#             target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)
#
#         if torch.rand(1).item() >= self.p:
#             return batch, target
#
#         # It's faster to roll the batch by one instead of shuffling it to create image pairs
#         batch_rolled = batch.roll(1, 0)
#         target_rolled = target.roll(1, 0)
#
#         # Implemented as on mixup paper, page 3.
#         lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
#         batch_rolled.mul_(1.0 - lambda_param)
#         batch.mul_(lambda_param).add_(batch_rolled)
#
#         target_rolled.mul_(1.0 - lambda_param)
#         target.mul_(lambda_param).add_(target_rolled)
#
#         return [batch, batch_dvs], target
#
#
# class MyRandomCutmix(jelly_transforms.RandomCutmix):
#     """Randomly apply Cutmix to the provided batch and targets.
#     The class implements the data augmentations as described in the paper
#     `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
#     <https://arxiv.org/abs/1905.04899>`_.
#
#     Args:
#         num_classes (int): number of classes used for one-hot encoding.
#         p (float): probability of the batch being transformed. Default value is 0.5.
#         alpha (float): hyperparameter of the Beta distribution used for cutmix.
#             Default value is 1.0.
#         inplace (bool): boolean to make this transform inplace. Default set to False.
#     """
#
#     def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
#         super().__init__(num_classes, p, alpha, inplace)
#
#     def forward(self, batch: Tensor, target: Tensor) -> Tuple[list, Union[Tensor, Any]]:
#         """
#         Args:
#             batch (Tensor): Float tensor of size (B, C, H, W)
#             target (Tensor): Integer tensor of size (B, )
#
#         Returns:
#             Tensor: Randomly transformed batch.
#         """
#         batch, batch_dvs = batch
#         if batch.ndim != 4:
#             raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
#         if target.ndim != 1:
#             raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
#         if not batch.is_floating_point():
#             raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
#         if target.dtype != torch.int64:
#             raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")
#
#         if not self.inplace:
#             batch = batch.clone()
#             target = target.clone()
#
#         if target.ndim == 1:
#             target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)
#
#         if torch.rand(1).item() >= self.p:
#             return batch, target
#
#         # It's faster to roll the batch by one instead of shuffling it to create image pairs
#         batch_rolled = batch.roll(1, 0)
#         target_rolled = target.roll(1, 0)
#
#         # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
#         lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
#         W, H = F.get_image_size(batch)
#
#         r_x = torch.randint(W, (1,))
#         r_y = torch.randint(H, (1,))
#
#         r = 0.5 * math.sqrt(1.0 - lambda_param)
#         r_w_half = int(r * W)
#         r_h_half = int(r * H)
#
#         x1 = int(torch.clamp(r_x - r_w_half, min=0))
#         y1 = int(torch.clamp(r_y - r_h_half, min=0))
#         x2 = int(torch.clamp(r_x + r_w_half, max=W))
#         y2 = int(torch.clamp(r_y + r_h_half, max=H))
#
#         batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
#         lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))
#
#         target_rolled.mul_(1.0 - lambda_param)
#         target.mul_(lambda_param).add_(target_rolled)
#
#         return [batch, batch_dvs], target


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
