# coding = utf-8

import os
from torch.utils.data import Dataset
from dataloader.dataloader_utils import *
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


# your own data dir
DIR = {'CIFAR10': '/data/zhan/CV_data/cifar10',
       'CIFAR10DVS': '/data/zhan/Event_Camera_Datasets/CIFAR10DVS',
       'CIFAR10DVS_CATCH': '/data/zhan/Event_Camera_Datasets/CIFAR10DVS_dst_cache',
       'SMALL_OFFICE31': '/data/zhan/transfer_data/Small_Office31',
       }



class Dataset(Dataset):
    def __init__(self, dataset_dir, dataset, class_names):
        self._dataset_dir = dataset_dir
        self._dataset = dataset
        self.class_names = class_names
        self._data = list()
        self._classified_data = list()
        self._classified_data_indexes = list()
        self._build_dataset()
        self._transform = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop(227), transforms.ToTensor(),
                                              transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))])

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        file_path, label = self._data[item]
        image = Image.open(file_path).convert("RGB")
        image = self._transform(image)
        return image, label

    def get_item_by_label(self, label):
        index = self._classified_data_indexes[label]
        file_path = self._classified_data[label][index]
        self._classified_data_indexes[label] = (self._classified_data_indexes[label] + 1) % len(self._classified_data[label])
        image = Image.open(file_path).convert("RGB")
        image = self._transform(image)
        return image

    # -------------------------------------------------------------------------------------------------------------------------

    def _build_dataset(self):
        for label, class_name in enumerate(self.class_names):
            file_dir = os.path.join(self._dataset_dir, self._dataset, "images", class_name)
            file_names = os.listdir(file_dir)
            random.shuffle(file_names)
            file_paths = list()

            for file_name in file_names:
                file_path = os.path.join(file_dir, file_name)
                self._data.append((file_path, label))
                file_paths.append(file_path)

            self._classified_data.append(file_paths)
            self._classified_data_indexes.append(0)


def load_dataset(dataset_dir, dataset, batch_size, shuffle=True, drop_last=True):
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop(227), transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, dataset, "images"), transform=transform)
    data_loader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=True)
    return data_loader


def get_small_office31(batch_size, source_name, target_name):
    """
    get the train loader and test loader of small_office31.
    :return: train_loader, test_loader
    """
    class_names = ["back_pack", "bike", "calculator", "desk_chair", "keyboard", "laptop_computer", "mobile_phone",
                   "monitor", "phone", "printer"]
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.RandomCrop(227),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    source_train_dataset = datasets.ImageFolder(root=os.path.join(DIR['SMALL_OFFICE31'], source_name),
                                                transform=transform)
    target_train_dataset = datasets.ImageFolder(root=os.path.join(DIR['SMALL_OFFICE31']+'_train_1_3', target_name),
                                                transform=transform)
    target_test_dataset = datasets.ImageFolder(root=os.path.join(DIR['SMALL_OFFICE31']+'_test_2_3', target_name),
                                               transform=transform)

    source_train_dataloader = DataLoaderX(source_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                          drop_last=True, pin_memory=True)

    target_train_dataloader = DataLoaderX(target_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                          drop_last=True, pin_memory=True)

    target_test_dataloader = DataLoaderX(target_test_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                         drop_last=False, pin_memory=True)

    return source_train_dataloader, target_train_dataloader, target_test_dataloader