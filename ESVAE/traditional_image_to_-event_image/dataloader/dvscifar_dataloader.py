import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import os
from scipy.io import loadmat


USERNAME = 'zhan'


class DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '{}.pt'.format(index))

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target.long()

    def __len__(self):
        return len(os.listdir(self.root))


def gather_addr(directory, train_test_portion, event_file_type='mat', get_front=True):
    fns = []
    num = 0
    filename_list = os.listdir(directory)
    filename_list.sort()
    for filename in filename_list:
        # filename获取类别名称
        class_path = os.path.join(directory, filename)
        if event_file_type == 'mat':
            class_data = glob.glob(os.path.join(class_path, '*.mat'))
        elif event_file_type == 'np':
            # 获取类别下的所有文件
            class_data = glob.glob(os.path.join(class_path, '*.npz'))
        class_data.sort()

        num = num + len(class_data)
        divid = int(len(class_data)*train_test_portion)

        # 获取下标范围
        if get_front:
            start_id = 0
            end_id = divid
        else:
            start_id = divid
            end_id = len(class_data)

        for i in range(start_id, end_id):
            if event_file_type == 'mat':
                file_name = class_path + '/' + f"{i}.mat"
            elif event_file_type == 'np':
                file_name = class_data[i]
                # file_name = class_path + '/' + f"cifar10_{filename}_{i}.npz"
            fns.append(file_name)

    print("num:",num)
    print("fns:",len(fns))
    return fns


def events_to_frames(filename, mapping, shape, dt, event_file_type='mat'):
    label_filename = filename[:].split('/')[-2]
    label = int(list(mapping.values()).index(label_filename))

    frames = np.zeros(shape)  # Caltech101: (240, 180), Cifar10: (128, 128), MNIST: (34, 34)

    if event_file_type == 'mat':
        events = loadmat(filename)['out1']
        # --- normal event concatanation
        # for i in range(shape[0]):
        #     frames[i, events[i * dt: (i+1) * dt, 3], events[i * dt: (i+1) * dt, 1], events[i * dt: (i+1) * dt, 2]] += 1
        # --- building time surfaces
        # print(events.shape, np_events['t'].shape, np_events['x'].shape, np_events['y'].shape, np_events['p'].shape) 
        # print(events[:5])
        # print([(np_events['t'][i], np_events['x'][i], np_events['y'][i], np_events['p'][i]) for i in range(5)])
        events = np.delete(events, [1,2], axis=1)
        events[:, 3] = events[:, 3] == 1
    elif event_file_type == 'np':
        np_events = np.load(filename)
        events = np.zeros((len(np_events['t']), 4), dtype=np.int32)

        events[:, 0] = np_events['t'].astype(np.int32)
        events[:, 1] = np_events['x'].astype(np.int32)
        events[:, 2] = np_events['y'].astype(np.int32)
        events[:, 3] = np_events['p'].astype(np.int32)
        # print(np.max(events[:, 1]), np.max(events[:, 2]))
    elif event_file_type == 'imagenet':
        np_events = np.load(filename)['event_data']
        events = np.zeros((len(np_events['t']), 4), dtype=np.int32)

        events[:, 0] = np_events['t'].astype(np.int32)
        events[:, 1] = np_events['x'].astype(np.int32)
        events[:, 2] = np_events['y'].astype(np.int32)
        events[:, 3] = np_events['p'].astype(np.int32)

        events[:, 1] = (events[:, 1] * (224 / 640)).astype(np.int32)
        events[:, 2] = (events[:, 2] * (224 / 480)).astype(np.int32)

    for i in range(shape[0]):
        # print("--------------------",i)
        # print(events.shape)
        r1 = i * (events.shape[0] // shape[0])
        r2 = (i + 1) * (events.shape[0] // shape[0])
        # print(i,events[r1:r2, 3].shape, events[r1:r2, 1].shape, events[r1:r2, 2].shape)
        frames[i, events[r1:r2, 3], events[r1:r2, 1], events[r1:r2, 2]] += 1  # events[r1:r2, 0]
    
    frame_result = []
    for i in range(shape[0]):
        tmp = frames[i, :, :, :] / np.max(frames[i, :, :, :])  # 每帧图像除以最大值归一化
        if event_file_type == 'np':   # np格式输入的数据，需要旋转帧图像
            tmp = tmp.transpose(1, 2, 0)
            tmp = np.transpose(tmp[::-1, ...][:, ::-1], axes=(1, 0, 2))[::-1, ...]  # 顺时针旋转90度
            tmp = tmp[:, ::-1]  # 镜面翻转
            tmp = tmp.transpose(2, 0, 1)
        frame_result.append(tmp)
    
    frame_result = np.stack(frame_result, axis=0)  # 将所有帧图像压缩合并为一个array

    return frame_result, label, label_filename


def create_npy(mapping, dataname, event_data_name, shape, train_test_portion=0.9, event_file_type='np'):
    root_path = f'/data/{USERNAME}/Event_Camera_Datasets/{dataname}/{event_data_name}/temporal_effecient_training_{train_test_portion}_{event_file_type}_{shape[0]}_timestep'

    if train_test_portion < 1.0:
        event_path = f'/data/{USERNAME}/Event_Camera_Datasets/{dataname}/{event_data_name}/events_{event_file_type}'
        fns_train = gather_addr(event_path, train_test_portion, event_file_type, get_front=True)
        fns_test = gather_addr(event_path, train_test_portion, event_file_type, get_front=False)
    else:
        event_path_train = f'/data/{USERNAME}/Event_Camera_Datasets/{dataname}/{event_data_name}/events_{event_file_type}/train'
        event_path_test = f'/data/{USERNAME}/Event_Camera_Datasets/{dataname}/{event_data_name}/events_{event_file_type}/test'
        fns_train = gather_addr(event_path_train, train_test_portion, event_file_type, get_front=True)
        fns_test = gather_addr(event_path_test, train_test_portion, event_file_type, get_front=True)

    train_filename = root_path + '/train/{}/{}_{}.pt'
    test_filename = root_path + '/test/{}/{}_{}.pt'

    if not os.path.exists(root_path):
        os.mkdir(root_path)
        os.mkdir(f'{root_path}/train')
        os.mkdir(f'{root_path}/test')
        for value in mapping.values():
            os.mkdir(f'{root_path}/train/{value}')
            os.mkdir(f'{root_path}/test/{value}')

    print("processing training data...")
    if dataname == 'ImageNet100':
        event_to_frame_file_type = 'imagenet'
    else:
        event_to_frame_file_type = event_file_type
    key = -1
    for file_d in fns_train:
        if key % 100 == 0:
            print("\r\tTrain data {:.2f}% complete\t\t".format(key*100 / len(fns_train)), end='')
        frames, label, label_filename = events_to_frames(file_d, mapping, shape, dt=5000, event_file_type=event_to_frame_file_type)
        key += 1
        torch.save(torch.Tensor(frames),
                   train_filename.format(label_filename, key, event_file_type))
        # torch.save([torch.Tensor(frames), torch.Tensor([label,])],
        #            train_filename.format(label_filename, key, event_file_type))
    print("training data end\n")

    # TODO:待注释
    print("\nprocessing testing data...")
    key = -1
    for file_d in fns_test:
        if key % 100 == 0:
            print("\r\tTest data {:.2f}% complete\t\t".format(key*100 / len(fns_test)), end='')
        frames, label, label_filename = events_to_frames(file_d, mapping, shape, dt=5000, event_file_type=event_to_frame_file_type)
        key += 1
        torch.save(torch.Tensor(frames),
                   test_filename.format(label_filename, key, event_file_type))
        # torch.save([torch.Tensor(frames), torch.Tensor([label,])],
        #            test_filename.format(key))
    print('testing data end\n')


def create_nimage_npy(mapping, dataname, event_data_name, shape, train_test_portion=0.9, event_file_type='np'):
    root_path = f'/data/{USERNAME}/Event_Camera_Datasets/{dataname}/{event_data_name}/temporal_effecient_training_{train_test_portion}_{event_file_type}_{shape[0]}_timestep'

    event_path_train = f'/data/{USERNAME}/Event_Camera_Datasets/{dataname}/{event_data_name}/events_{event_file_type}/train'
    event_path_val = f'/data/{USERNAME}/Event_Camera_Datasets/{dataname}/{event_data_name}/events_{event_file_type}/val'
    event_path_test_list = [f'/data/{USERNAME}/Event_Camera_Datasets/{dataname}/{event_data_name}/events_{event_file_type}/test{i}' for i in range(1, 10)]
    fns_train = gather_addr(event_path_train, train_test_portion, event_file_type, get_front=True)
    fns_val = gather_addr(event_path_val, train_test_portion, event_file_type, get_front=True)
    fns_test_list = [gather_addr(x, train_test_portion, event_file_type, get_front=True) for x in event_path_test_list]

    train_filename = root_path + '/train/{}/{}_{}.pt'
    val_filename = root_path + '/val/{}/{}_{}.pt'
    test_filename_list = [root_path + f'/test{i}' + '/{}/{}_{}.pt' for i in range(1, 10)]

    if not os.path.exists(root_path):
        os.mkdir(root_path)
        os.mkdir(f'{root_path}/train')
        os.mkdir(f'{root_path}/val')
        for value in mapping.values():
            os.mkdir(f'{root_path}/train/{value}')
            os.mkdir(f'{root_path}/val/{value}')

        for i in range(1, 10):
            os.mkdir(f'{root_path}/test{i}')
            for value in mapping.values():
                os.mkdir(f'{root_path}/test{i}/{value}')

    print("processing training data...")
    if dataname == 'ImageNet100':
        event_to_frame_file_type = 'imagenet'
    else:
        event_to_frame_file_type = event_file_type
    key = -1
    for file_d in fns_train:
        if key % 100 == 0:
            print("\r\tTrain data {:.2f}% complete\t\t".format(key*100 / len(fns_train)), end='')
        frames, label, label_filename = events_to_frames(file_d, mapping, shape, dt=5000, event_file_type=event_to_frame_file_type)
        key += 1
        torch.save(torch.Tensor(frames),
                   train_filename.format(label_filename, key, event_file_type))
        # torch.save([torch.Tensor(frames), torch.Tensor([label,])],
        #            train_filename.format(label_filename, key, event_file_type))
    print("training data end\n")

    print("\nprocessing val data...")
    key = -1
    for file_d in fns_val:
        if key % 100 == 0:
            print("\r\tVal data {:.2f}% complete\t\t".format(key*100 / len(fns_val)), end='')
        frames, label, label_filename = events_to_frames(file_d, mapping, shape, dt=5000, event_file_type=event_to_frame_file_type)
        key += 1
        torch.save(torch.Tensor(frames),
                   val_filename.format(label_filename, key, event_file_type))
        # torch.save([torch.Tensor(frames), torch.Tensor([label,])],
        #            test_filename.format(key))
    print('val data end\n')

    for i in range(1, 10):
        print(f"\nprocessing test{i} data...")
        key = -1
        for file_d in fns_test_list[i-1]:
            if key % 100 == 0:
                print("\r\tTest{} data {:.2f}% complete\t\t".format(i, key * 100 / len(fns_val)), end='')
            frames, label, label_filename = events_to_frames(file_d, mapping, shape, dt=5000,
                                                             event_file_type=event_to_frame_file_type)
            key += 1
            torch.save(torch.Tensor(frames),
                       test_filename_list[i-1].format(label_filename, key, event_file_type))
            # torch.save([torch.Tensor(frames), torch.Tensor([label,])],
            #            test_filename.format(key))
        print(f'test{i} data end\n')


def get_mapping(data_type):
    if data_type == 'CIFAR10':
        mapping = { 0 :'airplane',
                    1 :'automobile',
                    2 :'bird',
                    3 :'cat',
                    4 :'deer',
                    5 :'dog',
                    6 :'frog',
                    7 :'horse',
                    8 :'ship',
                    9 :'truck'}
        T = 10
        dataname = 'CIFAR10'
        event_data_name = 'CIFAR10DVS'
        shape = (T, 2, 128, 128)
        train_test_portion = 0.9
        event_file_type = 'mat'
    elif data_type == 'MNIST':
        mapping = { 0 :'0',
                    1 :'1',
                    2 :'2',
                    3 :'3',
                    4 :'4',
                    5 :'5',
                    6 :'6',
                    7 :'7',
                    8 :'8',
                    9 :'9', }
        T = 12
        dataname = 'MNIST'
        event_data_name = 'NMNIST'
        shape = (T, 2, 34, 34)
        train_test_portion = 1.0
        event_file_type = 'np'
    elif data_type == 'Caltech101':
        # 101个类别通过遍历类别文件名称获取
        mapping = {}
        category_path = f'/data/{USERNAME}/Event_Camera_Datasets/Caltech101/NCaltech101/extract/Caltech101' #101
        i = 0 
        for filename in os.listdir(category_path):
            mapping[i] = filename
            i = i + 1
        T = 10
        dataname = 'Caltech101'
        event_data_name = 'NCaltech101'
        shape = (T, 2, 240, 180)
        train_test_portion = 0.9
        event_file_type = 'np'
    elif data_type == 'ImageNet100':
        # 100个类别通过遍历类别文件名称获取
        mapping = {}
        category_path = f'/data/{USERNAME}/Event_Camera_Datasets/ImageNet100/NImageNet100/events_np/train'
        i = 0
        for filename in os.listdir(category_path):
            mapping[i] = filename
            i = i + 1
        T = 4
        dataname = 'ImageNet100'
        event_data_name = 'NImageNet100'
        shape = (T, 2, 224, 224)  # (T, 2, 640, 480)
        train_test_portion = 1.0
        event_file_type = 'np'
    return mapping, dataname, event_data_name, shape, train_test_portion, event_file_type


if __name__ == "__main__":

    # TODO:参数修改 'CIFAR10' 'Caltech101' 'MNIST'.获取文件路径和标签映射
    mapping, dataname, event_data_name, shape, train_test_portion, event_file_type = get_mapping(data_type='ImageNet100')
    print(dataname)
    # 末尾指定划分训练和测试集的比例，以及读取文件类型
    if dataname == 'ImageNet100':
        create_nimage_npy(mapping, dataname, event_data_name, shape, train_test_portion, event_file_type)
    else:
        create_npy(mapping, dataname, event_data_name, shape, train_test_portion, event_file_type)
