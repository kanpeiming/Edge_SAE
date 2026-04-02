import os
import scipy.io as scio
from PIL import Image
import numpy as np
import re

class event_load():

    def __init__(self, RGB_img_folder, event_folder):

        self.RGB_img_folder = RGB_img_folder
        self.event_folder = event_folder

    def get_data(self, RGB_img_file_name):

        RGB_img_path = os.path.join(self.RGB_img_folder, RGB_img_file_name)
        event_path = os.path.join(self.event_folder, RGB_img_file_name.replace('jpg','mat'))

        img = Image.open(RGB_img_path).convert('RGB')
        event_ary = self.loadMat(event_path) # keys : {'x', 'y', 'p', 'ts}

        # img.show()
        # print(event_ary)

        return img, event_ary

    def loadMat(self, filePath):
        data = scio.loadmat(filePath, verify_compressed_data_integrity=False)
        for key in data.keys():
            if isinstance(data[key], np.ndarray):
                data[key] = np.squeeze(data[key].astype(np.int64))
                if key == "p":
                    data[key] = np.where(data[key] == -1, 0, data[key])
        return data

if __name__ == '__main__':
    event_loader = event_load(RGB_img_folder="data/img",
                              event_folder="data/MAT/img")
    img, event_ary = event_loader.get_data("00000.jpg")
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    z = np.fromiter(zip(event_ary['x'], event_ary['y'], event_ary['ts'], event_ary['p']), dtype=dtype)

    data = np.genfromtxt(open('pathFile.csv', "rb"), delimiter=",", skip_header=1, dtype='U')
    print(data)