# -*-coding:utf-8-*-
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import os


def my_loader(path):
    files2 = os.listdir(path)
    if 'I' in files2[0]:
        I = np.loadtxt(path + '/' + files2[0])
        Q = np.loadtxt(path + '/' + files2[1])

    else:
        I = np.loadtxt(path + '/' + files2[1])
        Q = np.loadtxt(path + '/' + files2[0])

    I = I.reshape(8, 550)
    Q = Q.reshape(8, 550)

    data = np.zeros((2, 8, 550))
    for i in range(0, 8):
        data[0][i] = I[i]
        data[1][i] = Q[i]


    data = data.transpose(1, 2, 0)#调整数据维度
    #print(data.shape)
    return data






class MyDataset(Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        files=os.listdir(path)
        datas = []
        for file in files:
            data_path=path+'/'+file
            type=file[0]

            label = ''
            if type == 'A':
                label = int(0)
            if type == 'B':
                label = int(1)  # label
            if type == 'C':
                label = int(2)  # u label

            if type == 'F':
                label = int(3)  # u label
            if type == 'G':
                label = int(4)  # u label
            if type == 'H':
                label = int(5)  # u label
            if type == 'I':
                label = int(6)  # u label
            if type == 'J':
                label = int(7)  # u label
            if type == 'K':
                label = int(8)  # u label
            if type == 'L':
                label = int(9)
            if type == 'M':
                label = int(10)
            if type == 'N':
                label = int(11)
            if type == 'O':
                label = int(12)

            if label == '':
                continue

            datas.append((data_path,int(label)))
        self.imgs = datas
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = my_loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)




