import os
import torch
from torch.utils import data
from torchvision import transforms
import cv2
import numpy as np

class MyDatasset(data.Dataset):
    def __init__(self, data_dir, transforms=None, mode='train'):
        self.data_dir = data_dir
        self.mode = mode

        self.imgdir = data_dir + 'images/'
        self.labdir = data_dir + 'labels/'
        imgs = os.listdir(self.imgdir)
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        for img in imgs:
            self.train_data.append(img)
            if img.endswith('.jpeg'):
                filename = img[:-5]
            else:
                filename = img[:-4]
            self.train_label.append(filename + '.txt')
        
    
    def __getitem__(self, index):
        lowerBody = ['lowerBodyTrousers', 'lowerBodyShorts', 'others']
        upperBody = ['upperBodyLongSleeve', 'upperBodyNoSleeve', 'upperBodyShortSleeve', 'others']
        headAccessory = ['accessoryHat', 'others']
        age = ['personalLess15', 'personalLess30', 'personalLess45', 'personalLess60', 'personalLarger60', 'others']
        sex = ['personalMale', 'personalFemale', 'others']
        if self.mode == 'train':
            img = self.train_data[index]
        labelfile = self.train_label[index]
        imgpath = self.imgdir + img
        im = cv2.imread(imgpath)
        labelpath = self.labdir + labelfile
        label = []
        with open(labelpath) as f:
            line = f.readlines()[0].strip().split(' ')
            label.append(lowerBody.index(line[1]))
            label.append(upperBody.index(line[2]))
            label.append(headAccessory.index(line[3]))
            label.append(age.index(line[4]))
            label.append(sex.index(line[5]))
        label = np.array(label, float)
        im = np.array(im, float)
        return im, label

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        else:
            return len(self.test_data)