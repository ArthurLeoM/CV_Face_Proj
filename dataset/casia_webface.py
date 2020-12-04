#!/usr/bin/env python
# encoding: utf-8


import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import cv2
import os
import torch


def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            return img
    except IOError:
        print('Cannot load image ' + path)


class CASIAWebFace(data.Dataset):
    def __init__(self, root, file_list, transform=None, loader=img_loader):

        self.root = root
        self.transform = transform
        self.loader = loader

        image_list = []
        label_list = []
        with open(file_list) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            #print(info.split('\t'))
            image_path, label_name = tuple(info.split('\t')[:2])
            image_list.append(image_path)
            label_list.append(int(label_name))

        self.image_list = image_list
        #self.label_list = label_list
        #self.class_nums = len(np.unique(self.label_list))
        #print("dataset size: ", len(self.image_list), '/', self.class_nums)
        self.pairs = []
        self.labels = []
        for i in range(len(self.image_list)):
            self.pairs.append((image_list[i], image_list[i]))
            self.labels.append(1)
            tmp_id = i
            while (tmp_id == i):
                tmp_id = int(np.random.randint(0, len(self.image_list)))
            self.pairs.append((image_list[i], image_list[tmp_id]))
            self.labels.append(0)

    def __getitem__(self, index):
        img0_path, img1_path = self.pairs[index]
        label = self.labels[index]

        img0 = self.loader(os.path.join(self.root, img0_path))
        img1 = self.loader(os.path.join(self.root, img1_path))

        # random flip with ratio of 0.5
        # flip = np.random.choice(2) * 2 - 1
        # if flip == 1:
        #     img = cv2.flip(img, 1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        else:
            img0 = torch.from_numpy(img0)
            img1 = torch.from_numpy(img1)

        return img0, img1, label

    def __len__(self):
        return len(self.pairs)


if __name__ == '__main__':
    root = '../CASIA-WebFace'  #'D:/data/webface_align_112'
    file_list = 'names_2000.txt'

    # transform = transforms.Compose([
    #     transforms.Resize((112,112)),
    #     transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]  
    # ])
    augment = transforms.RandomChoice([
        transforms.RandomCrop(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-180, 180))
    ])
    transform_augment = transforms.Compose([
        transforms.ToPILImage(),
        augment,
        transforms.Resize((112,112)),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0] 
    ])
    dataset = CASIAWebFace(root, file_list, transform=transform_augment)
    trainloader = data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=False)
    print(len(dataset))
    for data in trainloader:
        print(data[0].shape)
        break