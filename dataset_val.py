import torch
import torch.utils.data as data
import os
import skimage
import skimage.io as sio

import data_augment

class datasetVal(data.Dataset):
    def __init__(self, args):
        self.nVal = args.nVal

        self.valDir = 'image_val'
        self.imgInPrefix = 'LR_zebra_val'
        self.imgTarPrefix = 'HR_zebra_val'

    def __getitem__(self, idx):
        idx = (idx % self.nVal) + 1

        nameIn, nameTar = self.getFileName(idx)
        imgIn = sio.imread(nameIn)
        imgTar = sio.imread(nameTar)

        return data_augment.np2PytorchTensor(imgIn, imgTar)


    def __len__(self):
        return self.nVal

    def getFileName(self, idx):
        fileName = '{:0>4}'.format(idx)
        nameIn = '{}_{}.png'.format(self.imgInPrefix, fileName)
        nameIn = os.path.join(self.valDir, nameIn)
        nameTar = '{}_{}.png'.format(self.imgTarPrefix, fileName)
        nameTar = os.path.join(self.valDir, nameTar)

        return nameIn, nameTar
