import torch
import torch.utils.data as data
import os
import skimage
import skimage.io as sio

import data_augment

class datasetTrain(data.Dataset):
    def __init__(self, args):
        self.patchSize = args.patchSize
        self.epochSize = args.epochSize
        self.batchSize = args.batchSize
        self.nTrain = args.nTrain

        self.trainDir = 'image_train'
        self.imgInPrefix = 'LR_zebra_train'
        self.imgTarPrefix = 'HR_zebra_train'

    def __getitem__(self, idx):
        idx = (idx % self.nTrain) + 1

        nameIn, nameTar = self.getFileName(idx)
        imgIn = sio.imread(nameIn)
        imgTar = sio.imread(nameTar)
 
        imgIn, imgTar = data_augment.randomCrop(imgIn, imgTar, self.patchSize)
        imgIn, imgTar = data_augment.augment(imgIn, imgTar)

        return data_augment.np2PytorchTensor(imgIn, imgTar)


    def __len__(self):
        return self.epochSize*self.batchSize

    def getFileName(self, idx):
        fileName = '{:0>4}'.format(idx)
        nameIn = '{}_{}.png'.format(self.imgInPrefix, fileName)
        nameIn = os.path.join(self.trainDir, nameIn)
        nameTar = '{}_{}.png'.format(self.imgTarPrefix, fileName)
        nameTar = os.path.join(self.trainDir, nameTar)

        return nameIn, nameTar
