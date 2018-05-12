import random
import torch

def randomCrop(imgIn, imgTar, patchSize, scale=4):
    (ih, iw, c) = imgIn.shape
    (th, tw) = (scale * ih, scale * iw)

    tp = patchSize
    ip = tp // scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    (tx, ty) = (scale * ix, scale * iy)
    imgIn = imgIn[iy:iy + ip, ix:ix + ip, :]
    imgTar = imgTar[ty:ty + tp, tx:tx + tp, :]

    return imgIn, imgTar

def np2PytorchTensor(imgIn, imgTar):
    ts = (2, 0, 1)
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float))
    imgTar = torch.Tensor(imgTar.transpose(ts).astype(float))

    return imgIn, imgTar

def augment(imgIn, imgTar, flip=True, rotation=True):
    if random.random() < 0.5 and flip:
        imgIn = imgIn[:, ::-1, :]
        imgTar = imgTar[:, ::-1, :]

    if rotation:
        if random.random() < 0.5:
            imgIn = imgIn[::-1, :, :]
            imgTar = imgTar[::-1, :, :]
        if random.random() < 0.5:
            imgIn = imgIn.transpose(1, 0, 2)
            imgTar = imgTar.transpose(1, 0, 2)

    return imgIn, imgTar

