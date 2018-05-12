from __future__ import print_function

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import skimage
import skimage.io as sio
import scipy.misc
import argparse

from calc_psnr import calc_psnr

#===== Arguments =====

# Testing settings
parser = argparse.ArgumentParser(description='NTHU EE - CP HW3 - ZebraSRNet')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str, required=True, help='where to save the output image')
parser.add_argument('--compare_image', type=str, help='ground-truth image to compare with the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
args = parser.parse_args()

print(args)

if args.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

#===== load ZebraSRNet model =====
print('===> Loading model')

net = torch.load(args.model)

if args.cuda:
    net = net.cuda()

#===== Load input image =====
imgIn = sio.imread(args.input_image)
imgIn = imgIn.transpose((2,0,1)).astype(float)
imgIn = imgIn.reshape(1,imgIn.shape[0],imgIn.shape[1],imgIn.shape[2])
imgIn = torch.Tensor(imgIn)

#===== Test procedures =====
varIn = Variable(imgIn)
if args.cuda:
    varIn = varIn.cuda()

prediction = net(varIn)
prediction = prediction.data.cpu().numpy().squeeze().transpose((1,2,0))

scipy.misc.toimage(prediction, cmin=0.0, cmax=255.0).save(args.output_filename)

#===== Ground-truth comparison =====
if args.compare_image is not None:
    imgTar = sio.imread(args.compare_image)
    psnr = calc_psnr(prediction, imgTar, max_val=255.0)
    print('===> PSNR: %.3f dB'%(psnr))

#for iteration, (tenIn, tenTar) in enumerate(train_data_loader, 1):
#    print(iteration)

#import numpy
#npIn = tenIn.numpy()
#npIn = npIn[0,:,:,:].squeeze()
#npIn = numpy.transpose(npIn,(1,2,0))

#import scipy.misc
#scipy.misc.toimage(npIn, cmin=0.0, cmax=255.0).save('./test.png')

#npTar = tenTar.numpy()
#npTar = npTar[0,:,:,:].squeeze()
#npTar = numpy.transpose(npTar,(1,2,0))

#scipy.misc.toimage(npTar, cmin=0.0, cmax=255.0).save('./test_target.png')
