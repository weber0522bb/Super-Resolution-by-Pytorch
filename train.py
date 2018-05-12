from __future__ import print_function

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse

from model import ZebraSRNet
from dataset_train import datasetTrain
from dataset_val import datasetVal

#===== Arguments =====

# Training settings
parser = argparse.ArgumentParser(description='NTHU EE - CP HW3 - ZebraSRNet')
parser.add_argument('--patchSize', type=int, default=128, help='HR image cropping (patch) size for training')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--epochSize', type=int, default=250, help='number of batches as one epoch (for validating once)')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--nFeat', type=int, default=16, help='channel number of feature maps')
parser.add_argument('--nResBlock', type=int, default=2, help='number of residual blocks')
parser.add_argument('--nTrain', type=int, default=2, help='number of training images')
parser.add_argument('--nVal', type=int, default=1, help='number of validation images')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use, if Your OS is window, please set to 0')
parser.add_argument('--seed', type=int, default=715, help='random seed to use. Default=715')
parser.add_argument('--printEvery', type=int, default=50, help='number of batches to print average loss ')
args = parser.parse_args()

print(args)

if args.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#===== Datasets =====
print('===> Loading datasets')
train_set = datasetTrain(args)
train_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True)
val_set = datasetVal(args)
val_data_loader = DataLoader(dataset=val_set, num_workers=args.threads, batch_size=1, shuffle=False)

#===== ZebraSRNet model =====
print('===> Building model')
net = ZebraSRNet(nFeat=args.nFeat, nResBlock=args.nResBlock)

if args.cuda:
    net = net.cuda()

#===== Loss function and optimizer =====
criterion = torch.nn.L1Loss()

if args.cuda:
    criterion = criterion.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

#===== Training and validation procedures =====
def train(epoch):
    net.train()
    epoch_loss = 0
    for iteration, batch in enumerate(train_data_loader):
        varIn, varTar = Variable(batch[0]), Variable(batch[1])
        if args.cuda:
            varIn = varIn.cuda()
            varTar = varTar.cuda()

        optimizer.zero_grad()
        loss = criterion(net(varIn), varTar)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        if (iteration+1)%args.printEvery == 0:
            print("===> Epoch[{}]({}/{}): Avg. Loss: {:.4f}".format(epoch, iteration+1, len(train_data_loader), epoch_loss/args.printEvery))
            epoch_loss = 0

from math import log10
def validate():
    net.eval()
    avg_psnr = 0
    mse_criterion = torch.nn.MSELoss()
    for batch in val_data_loader:
        varIn, varTar = Variable(batch[0]), Variable(batch[1])
        if args.cuda:
            varIn = varIn.cuda()
            varTar = varTar.cuda()

        prediction = net(varIn)
        mse = mse_criterion(prediction, varTar)
        psnr = 10 * log10(255*255/mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(val_data_loader)))


def checkpoint(epoch):
    model_out_path = "./model_pretrained/net_F{}B{}_epoch_{}.pth".format(args.nFeat, args.nResBlock, epoch)
    torch.save(net, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

#===== Main procedure =====
for epoch in range(1, args.nEpochs + 1):
    train(epoch)
    validate()
    checkpoint(epoch)
