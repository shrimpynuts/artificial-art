import re
import numpy as np
import pickle
from scipy import misc
import argparse


parser = argparse.ArgumentParser(description="parser")
parser.add_argument('--image-genre', action='store', type=str, metavar='N',
                    help='The folder of all images')
args = parser.parse_args()


ntrain = 1000
ntest = 100
# nclass = 10
imsize = 28
nchannels = 1   # grayscale only one channel

# Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
# Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
# LTrain = np.zeros((ntrain*nclass,nclass))
# LTest = np.zeros((ntest*nclass,nclass))

itrain = -1
itest = -1
# for iclass in range(0, nclass):

for isample in range(0, ntrain):
    # path = '~/CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
    # path = '/Users/yiyangxu/Work/CIFAR10/Train/%d/Image%05d.png' % (iclass, isample)
    path = '/User/%d/Image%05d.png' % (iclass, isample)
    im = misc.imread(path); # 28 by 28
    im = im.astype(float)/255
    itrain += 1
    Train[itrain,:,:,0] = im
    LTrain[itrain,iclass] = 1 # 1-hot label
for isample in range(0, ntest):
    # path = '~/CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
    # path = '/Users/yiyangxu/Work/CIFAR10/Test/%d/Image%05d.png' % (iclass, isample)
    path = '/home/yx48/Workspace/Class/CIFAR10/Test/%d/Image%05d.png' % (iclass, isample)
    im = misc.imread(path); # 28 by 28
    im = im.astype(float)/255
    itest += 1
    Test[itest,:,:,0] = im
    LTest[itest,iclass] = 1 # 1-hot label


