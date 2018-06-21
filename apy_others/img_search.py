import torch
from torchvision import datasets, models, transforms
import numpy as np
from PIL import Image
import argparse
import os
train_binary = torch.load('/disks/sdb/mingyu_ding/pytorch_deephash/result/train_binary_apy')

parser = argparse.ArgumentParser(description='Image Search')
parser.add_argument('--pretrained', type=str, default=92, metavar='pretrained_model',
                    help='')
parser.add_argument('--querypath', type=str, default='pic/test/20.png', metavar='',
                    help='')
args = parser.parse_args()


'''
trainset = datasets.CIFAR10(root='data', train=True, download=False,
                            transform=None)
for i in range(50000):
    trainset[i][0].save('pic/train/%d.png' % i)
    if i %1000 == 0:
        print i

testset = datasets.CIFAR10(root='data', train=False, download=False,
                           transform=None)
for i in range(10000):
    testset[i][0].save('pic/test/%d.png' % i)
    if i %1000 == 0:
        print i
'''
transform_test = transforms.Compose(
    [transforms.Resize(227),
     transforms.CenterCrop(227),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

query_pic = Image.open(args.querypath)
query_pic = transform_test(query_pic)
print query_pic.size()
from net import AlexNetPlusLatent
net = AlexNetPlusLatent(48)
net.load_state_dict(torch.load('/disks/sdb/mingyu_ding/pytorch_deephash/{}/{}'.format('model', 'apy')))
use_cuda = torch.cuda.is_available()
if use_cuda:
    net.cuda()
    query_pic = query_pic.cuda().unsqueeze(0)
net.eval()
outputs, _ = net(query_pic)
query_binary = (outputs[0] > 0.5).cpu().numpy()

#print query_binary

trn_binary = train_binary.cpu().numpy()
query_result = np.count_nonzero(query_binary != trn_binary, axis=1)    #don't need to divide binary length
sort_indices = np.argsort(query_result)
f = open('/disks/sdb/images.txt').readlines()


os.system('cp ' + args.querypath + ' /var/www/html/img/0.png')
for i in range(30):
    os.system('cp /disks/sdb/' + f[sort_indices[i]].strip().split()[0] + ' /var/www/html/img/%s.png' % str(i+1))


os.system('touch ok')
print 'ok'


'''
import matplotlib.pyplot as plt
%matplotlib inline
for i in range(9):
    img = np.array(Image.open('pic/train/' + str(sort_indices[i]) + '.png'))
    plt.subplot(33*10 + i)
    plt.imshow(img)
'''
