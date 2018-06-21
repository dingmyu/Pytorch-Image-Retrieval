import torch
from torchvision import datasets, models, transforms
import numpy as np
from PIL import Image

transform_test = transforms.Compose(
    [transforms.Resize(227),
     transforms.CenterCrop(227),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

import torch.utils.data as data
import os
from net import AlexNetPlusLatent

class MyDataset(data.Dataset):
    def __init__(self, file, dir_path, transform=None):
        imgs = []
        fw = open(file, 'r')
        lines = fw.readlines()
        for line in lines:
            words = line.strip().split()
            imgs.append((words[0], words[1]))
        self.imgs = imgs
        self.dir_path = dir_path
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.imgs[index]
        path = os.path.join(self.dir_path, path)
        img = Image.open(path).convert('RGB')
        label = int(label) - 1
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
    

train_data = MyDataset('/disks/sdb/images.txt', '/disks/sdb/',transform_test)

from torch.autograd import Variable
def binary_output(dataloader):
    net = AlexNetPlusLatent(48)
    net.load_state_dict(torch.load('./{}/{}'.format('model', 92)))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
    full_batch_output = torch.cuda.FloatTensor()
    full_batch_label = torch.cuda.LongTensor()
    net.eval()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print batch_idx
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs, _ = net(inputs)
        full_batch_output = torch.cat((full_batch_output, outputs.data), 0)
        full_batch_label = torch.cat((full_batch_label, targets.data), 0)
    return torch.round(full_batch_output), full_batch_label

trainloader = torch.utils.data.DataLoader(train_data, batch_size=100,
                                          shuffle=False, num_workers=2)

train_binary, train_label = binary_output(trainloader)

query_pic = train_data[2][0]
net = AlexNetPlusLatent(48)
net.load_state_dict(torch.load('./{}/{}'.format('model', 92)))
use_cuda = torch.cuda.is_available()
if use_cuda:
    net.cuda()
    query_pic = query_pic.cuda().unsqueeze(0)
net.eval()
outputs, _ = net(query_pic)
query_binary = (outputs[0] > 0.5).cpu().numpy()
trn_binary = train_binary.cpu().numpy()
query_result = np.count_nonzero(query_binary != trn_binary, axis=1)    #don't need to divide binary length
sort_indices = np.argsort(query_result)

print sort_indices
