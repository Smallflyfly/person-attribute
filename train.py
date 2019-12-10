from model import DenseNet121
from dataset import MyDatasset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import tensorboardX as tb

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def train():
    mydataset =  MyDatasset('./dataset/PETA/')
    dataloader = DataLoader(mydataset, shuffle=True, batch_size=4)
    net = DenseNet121()
    net.cuda()
    # init weight
    net.apply(weights_init_kaiming)
    # optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9,
                        weight_decay = 5e-4, nesterov = True,)
    # CrossEntropyLoss
    loss_func_CEloss = nn.CrossEntropyLoss()
    # BinCrossEntropyLoss
    loss_func_BCEloss = nn.BCELoss()
    num_epochs = 20
    for epoch in range(num_epochs):
        print('Epoch {} / {}'.format(epoch+1, num_epochs))
        for index, data in enumerate(dataloader):
            im, label = data
            im = im.cuda()
            label = label.cuda()

            optimizer.zero_grad()

            outputs = net(im)





if __name__ == "__main__":
    train()




