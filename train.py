# from model import DenseNet121, ResNet101, resnet101_fang
from dataset import MyDatasset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import tensorboardX as tb
from utils import *
from torchvision.models.resnet import *
from networks.myresnet50 import *


def train():
    mydataset =  MyDatasset('./dataset/PETA/')
    dataloader = DataLoader(mydataset, batch_size=2, shuffle=True)
    # print(len(dataloader))
    # net = resnet101_fang(pretrained=False, progress=True)
    net = myresnet50(num_classes=6)
    net = ResNet50()
    net.apply(weights_init_kaiming)
    net = self.load_network(net)
    print(net)
    fang[-1]
    if torch.cuda.is_available():
        net.cuda()
    # init weight
    # net.apply(utils.weights_init_kaiming)
    # optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.0001, momentum = 0.9,
                        weight_decay = 5e-4, nesterov = True,)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 15, gamma = 0.1,)
    # CrossEntropyLoss
    loss_func_CEloss = nn.CrossEntropyLoss()
    # BinCrossEntropyLoss
    # loss_func_BCEloss = nn.BCELoss()
    num_epochs = 50
    all_count = 0
    writer = tb.SummaryWriter()
    for epoch in range(num_epochs):

        print('Epoch {} / {}'.format(epoch+1, num_epochs))
        count_epoch = 0
        # for index, data in enumerate(dataloader):
        for data in dataloader:
            all_count += 1
            im, label = data
            label = label.long()
            if torch.cuda.is_available():
                im = im.cuda()
                label = label.cuda()
            # print(im.size())
            # print(im)
            optimizer.zero_grad()
            # out1, out2, out3, out4, out5 = net(im)
            out = net(im)
            # loss1 = loss_func_CEloss(out1, label[:, 0])
            # loss2 = loss_func_CEloss(out2, label[:, 1])
            # loss3 = loss_func_CEloss(out3, label[:, 2])
            # loss4 = loss_func_CEloss(out4, label[:, 3])
            # loss5 = loss_func_CEloss(out5, label[:, 4])
            # loss = loss1 + loss2 + loss3 + loss4 + loss5
            loss = loss_func_CEloss(out, label[:, 3])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            writer.add_scalar('loss',loss, all_count)
            count_epoch += 1

            if count_epoch % 10 == 0 or (count_epoch+1)==len(dataloader):
                print('{} / {} ------>loss {}'.format(count_epoch, len(dataloader), loss))
                # print('----------->loss1 {}'.format(loss1))
                # print('----------->loss2 {}'.format(loss2))
                # print('----------->loss3 {}'.format(loss3))
                # print('----------->loss4 {}'.format(loss4))
                # print('----------->loss5 {}'.format(loss5))
        if (epoch+1) % 10 == 10:
            utils.save_network(net, epoch+1)
    
    writer.close()


if __name__ == "__main__":
    train()




