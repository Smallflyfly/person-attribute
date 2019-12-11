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
    count_epoch = 0
    writer = tb.SummaryWriter()
    for epoch in range(num_epochs):
        print('Epoch {} / {}'.format(epoch+1, num_epochs))
        for index, data in enumerate(dataloader):
            print(index, data)
            fang[-1]
            im, label = data
            im = im.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            out1, out2, out3, out4, out5 = net(im)
            loss1 = loss_func_CEloss(out1, label[0])
            loss2 = loss_func_CEloss(out2, label[1])
            loss3 = loss_func_BCEloss(out3, label[2])
            loss4 = loss_func_CEloss(out4, label[3])
            loss5 = loss_func_CEloss(out5, label[4])
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss',loss,count_epoch)
            count_epoch += 1

            if index%10 == 0:
                print('------>loss {}'.format(loss))
    
    writer.close()


if __name__ == "__main__":
    train()




