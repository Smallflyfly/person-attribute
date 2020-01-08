# from model import DenseNet121, ResNet101, resnet101_fang
from dataset import MyDatasset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from utils import *
from torchvision.models.resnet import *
from networks.myresnet50 import *


imgs = os.listdir('./samples/')

net = net.load_network(net)

print(net)


def train(bottleneck, layers):
    mydataset =  MyDatasset('./dataset/PETA/')
    dataloader = DataLoader(mydataset, batch_size=32, shuffle=True)
    # print(len(dataloader))
    # net = resnet101_fang(pretrained=False, progress=True)
    # net = myresnet50(num_classes=6)
    net = ResNet50(block=bottleneck, layers=layers, num_classes=2)
    net.apply(weights_init_kaiming)
    net = net.load_network(net)
    # print(net)
    # fang[-1]
    if torch.cuda.is_available():
        net.cuda()
    # init weight
    # net.apply(utils.weights_init_kaiming)
    # optimizer
    # optimizer = torch.optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9,
    #                     weight_decay = 5e-4, nesterov = True,)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.1,)
    # CrossEntropyLoss
    # loss_func_CEloss = nn.CrossEntropyLoss()
    # loss_func_BCEloss = nn.BCELoss()
    # BinCrossEntropyLoss
    # loss_func_BCEloss = nn.BCELoss()
    # num_epochs = 20
    # all_count = 0
    # writer = tb.SummaryWriter()
    net.train(False)
    for epoch in range(num_epochs):

        print('Epoch {} / {}'.format(epoch+1, num_epochs))
        count_epoch = 0
        
        # for index, data in enumerate(dataloader):"""  """
        for data in dataloader:
            all_count += 1
            im, label = data
            label = label.long()
            if torch.cuda.is_available():
                im = im.cuda()
                label = label.cuda()
            label = label.long()
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
            # print(label[:, 4])
            # print(out.size())
            # print(label[:, 4])
            loss = loss_func_CEloss(out, label[:, 4])
            # fang[-1]
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            writer.add_scalar('loss', loss, all_count)
            count_epoch += 1
        
            # fang[-1]

            if count_epoch % 5 == 0 or (count_epoch+1)==len(dataloader):
                print('{} / {} ------>loss {}'.format(count_epoch, len(dataloader), loss))
                # print('----------->loss1 {}'.format(loss1))
                # print('----------->loss2 {}'.format(loss2))
                # print('----------->loss3 {}'.format(loss3))
                # print('----------->loss4 {}'.format(loss4))
                # print('----------->loss5 {}'.format(loss5))
        
        # adjust lr rate    
        lr_scheduler.step()

        if (epoch+1) % 10 == 0:
            save_network(net, epoch+1)
    
    writer.close()


# if __name__ == "__main__":

#     layers = [3, 4, 6, 3]
#     bottleneck = Bottleneck
#     train(bottleneck, layers) 




