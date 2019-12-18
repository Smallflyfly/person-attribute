import torch
import torch.nn as nn
import os

def save_network(network, epoch_label):
                save_filename = 'net_%s.pth'% epoch_label
                save_path = os.path.join('./weights/', save_filename)
                torch.save(network.cuda().state_dict(), save_path)
                network.cuda()

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0.01, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0.01, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.002)
        nn.init.constant_(m.bias.data, 0.0)

