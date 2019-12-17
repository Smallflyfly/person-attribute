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

def load_state_dict(state_dict):
    """
    Because we remove the definition of fc layer in resnet now, it will fail when loading 
    the model trained before.
    To provide back compatibility, we overwrite the load_state_dict
    """
    # print(type(state_dict))
    # print(state_dict['fc.bias'])
    state_dict.pop('fc.bias')
    state_dict.pop('fc.weight')
    # print(state_dict)
    # fang[-1]

    nn.Module.load_state_dict(self, state_dict=state_dict)

def load_network(network):
    # save_path = os.path.join(model_dir,'net_%s.pth'%args.which_epoch)
    save_path = './resnet50.pth'
    load_state_dict(torch.load(save_path))
    return network