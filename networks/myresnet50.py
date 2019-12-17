import torch
import torch.nn as nn
from torch.nn import init
from collections import OrderedDict
from torchvision.models.densenet import *
from torchvision.models.resnet import *

model = resnet50(num_classes=6)
print(model)
