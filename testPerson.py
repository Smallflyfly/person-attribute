# from model import DenseNet121, ResNet101, resnet101_fang
from dataset import MyDatasset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from utils import *
from torchvision.models.resnet import *
from networks.myresnet50 import *
from PIL import Image
from torchvision import transforms as T

sex = ['personalMale', 'personalFemale']
lowerBody = ['lowerBodyTrousers', 'lowerBodyShorts']
upperBody = ['upperBodyLongSleeve', 'upperBodyNoSleeve', 'upperBodyShortSleeve']
age = ['personalLess15', 'personalLess30', 'personalLess45', 'personalLess60', 'personalLarger60']


transform = T.Compose([
                    T.Resize(size=(288, 144)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

imgs = os.listdir('./samples/')
net = ResNet50(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=2)
net = net.load_network(net)
if torch.cuda.is_available():
    net.cuda()
net.eval()
print(net)
softmaxLayer = nn.Softmax()

for img in imgs:
    print(img)
    im = Image.open('./samples/'+img)
    # im.show()
    print(im.size)
    print(im)
    im = im.resize((144, 288), Image.ANTIALIAS)
    im = transform(im)
    # print(im.size()) # 3 * 288 * 144
    im = im.reshape(1, im.shape[0], im.shape[1], im.shape[2])
    # print(im.size()) # 1 * 3 * 288 * 144
    if torch.cuda.is_available():
        im = im.cuda()
    out = net(im)
    # print(out)
    out1, out2, out3, out4 = out
    out1 = softmaxLayer(out1)
    prob1 = out1.max(1)
    result1 = sex[out1.argmax(1)]

    out2 = softmaxLayer(out2)
    prob2 = out2.max(1)
    result2 = lowerBody[out2.argmax(1)]

    out3 = softmaxLayer(out3)
    prob3 = out3.max(1)
    result3 = upperBody[out3.argmax(1)]

    out4 = softmaxLayer(out4)
    prob4 = out4.max(1)
    result4 = age[out4.argmax(1)]

    print(result1, result2, result3, result4)

    # print(out)
    # fang[-1]




