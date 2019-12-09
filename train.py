from model import DenseNet121
from dataset import MyDatasset
from torch.utils.data import DataLoader

dataloader = DataLoader(MyDatasset('./dataset/PETA/'), batch_size=1)
for data in dataloader:
    print(data)

net = DenseNet121(num_classes=2)


