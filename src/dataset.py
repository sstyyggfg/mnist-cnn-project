import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 数据预处理流水线
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_train_loader(batch_size=64):
    """获取训练集数据加载器"""
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def get_test_loader(batch_size=1000):
    """获取测试集数据加载器"""
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
