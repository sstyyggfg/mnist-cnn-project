import torch
import torch.nn as nn
import torch.optim as optim
import os
from src.model import CNN
from src.dataset import get_train_loader

def train_model(epochs=5, lr=0.001, save_path='checkpoints/mnist_cnn.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader = get_train_loader()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | 损失: {running_loss/len(train_loader):.4f} | 训练准确率: {acc:.2f}%")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至 {save_path}")
    return model
