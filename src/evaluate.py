import torch
from src.dataset import get_test_loader

def evaluate_model(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loader = get_test_loader()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc = 100. * correct / total
    print(f"测试集准确率: {acc:.2f}%")
    return acc
