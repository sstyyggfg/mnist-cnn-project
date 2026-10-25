from src.train import train_model
from src.evaluate import evaluate_model

if __name__ == '__main__':
    print("开始训练模型...")
    model = train_model(epochs=5)
    print("\n开始测试评估...")
    evaluate_model(model)
