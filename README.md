# 基于卷积神经网络的手写数字识别

基于 PyTorch 实现 CNN 网络，完成 MNIST 手写数字识别任务。

## 项目结构

mnist-cnn-project/

├── data/               # 数据集存放目录

├── checkpoints/        # 模型权重保存目录

├── src/

│   ├── \*\*init\*\*.py

│   ├── dataset.py      # 数据集加载与预处理

│   ├── model.py        # CNN 模型定义

│   ├── train.py        # 模型训练逻辑

│   └── evaluate.py     # 模型测试评估

├── main.py             # 项目入口

├── requirements.txt    # 依赖清单

├── .gitignore

├── README.md

└── LICENSE

