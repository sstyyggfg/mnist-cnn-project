# 基于卷积神经网络的手写数字识别

基于 PyTorch 实现 CNN 网络，完成 MNIST 手写数字识别任务，测试集准确率可达 98.93%。



## 项目结构

mnist-cnn-project/

├── data/               # 数据集存放目录（自动生成）

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



## 数据集说明

- **数据集**：MNIST 手写数字数据集

- **来源**：由 Yann LeCun 提供，通过 torchvision 内置接口自动下载

- **数据规模**：60000 张训练集，10000 张测试集，均为 28×28 灰度图，共 0-9 十个分类

- **预处理流程**：

&#x20; 1.  转为 Tensor 张量，像素值归一化至 \[0,1]

&#x20; 2.  使用均值 0.1307、标准差 0.3081 做标准化

- **划分方式**：使用官方预设的训练集 / 测试集划分，无需手动切分



## 环境要求

- Python 3.8+

- PyTorch 2.0+

- torchvision 0.15+



## 运行方法

1.  克隆仓库

```bash

git clone https://github.com/sstyyggfg/mnist-cnn-project.git

cd mnist-cnn-project

2.安装依赖

pip install -r requirements.txt

3.启动训练与评估

python main.py

实验结果

训练轮数：5 epoch

优化器：Adam，学习率 0.001

损失函数：交叉熵损失

训练集最终准确率：99.45%

测试集准确率：98.93%



开源协议

本项目基于 MIT License 开源。

