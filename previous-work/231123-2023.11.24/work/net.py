import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torchsummary import summary
import torch.optim as optim
from torch.optim import lr_scheduler




class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__() #[32, 1, 10000]
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2) # [批量大小, 32, 时间步数]  2 个单位的 padding，确保卷积后的长度不变
        self.pool1 = nn.MaxPool1d(kernel_size=2) # [批量大小, 32, 时间步数 / 2 ]
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2) # [批量大小, 64, 时间步数]
        self.pool2 = nn.MaxPool1d(kernel_size=2) # [批量大小, 64, 时间步数 / 2 ]
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2) #[批量大小, 128, 时间步数]
        # self.pool3 = nn.MaxPool1d(kernel_size=2) # [批量大小, 128, 时间步数 / 2 ]

        num = 10000
        for _ in range(2):  # 三个池化层
            num = (num + 1) // 2  # 池化层缩小时间步数的一半，加1是为了向上取整
            
        self.fc1 = nn.Linear(128 * num, 256)  # 调整全连接层神经元数量 
        self.fc2 = nn.Linear(256, 5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 添加Dropout层

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        # x = self.pool3(nn.functional.relu(self.conv3(x)))  
        x = nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1) #多维张量展平为一维向量，输入全连接层

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 创建模型、优化器和损失函数
model = CNNModel()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001) #SGD不行
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
criterion = nn.CrossEntropyLoss()
summary(model, input_size=(1, 10000))
#input_size 中的 (1, 10000) 表示每个样本只有一个通道（特征），具有10000个时间步。一维卷积模型的预期输入形状。
