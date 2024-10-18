# 三层卷积两层池化 padding=2
# 输出数量混淆矩阵; 百分比混淆矩阵; 损失、准确率、学习率曲线; print 网络参数

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim import lr_scheduler
from torchsummary import summary
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



# 数据加载和预处理
data = []
labels = []

excel_path = ['E:\北化\excel数据\拇指\\0-1-110.xlsx', 'E:\北化\excel数据\拇指\\0-111-220.xlsx', 'E:\北化\excel数据\食指\\1-1-110.xlsx', 'E:\北化\excel数据\食指\\1-111-220.xlsx',
              'E:\北化\excel数据\中指\\2-1-100.xlsx', 'E:\北化\excel数据\中指\\2-101-220.xlsx', 'E:\北化\excel数据\无名指\\3-1-110.xlsx', 'E:\北化\excel数据\无名指\\3-111-220.xlsx', 
              'E:\北化\excel数据\小指\\4-1-110.xlsx', 'E:\北化\excel数据\小指\\4-111-220.xlsx']

startTime = [32000, 38000, 32000, 34000, 57000, 84000, 38000, 34000, 37000, 34000]#31000
thresholds = [0.05, 0.05, 0.08, 0.08, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1] #阈值

for i in range(0, 10):
    df = pd.read_excel(excel_path[i], header=None)
    # 获取信号数据
    signal = df.iloc[:, 1].values #提取b列信号
    n = 0 #该excel中提取到的样本数

    # 扫描信号
    for j in range(len(signal)):
        if j<startTime[i]:
            continue
        
        if signal[j] > thresholds[i]:
            # 找到波峰的起始点
            start_index = j
            while j < len(signal) and signal[j] > thresholds[i]:
                j += 1
            # 找到波峰的结束点
            end_index = j

            # 取波峰中心点，左右各取相同长度
            center_index = (start_index + end_index) // 2
            half_length = 5000  # 左右各取5000个点
            if center_index - half_length < 0 or center_index + half_length > len(signal): 
                break 
            start_index =  center_index - half_length
            end_index =  center_index + half_length
            
            j = end_index

            # 提取样本
            sample = signal[start_index:end_index]

            # 将样本添加到列表中
            data.append(sample)
            labels.append(i // 2)  # i // 2 - 1
            n +=1
            
            # #可视化每个样本
            # if n%10==0 :
            #     plt.figure(figsize=(10, 6))
            #     plt.plot(sample)
            #     plt.show()
            #     plt.close()

        
        if i==4:
            if n>90:
                break
        else:
            if n>100:
                break
        
        

# 数据格式化
# data = [torch.tensor(arr, dtype=torch.float32) for arr in data]
data = torch.tensor(data, dtype=torch.float32)
data = data.unsqueeze(1)  # 添加一个通道维度
# data = data.transpose(0,2,1)
print(data.size())
labels = torch.tensor(labels, dtype=torch.long)


# 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(len(X_train), len(y_train)) 
# 创建训练数据集和测试数据集的实例
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
# 创建数据加载器
batch_size = 32  # 指定每个批次的大小
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #shuffle=True 表示打乱
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 构建卷积神经网络模型
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
        x = nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1) #多维张量展平为一维向量，输入全连接层

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# 创建模型、优化器和损失函数
model = CNNModel()
optimizer = optim.Adam(model.parameters(), lr=0.001) #SGD不行
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
criterion = nn.CrossEntropyLoss()

summary(model, input_size=(1, 10000))


# 训练模型
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 在每个 epoch 结束后更新学习率
        # scheduler.step()
        running_loss += loss.item()
    
    # 返回平均训练损失
    return running_loss / len(train_loader)

#测试模型
def test_model(model, test_loader):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    all_predicted = []  # 用于存储所有预测结果
    # predicted_last = torch.Tensor()

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # 将当前批次的预测结果添加到列表中
            all_predicted.extend(predicted.tolist())
            # predicted_last = predicted

    # 返回测试准确率
    return correct_predictions / total_samples, all_predicted 

# 函数用于训练和评估模型，并绘制结果
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    train_losses = []
    test_accuracies = []
    learning_rates = []
    predicted_list = []

    for epoch in range(num_epochs):
        # 在每个 epoch 开始时调用学习率调度器
        if epoch>0:
            scheduler.step()
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 计算每个 epoch 的平均训练损失
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # 在测试集上评估模型
        test_accuracy, predicted = test_model(model, test_loader)
        test_accuracies.append(test_accuracy)
        predicted_list.append(predicted)

        # 记录每个 epoch 的学习率
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # 打印和显示结果
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"训练损失: {epoch_loss:.4f}, "
              f"测试准确率: {test_accuracy:.4f}, "
              f"学习率: {optimizer.param_groups[0]['lr']:.6f}")

    # 创建一个 DataFrame 来存储结果
    results_df = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Loss function': train_losses,
        'Learning accuracy': test_accuracies,
        'Learning rate': learning_rates
    })

    # 将结果保存到 CSV 文件
    results_df.to_csv('training_results.csv', index=False)

    # 绘制结果在同一张图中
    plt.figure(figsize=(12, 8))

    # 训练损失曲线
    plt.plot(results_df['Epoch'], results_df['Loss function'], label='Loss function', color='blue') #训练损失
    # 测试准确率曲线 
    plt.plot(results_df['Epoch'], results_df['Learning rate'], label='Learning rate', color='green') #测试准确率
    # 学习率曲线
    plt.plot(results_df['Epoch'], results_df['Learning accuracy'], label='Learning accuracy', color='red') #学习率

    plt.title('training_result')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
    #混淆矩阵
    cm = confusion_matrix(y_test, predicted_list[-1])
    class_totals = cm.sum(axis=1) #计算每个类别的总数，即混淆矩阵中每一行的总和
    confusion_matrix_percentage = (cm.T / class_totals).T #转换为百分比
    plt.imshow(confusion_matrix_percentage, cmap='Blues', interpolation='nearest')
    plt.colorbar() #添加颜色条
    class_names = ["Thumb", "IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger"]
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45) #刻度标签及旋转角度
    plt.yticks(tick_marks, class_names)
    #绘制
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f'{confusion_matrix_percentage[i, j] * 100:.2f}%'
            plt.text(j, i, text, horizontalalignment="center", color="black")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, predicted_list[-1])
    labels = ["Thumb", "IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger"]
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


# 训练和评估模型，并绘制结果
train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=100)

# #绘制损失曲线
# plt.plot(losses, label='Training Loss')
# plt.xlabel('Training Steps')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()



