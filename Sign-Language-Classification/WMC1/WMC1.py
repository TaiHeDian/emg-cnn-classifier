
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
from sklearn.preprocessing import StandardScaler
import random


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载和预处理
data = []
labels = []

excel_path = [r'D:\DeepLearning\WMC代码\分组前数据\A.xlsx',
              r'D:\DeepLearning\WMC代码\分组前数据\H.xlsx', 
              r'D:\DeepLearning\WMC代码\分组前数据\M.xlsx', 
              r'D:\DeepLearning\WMC代码\分组前数据\N.xlsx',
              r'D:\DeepLearning\WMC代码\分组前数据\U.xlsx', ]
excel_path1 = [r'D:\DeepLearning\WMC代码\分组前数据\A',
              r'D:\DeepLearning\WMC代码\分组前数据\H', 
              r'D:\DeepLearning\WMC代码\分组前数据\M', 
              r'D:\DeepLearning\WMC代码\分组前数据\N',
              r'D:\DeepLearning\WMC代码\分组前数据\U', ]
startTime = 3#31000
thresholds = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000] #阈值

    


def replace_nan_with_zero_in_list(input_list):
    # 遍历列表中的每个元素
    for i in range(len(input_list)):
        # 检查元素是否为NaN
        if isinstance(input_list[i], float) and pd.isna(input_list[i]):
            # 将NaN替换为0
            input_list[i] = 0
 
    return input_list

for i in range(0, 5):
    df = pd.read_excel(excel_path[i], header=None)
    # 获取信号数据
    signal = df.iloc[:, 1:6].values #提取b列信号
    n = 0 #该excel中提取到的样本数
    end_index = 0
    # 扫描信号
    j=0
    for j in range(len(signal)):                
        if j<3:
            continue
        if j<end_index:
            continue
        signal_row = df.iloc[j, 1:5].values
        if max(signal_row) > thresholds[i]:
            # 找到波峰的起始点
            start_index = j
            while j < len(signal) and max(signal_row) > thresholds[i]:
                j += 1 
                signal_row = df.iloc[j, 1:5].values
            # 找到波峰的结束点
            end_index = j

            # 取波峰中心点，左右各取相同长度
            center_index = (start_index + end_index) // 2
            half_length = 25  # 左右各取5000个点
            if center_index - half_length < 0 or center_index + half_length > len(signal): 
                break 
            start_index =  center_index - half_length
            end_index =  center_index + half_length
            
            j = end_index

            # 提取样本
            sample = df.iloc[start_index:end_index, 1:6].values

            # 归一化数据
            scaler = StandardScaler()
            sample = scaler.fit_transform(sample)
            # sample_df.to_csv(csv_filename, index=False)
            sample = sample.reshape(-1, order='F').tolist()            # 将样本添加到列表中
            if pd.isna(sample).any():                            
               
                sample =  replace_nan_with_zero_in_list(sample)
                #print(sample)
            #print(len(sample))
            data.append(sample)
            labels.append(i)  # i // 2 - 1
            
            n +=1
            
            # #可视化每个样本
            # plt.figure(figsize=(10, 6))
            # for col in range(sample.shape[1]):
            #     plt.plot(sample[:, col], label=f'Column {col + 1}')
            # plt.show()
            # plt.close()
        if n>=200:
            break
        
def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


data = torch.tensor(data, dtype=torch.float32)
data = data.unsqueeze(1)  # 添加一个通道维度
# data = data.transpose(0,2,1)
print(data.size())
print(labels)

labels = torch.tensor(labels, dtype=torch.long)
print(labels.size())

# 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=random.randint(1,100))
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
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=4, padding=2) 
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)    
        self.conv2 = nn.Conv1d(64, 128, kernel_size=4, padding=2) 
        self.pool2 = nn.MaxPool1d(kernel_size=2) 
        self.conv3 = nn.Conv1d(128, 256, kernel_size=4, padding=2) 
        self.pool3 = nn.MaxPool1d(kernel_size=2)         
        self.fc1 = nn.Linear(8064, 128) 
        self.fc2 = nn.Linear(128, 5)
        
    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 创建模型、优化器和损失函数
model = CNNModel().to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001) #SGD不行
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
criterion = nn.CrossEntropyLoss()
model.apply(weights_init)
summary(model, input_size=(1, 250))



#测试模型
def test_model(model, test_loader):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    all_predicted = []  # 用于存储所有预测结果
    # predicted_last = torch.Tensor()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device)
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
        scheduler.step()
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
                # 检查输入数据是否包含 NaN 或 Inf
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print("输入数据包含 NaN 或 Inf。")
                continue
            inputs, labels = inputs.to(device).float(), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print(loss)
            loss.backward()
            
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print("梯度包含 NaN 值。")    
                
            optimizer.step()
            running_loss += loss.item()

             # 添加调试信息
        print(f"Running loss: {running_loss}")

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
        'Loss fuction': train_losses,
        'Learning accuracy': test_accuracies,
        'Learning rate': learning_rates
    })

    # 将结果保存到 CSV 文件
    results_df.to_csv('training_results.csv', index=False)

    # 绘制结果在同一张图中
    plt.figure(figsize=(12, 8))

    # 训练损失曲线
    plt.plot(results_df['Epoch'], results_df['Loss fuction'], label='Loss fuction', color='blue') #训练损失
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
    class_names = ["A", "H", "M", "N", "U"]
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
    labels = ["A", "H", "M", "N", "U"]
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


# 训练和评估模型，并绘制结果
train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=150)

# #绘制损失曲线
# plt.plot(losses, label='Training Loss')
# plt.xlabel('Training Steps')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()




