
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
import os
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载和预处理

# 数据加载和预处理
data = []
labels = []

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取包含文件的目录路径
current_directory = os.path.dirname(current_file_path)

last_path = [r'12345',
              r'23456', 
              r'32416', 
              r'45671',
              r'65472', ]

excel_path  = [os.path.join(current_directory, path) for path in last_path]


startTime = 3#31000
thresholds = [1E-4, 1E-4, 1E-4, 1E-4, 1E-4, 1E-4, 1E-4, 1E-4, 1E-4, 1E-4] #阈值

    
print(excel_path)
n=0
for data_dir in excel_path:
    dir_name = os.listdir(data_dir)
    print(dir_name)
    First_path = []
    data_path = []  #数据集
    Data_index = 0
    label_list = [] #标签集
    #进入每一个文件夹，
    for index in range(len(dir_name)):
        temp_path = os.path.join(data_dir,dir_name[index])   
        df = pd.read_csv(temp_path, header=None)
        # 获取信号数据
        sample = df.iloc[1:, 0:7].values #提取b列信号
 


                    
        # #可视化每个样本
        # plt.figure(figsize=(10, 6))
        # for col in range(sample.shape[1]):
        #     plt.plot(sample[:, col], label=f'Column {col + 1}')
        # plt.show()
        # plt.close()            
        # #检查sample是否包含NaN


        #归一化数据
        scaler = StandardScaler()
        sample = scaler.fit_transform(sample)
        sample = sample.reshape(-1, order='F').tolist()            # 将样本添加到列表中

        #print(len(sample))
        data.append(sample)
        labels.append(n)  # i // 2 - 1
    n+=1



        
def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


# 数据格式化
# data = [torch.tensor(arr, dtype=torch.float32) for arr in data]
data = torch.tensor(data, dtype=torch.float32)
data = data.unsqueeze(1)  # 添加一个通道维度
# data = data.transpose(0,2,1)
print(data.size())
print(data)

print(labels)

labels = torch.tensor(labels, dtype=torch.long)
print(labels.size())
# 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=random.randint(1,100))#42
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
        
        input_size = [32, 1, 560]
        self.conv1_num = 8
        self.conv2_num = 16
        self.conv3_num = 32

        
        self.conv1 = nn.Conv1d(1, self.conv1_num, kernel_size=3, padding=1) # [批量大小, 32, 时间步数]  2 个单位的 padding，确保卷积后的长度不变     
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2) # [批量大小, 32, 时间步数 / 2 ]       
        self.conv2 = nn.Conv1d(self.conv1_num, self.conv2_num, kernel_size=3, padding=1) # [批量大小, 64, 时间步数]     
        self.pool2 = nn.MaxPool1d(kernel_size=2) # [批量大小, 64, 时间步数 / 2 ]
        self.conv3 = nn.Conv1d(self.conv2_num, self.conv3_num, kernel_size=3, padding=1) #[批量大小, 128, 时间步数]
        


        input_size[2] = (input_size[2] + 2 * self.conv1.padding[0] - self.conv1.kernel_size[0]) // 1 + 1
        input_size = [input_size[0], self.conv1_num, input_size[2]]        
        input_size = [input_size[0], input_size[1], input_size[2] // 2]        
        input_size[2] = (input_size[2] + 2 * self.conv2.padding[0] - self.conv2.kernel_size[0]) // 1 + 1
        input_size = [input_size[0], self.conv2_num, input_size[2]]        
        input_size = [input_size[0], input_size[1], input_size[2] // 2]        
        input_size[2] = (input_size[2] + 2 * self.conv3.padding[0] - self.conv3.kernel_size[0]) // 1 + 1
        input_size = [input_size[0], self.conv3_num, input_size[2]]
        # num = 250
        # for _ in range(3):  # 三个池化层
        #     num = (num + 1) // 2  # 池化层缩小时间步数的一半，加1是为了向上取整
        self.num_features = input_size[1] * input_size[2]
    
        self.fc1 = nn.Linear(self.num_features, 64)  # 调整全连接层神经元数量 
        self.fc2 = nn.Linear(64, 5)
        
        #self.dropout = nn.Dropout(0.2)  # 添加Dropout层

    def forward(self, x):
        #print("Input shape:", x.shape)
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        #print("After conv1 and pool1 shape:", x.shape)
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        #print("After conv2 and pool2 shape:", x.shape)
        x = nn.functional.relu(self.conv3(x))
        #print("After conv3 and pool3 shape:", x.shape)
        x = x.view(x.size(0), -1)
        #print("After view shape:", x.shape)
        x = self.relu(self.fc1(x))
        #print("After fc1, relu, and dropout shape:", x.shape)
        x = self.fc2(x)
        #x = self.dropout(x)
        
        #print("Output shape:", x.shape)
        return x


# 创建模型、优化器和损失函数
model = CNNModel().to(device)
#optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.0001) #SGD不行
#scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = nn.CrossEntropyLoss()
#model.apply(weights_init)
summary(model, input_size=(1, 560))


# 训练模型
# def train_model(model, train_loader, criterion, optimizer):
#     model.train()
#     running_loss = 0.0
    
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device).float(), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         print(loss)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#      # 添加调试信息
#     print(f"Running loss: {running_loss}")
#     # 返回平均训练损失
#     # 防止除以零
#     if len(train_loader) > 0:
#         # 返回平均训练损失
#         return running_loss / len(train_loader)
#     else:
#         # 如果 len(train_loader) 为零，返回一个标记值
#         return -1.0  # 你可以根据需要修改为其他标记值

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
        #scheduler.step()
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
                if torch.isnan(param.grad).any():
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
              f"训练损失: {epoch_loss:.8f}, "
              f"测试准确率: {test_accuracy:.8f}, "
              f"学习率: {optimizer.param_groups[0]['lr']:.8f}")

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
    class_names = ["12345", "23456", "32416", "45671", "65472"]
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
    labels = ["12345", "23456", "32416", "45671", "65472"]
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





