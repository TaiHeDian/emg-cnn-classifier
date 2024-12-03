
import numpy
from DataSet import TestDataset
from DataSet import TrainDataset
from torch.utils.data import Dataset
from NET import NET_1D_CNN
from NET import NET_1D_CNN_NEW
from NET import NET_1D_CNN_16C
import numpy as np
import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt

use_gpu = torch.cuda.is_available()

STARTTIME = time.time()

train = TrainDataset()
train.__int__()
test = TestDataset()
test.__int__()


each_size = 10
test_size=1

train_dataloader = torch.utils.data.DataLoader(train, batch_size=each_size, shuffle=True)#,num_workers=16多线程，会变慢？？    #, batch_size=each_size
test_dataloader = torch.utils.data.DataLoader(test, batch_size=test_size, shuffle=True)#,num_workers=16



#net = NET_1D_CNN_16C
#1
# NET_1D_CNN_16C= nn.Sequential(
#                     nn.Conv1d(16,9,3),
#                     nn.Conv1d(9,5,3),
#                     nn.Conv1d(5,1,5),
#                     nn.MaxPool1d(2,2),
#                     nn.Linear(256, 64),
#                     nn.ELU(),
#                     nn.Linear(64, 20)
#                     )




# #90
# NET_1D_CNN_16C= nn.Sequential(
#                     nn.Conv1d(16,9,3),
#                     nn.Conv1d(9,5,3),
#                     nn.Conv1d(5,1,5),
#                     #nn.MaxPool1d(2,2),
#                     nn.Linear(512, 320),
#                     nn.ELU(),
#                     nn.Linear(320, 256),
#                     nn.ELU(),
#                     nn.Linear(256, 192),
#                     nn.ELU(),
#                     nn.Linear(192, 128),
#                     nn.ELU(),
#                     nn.Linear(128, 64),
#                     nn.ELU(),
#                     nn.Linear(64, 20)
#                     )


#90
NET_1D_CNN_16C= nn.Sequential(
                    nn.Conv1d(16,9,3),
                    nn.Conv1d(9,5,3),
                    nn.Conv1d(5,1,5),
                    #nn.MaxPool1d(2,2),
                    nn.Linear(512, 320),
                    nn.ELU(),
                    nn.Linear(320, 128),
                    nn.ELU(),
                    nn.Linear(128, 20),
                    )


net = NET_1D_CNN_16C
print(net)

loss_fn = nn.CrossEntropyLoss()


learning_rate = 0.005
epoch = 50

optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

#初始化网络参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.1)
    elif type(m) == nn.Conv1d:
        nn.init.normal_(m.weight, std=0.1)
net.apply(init_weights)                     

T1TIME = time.time()

sunshi=[]
zhunquelv=[]
nsunshi=[]

for i in range(epoch):
    
    a = 0
    ave = 0
    if use_gpu:
        net.cuda()
    net.train()

    for data in train_dataloader:
        train_data, train_label = data
        outputs_label = net(train_data.cuda())
        outputs_label = outputs_label.reshape(1, each_size, 20)
        matrix = [[0 for j in range(20)] for h in range(each_size)]  # 把标签处理为01矩阵
        for k in range(each_size):
            matrix[k][train_label[k]] = 1
        matrix = torch.from_numpy(np.array(matrix).reshape(1, each_size, 20)).to(torch.float32).cuda()
        loss = loss_fn(outputs_label, matrix)
        # print("train_loss:", loss)
        a += 1
        ave = ave + loss
        nave=ave.cpu().tolist()

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 误差反向传播
        optimizer.step()  # 参数更新
    
    
    #准确率计算
    net.eval()
    with torch.no_grad():
        test=[]
        true=[]
        right=0
        for tdata in test_dataloader:
            
            test_data, test_label = tdata
            test_outputs = net(test_data.cuda()).cpu().numpy()
            where=np.where(test_outputs ==np.max(test_outputs))
            test.append(where[2])
            true.append(test_label)
        for ij in range(len(test)):
            if (int(test[ij])==int(true[ij])):
                right=right+1
    zhunquelv.append(right/len(test))
    sunshi.append(ave/a)
    nsunshi.append(nave/a)
    
    print('epoch:',i,"准确率为:", right/len(test),'平均损失:',nave / a)
    # print("准确率为:", right/len(test))
    # print('平均损失:',nave / a)#打印每一轮的平均损失

print('准确率',zhunquelv)
print('损失',nsunshi)

epochs = range(0,epoch)
plt.figure()
plt.subplot(1,2,1)
plt.plot(epochs,zhunquelv,c='green', label="准确率")
plt.subplot(1,2,2)
plt.plot(epochs,nsunshi,c='blue', label="平均损失")
plt.show()


T2TIME = time.time()

torch.save(net, 'CNN.model')            #保存模型
ENDTIME = time.time()

print("训练时间：",T2TIME - T1TIME)
print(ENDTIME - STARTTIME)              #输出耗时



#混淆矩阵模块
net.eval()
with torch.no_grad():
    ntest=[]
    ntrue=[]
    nright=0
    for tdata in test_dataloader:
        test_data, test_label = tdata
        test_outputs = net(test_data.cuda()).cpu().numpy()
        where=np.where(test_outputs ==np.max(test_outputs))
        ntest.append(where[2])
        ntrue.append(test_label)

    print(len(ntest))
    cm=np.zeros((20,20), dtype = int)                       #混淆矩阵初始化
    for i in range(0,len(ntest)):
        cm[int(ntest[i])][int(ntrue[i])]=cm[int(ntest[i])][int(ntrue[i])]+1

    for i in range(0,len(ntest)):
        if (int(ntest[i])==int(ntrue[i])):
            nright=nright+1
    print(cm)

    print("准确率为：", nright/len(ntest))
