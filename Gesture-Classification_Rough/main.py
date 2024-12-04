import numpy
from DataSet import TestDataset
from DataSet import TrainDataset
from torch.utils.data import Dataset
from NET import NET_1D_CNN
from NET import NET_1D_CNN_NEW
from NET import NET_1D_CNN_16C
# from NET import init_weights
import numpy as np
import torch.nn as nn
import torch
import time

use_gpu = torch.cuda.is_available()

STARTTIME = time.time()

train = TrainDataset()
train.__int__()
test = TrainDataset()
test.__int__()


each_size = 10
test_size=1
train_dataloader = torch.utils.data.DataLoader(train, batch_size=each_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test, batch_size=test_size, shuffle=False)
net = NET_1D_CNN_16C
# net = NET_1D_CNN_NEW
print(net)

#loss_fn = nn.MSELoss()
#loss_fn = nn.KLDivLoss(size_average=None, reduce=None, reduction='elementwise_mean')
loss_fn = nn.CrossEntropyLoss()
#loss_fn = nn.NLLLoss()

learning_rate = 0.02
epoch = 1
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)



def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.1)
    elif type(m) == nn.Conv1d:
        nn.init.normal_(m.weight, std=0.1)
net.apply(init_weights)                     #初始化网络参数

T1TIME = time.time()

sunshi=[]
zhunquelv=[]

for i in range(epoch):
    print(i)
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
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 误差反向传播
        optimizer.step()  # 参数更新
    

    net.eval()
    with torch.no_grad():
        test=[]
        true=[]
        right=0
        for tdata in test_dataloader:
            test_data, test_label = tdata
            # print(test_data)
            #print(test_label)
            test_outputs = net(test_data.cuda()).cpu().numpy()
            #print([test_outputs])
            where=np.where(test_outputs ==np.max(test_outputs))
            #print(where[2])
            test.append(where[2])
            true.append(test_label)
        for i in range(len(test)):
            if (int(test[i])==int(true[i])):
                right=right+1
    zhunquelv.append(right/len(test))
    sunshi.append(ave/a)
    print("准确率为：", right/len(test))
    print(ave / a)#打印每一轮的平均损失


print(zhunquelv)
print(sunshi)

T2TIME = time.time()
net.eval()
with torch.no_grad():
    test=[]
    true=[]
    right=0
    for tdata in test_dataloader:
        test_data, test_label = tdata
        # print(test_data)
        #print(test_label)
        test_outputs = net(test_data.cuda()).cpu().numpy()
        #print([test_outputs])
        where=np.where(test_outputs ==np.max(test_outputs))
        #print(where[2])
        test.append(where[2])
        true.append(test_label)
    ntest=np.array(test)
    ntrue=np.array(true)
    cm=np.zeros((20,20), dtype = int)
    # for j in range(0,len(test)):
    #     cm[test[j]][true[j]]=test[test[j]][true[j]]+1
    print(ntest[1])
    print(ntrue[1])
    print('混淆矩阵：')
    print(cm)
    for i in range(0,len(test)):
        if (int(test[i])==int(true[i])):
            right=right+1
    

    print("准确率为：", right/len(test))

torch.save(net, 'CNN.model')            #保存模型
ENDTIME = time.time()

print("训练时间：",T2TIME - T1TIME)
print(ENDTIME - STARTTIME)              #输出耗时
