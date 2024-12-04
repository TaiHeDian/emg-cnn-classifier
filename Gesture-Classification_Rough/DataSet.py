
'''
这部分为Dataset的基本写作过程参照以下网站：
https://zhuanlan.zhihu.com/p/105507334
https://zhuanlan.zhihu.com/p/105578087
https://zhuanlan.zhihu.com/p/466699075
https://zhuanlan.zhihu.com/p/500839903
'''
# class CustomDataset(data.Dataset):#需要继承data.Dataset
#     def __init__(self):
#         # TODO
#         # 1. Initialize file path or list of file names.
#         pass
#     def __getitem__(self, index):
#         # TODO
#         # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
#         # 2. Preprocess the data (e.g. torchvision.Transform).
#         # 3. Return a data pair (e.g. image and label).
#         #这里需要注意的是，第一步：read one data，是一个data
#         pass
#     def __len__(self):
#         # You should change 0 to the total size of your dataset.
#         return 0
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __int__(self):
        self.train_iter =np.array(pd.read_csv(r'train.csv', header=None), dtype=np.float32).T
        print('训练集数据读取完成')
    def __getitem__(self,indx):
        labellist=[]
        datalist = [[] for i in range(2400)]  # 创建的是多行三列的二维列表16*100,这里是20*120。20是手势数量，120是数据量大小
        for i in range(2400):
            labellist.append(int(i / 120))  # 这里的120指的是有一个标签中有120个数据，测试集再写一个; int自动向下取整
            for k in range(16):
                datalist[i].append(i * 16 + k)#通道数
        one_data = self.train_iter[datalist[indx]]#一组数据
        one_label=labellist[indx]#一个标签
        return one_data,one_label

    def __len__(self):#20*120
        return 2400


class TestDataset(Dataset):
    def __int__(self):
        self.train_iter =np.array(pd.read_csv(r'test.csv', header=None), dtype=np.float32).T
        print('测试集数据读取完成')
    def __getitem__(self,indx):
        labellist=[]
        datalist = [[] for i in range(600)]  # 创建的是多行三列的二维列表
        for i in range(600):
            labellist.append(int(i / 30))  # 这里的100指的是有一个标签中有100个数据，测试集再写一个; int自动向下取整
            for k in range(16):
                datalist[i].append(i * 16 + k)
        one_data = self.train_iter[datalist[indx]]#一组数据
        one_label=labellist[indx]#一个标签
        return one_data,one_label

    def __len__(self):#20*30
        return 600





