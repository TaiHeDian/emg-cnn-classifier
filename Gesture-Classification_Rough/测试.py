import numpy
import torch
import numpy as np
import pandas as pd
a=numpy.array([[1,2,3,4],[4,5,6,8]])
print(a.shape)
print(range(0,5)[0])
print(range(0,5)[4])
print(range(0,5).__len__())


list=[]
list.append(1)
print(list[0])

print(int(295/100))

lists = [[] for i in range(1600)]  # 创建的是多行三列的二维列表

for i in range(1600):
    for k in range(9):
        lists[i].append(i*9+k)

print("lists is:", lists)
print(len(lists))
a=lists[1]
print(lists[1])
print(len(a))

#对dataset的测试
# train_iter =np.array(pd.read_csv(r'train_data.csv', header=None), dtype=np.float32).T
# print(train_iter[0][0])
# print(len(train_iter[lists[1]]))
# b=train_iter[lists[1]]
# print(b)



matrix = [[0 for i in range(16)] for i in range(1)]#把标签处理为01矩阵
for i in range(1):
     matrix[i][6] = 1
matrix=numpy.array(matrix)
matrix=matrix.reshape(1,1,1,16)
matrix=torch.from_numpy(matrix)
print(matrix)
