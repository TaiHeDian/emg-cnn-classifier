import torch.nn as nn
NET_1D_CNN = nn.Sequential(
                    nn.Conv1d(9,7,3),
                    nn.Conv1d(7,5,5),
                    nn.AvgPool1d(2,2),
                    nn.Conv1d(5,3,3),
                    nn.Conv1d(3,1,5),
                    nn.MaxPool1d(2,2),

                    nn.Linear(120, 64),
                    nn.ELU(),
                    nn.Linear(64, 36),
                    nn.ELU(),
                    nn.Linear(36, 16)
                    )
NET_1D_CNN_NEW = nn.Sequential(
                    nn.Conv1d(9,5,3),
                    nn.Conv1d(5,1,5),
                    nn.MaxPool1d(2,2),
                    nn.Linear(247, 64),
                    nn.ELU(),
                    nn.Linear(64, 16)
                    )

#demo1
NET_1D_CNN_16C_demo1 = nn.Sequential(
                    nn.Conv1d(16,9,3),
                    nn.Conv1d(9,5,3),
                    nn.Conv1d(5,1,5),
                    nn.MaxPool1d(2,2),
                    nn.Linear(256, 64),
                    nn.ELU(),
                    nn.Linear(64, 20)
                    )

# #demo2
# NET_1D_CNN_16C = nn.Sequential(
#                     nn.Conv1d(16,12,5),
#                     nn.Conv1d(12,8,3),

#                     nn.Conv1d(8,5,5),
#                     nn.Conv1d(5,1,3),

#                     nn.MaxPool1d(2,2),
#                     nn.Linear(254, 64),
#                     nn.ELU(),
#                     nn.Linear(64, 20)
#                     )

#demo3
NET_1D_CNN_16C = nn.Sequential(
                    nn.Conv1d(16,12,5),
                    nn.Conv1d(12,8,3),
                    nn.MaxPool1d(2,2),

                    nn.Conv1d(8,5,5),
                    nn.Conv1d(5,1,3),

                    nn.MaxPool1d(2,2),
                    nn.Linear(254, 64),
                    nn.ELU(),
                    nn.Linear(64, 20)
                    )




#2
# NET_1D_CNN_16C= nn.Sequential(
#                     nn.Conv1d(16,9,3),
#                     nn.Conv1d(9,5,3),
#                     nn.Conv1d(5,1,3),
#                     nn.MaxPool1d(2,2),
#                     nn.Linear(257, 64),
#                     nn.ELU(),
#                     nn.Linear(64, 20)
#                     )
# net = NET_1D_CNN_16C
# #4
# NET_1D_CNN_16C= nn.Sequential(
#                     nn.Conv1d(16,13,3),
#                     nn.Conv1d(13,11,3),
#                     nn.Conv1d(11,7,3),
#                     nn.Conv1d(7,5,3),
#                     nn.Conv1d(5,3,3),
#                     nn.Conv1d(3,1,5),
#                     nn.MaxPool1d(2,2),
#                     nn.Linear(253, 64),
#                     nn.ELU(),
#                     nn.Linear(64, 20)
#                     )
