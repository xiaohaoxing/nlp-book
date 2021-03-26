# coding: utf-8

import sys

sys.path.append('..')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc("font", family='Mi Lanting')

# mini-batch 大小
N = 2
# 隐藏向量的维度
H = 3
# 时序数据的长度
T = 20

dh = np.ones((N, H))
np.random.seed(3)
Wh = np.random.randn(H, H) * 0.5

norm_list = []
for t in range(T):
    dh = np.dot(dh, Wh.T)
    norm = np.sqrt(np.sum(dh ** 2)) / N
    norm_list.append(norm)

plt.plot(norm_list)
plt.xlabel('时间步长')
plt.ylabel('梯度范数')
plt.show()
