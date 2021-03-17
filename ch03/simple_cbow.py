# coding:utf-8

import sys

sys.path.append('..')
from common.layers import MatMul, SoftmaxWithLoss
import numpy as np

eps = 0.01


class SimpleCBOW:
    def __init__(self, input_size, hidden_size):
        V, H = input_size, hidden_size

        # 初始化权重矩阵
        W_in = eps * np.random.randn(V, H).astype('f')
        W_out = eps * np.random.randn(H, V).astype('f')

        # 生成神经网络的层
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 保存参数
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        # 将输入层权重保存为成员变量——用作词向量使用
        self.word_vec = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da = da * 0.5
        self.in_layer0.backward(da)
        self.in_layer1.backward(da)
        return None
