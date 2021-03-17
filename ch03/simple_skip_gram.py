import sys

sys.path.append('..')
from common.layers import MatMul, SoftmaxWithLoss
import numpy as np

eps = 1e-2


class SimpleSkipGram:
    '''
    仅支持 context_size=1 的时候的 skip gram 模型
    '''

    def __init__(self, input_size, hidden_size):
        V, H = input_size, hidden_size
        # 1. 初始化权重
        W_in = eps * np.random.randn(V, H).astype('f')
        W_out = eps * np.random.randn(H, V).astype('f')
        # 2. 初始化层
        self.in_layer = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer1 = SoftmaxWithLoss()
        self.loss_layer2 = SoftmaxWithLoss()
        # 3. 保存参数
        layers = [self.in_layer, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vec = W_in

    def forward(self, context, target):
        h = self.in_layer.forward(target)
        h = self.out_layer.forward(h)
        loss1 = self.loss_layer1.forward(h, context[:, 0])
        loss2 = self.loss_layer2.forward(h, context[:, 1])
        return loss1 + loss2

    def backward(self, dout=1):
        dout = dout * 0.5
        ds1 = self.loss_layer1.backward(dout)
        ds2 = self.loss_layer2.backward(dout)
        da = self.out_layer.backward(ds1 + ds2)
        self.in_layer.backward(da)
        return None
