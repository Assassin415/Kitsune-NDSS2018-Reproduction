# ↓ 以下为 MIT 许可证原文（法律文本，保留英文，请勿修改）
# Copyright (c) 2017 Yusuke Sugomori
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# 本文件部分代码改编自 Yusuke Sugomori 在 GitHub 上的实现：
# https://github.com/yusugomori/DeepLearning

import sys
import numpy
from KitNET.utils import *
import json

class dA_params:
    def __init__(self,n_visible = 5, n_hidden = 3, lr=0.001, corruption_level=0.0, gracePeriod = 10000, hiddenRatio=None):
        self.n_visible = n_visible# 可见层（输入层）的神经元个数
        self.n_hidden = n_hidden# 隐藏层的神经元个数
        self.lr = lr
        self.corruption_level = corruption_level
        self.gracePeriod = gracePeriod
        self.hiddenRatio = hiddenRatio

class dA:
    def __init__(self, params):
        self.params = params

        if self.params.hiddenRatio is not None:
            self.params.n_hidden = int(numpy.ceil(self.params.n_visible*self.params.hiddenRatio))

        # 下面两个数组用于 0-1 归一化（记录每个特征历史出现过的最大值/最小值）
        self.norm_max = numpy.ones((self.params.n_visible,)) * -numpy.inf
        self.norm_min = numpy.ones((self.params.n_visible,)) * numpy.inf
        self.n = 0

        self.rng = numpy.random.RandomState(1234)

        a = 1. / self.params.n_visible
        self.W = numpy.array(self.rng.uniform(  # 权重矩阵 W 用均匀分布随机初始化
            low=-a,
            high=a,
            size=(self.params.n_visible, self.params.n_hidden)))

        self.hbias = numpy.zeros(self.params.n_hidden)  # 隐藏层偏置初始化为 0
        self.vbias = numpy.zeros(self.params.n_visible)  # 可见层偏置初始化为 0
        self.W_prime = self.W.T


    def get_corrupted_input(self, input, corruption_level):
        assert corruption_level < 1

        return self.rng.binomial(size=input.shape,
                                 n=1,
                                 p=1 - corruption_level) * input

    # 编码：输入 → 隐藏层
    def get_hidden_values(self, input):
        return sigmoid(numpy.dot(input, self.W) + self.hbias)

    # 解码：隐藏层 → 还原输入
    def get_reconstructed_input(self, hidden):
        return sigmoid(numpy.dot(hidden, self.W_prime) + self.vbias)

    def train(self, x):
        self.n = self.n + 1
        # 更新归一化用的最大值 / 最小值
        self.norm_max[x > self.norm_max] = x[x > self.norm_max]
        self.norm_min[x < self.norm_min] = x[x < self.norm_min]

        # 做 0-1 归一化（分母加一个极小的数，避免除以 0）
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)

        if self.params.corruption_level > 0.0:
            tilde_x = self.get_corrupted_input(x, self.params.corruption_level)
        else:
            tilde_x = x
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        L_h2 = x - z
        L_h1 = numpy.dot(L_h2, self.W) * y * (1 - y)

        L_vbias = L_h2
        L_hbias = L_h1
        L_W = numpy.outer(tilde_x.T, L_h1) + numpy.outer(L_h2.T, y)

        self.W += self.params.lr * L_W
        self.hbias += self.params.lr * L_hbias
        self.vbias += self.params.lr * L_vbias
        return numpy.sqrt(numpy.mean(L_h2**2)) # 训练时的 RMSE 重构误差


    def reconstruct(self, x):
        y = self.get_hidden_values(x)
        z = self.get_reconstructed_input(y)
        return z

    def execute(self, x): # 返回 x 的重构误差（RMSE）
        if self.n < self.params.gracePeriod:
            return 0.0
        else:
            # 0-1 归一化
            x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
            z = self.reconstruct(x)
            rmse = numpy.sqrt(((x - z) ** 2).mean()) # 均方误差再开根号 = RMSE
            return rmse


    def inGrace(self):
        return self.n < self.params.gracePeriod
