import numpy as np
import KitNET.dA as AE
import KitNET.corClust as CC

# 这个类就是 KitNET 学习器本身。
# KitNET 是一种轻量的在线异常检测算法，核心是"一组自编码器的集成"。
# 详细说明和引用请见我们的 NDSS'18 论文：
#   Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection
# 许可证信息见本文件末尾

class KitNET:
    # n: 输入数据的特征个数（x 属于 R^n）
    # m: 集成层里每个自编码器允许的最大输入维度（论文里的 m）
    # AD_grace_period: 输出异常分数之前，网络要先学习的实例个数
    # FM_grace_period: 用来学习特征映射的实例个数；如果为 None，就默认等于 AD_grace_period
    # learning_rate: 所有自编码器共用的随机梯度下降学习率
    # hidden_ratio: 隐藏层神经元与输入层神经元的比例。例如 0.75 表示隐藏层压缩约 25%
    # feature_map: 可以选择不学习、直接传入一个特征映射。它必须是一个列表，
    #           第 i 个元素给出分配给第 i 个自编码器的特征下标列表，
    #           例如 [[2,5,3],[4,0,1],[6,7]]
    def __init__(self,n,max_autoencoder_size=10,FM_grace_period=None,AD_grace_period=10000,learning_rate=0.1,hidden_ratio=0.75, feature_map = None):
        # 参数
        self.AD_grace_period = AD_grace_period
        if FM_grace_period is None:
            self.FM_grace_period = AD_grace_period
        else:
            self.FM_grace_period = FM_grace_period
        if max_autoencoder_size <= 0:
            self.m = 1
        else:
            self.m = max_autoencoder_size
        self.lr = learning_rate
        self.hr = hidden_ratio
        self.n = n

        # 内部变量
        self.ensembleLayer = []
        self.outputLayer = None
        self.n_trained = 0 # 到目前为止用于训练的实例数
        self.n_executed = 0 # 到目前为止做过检测的实例数
        self.v = feature_map
        if self.v is None:
            print("Feature-Mapper: train-mode, Anomaly-Detector: off-mode")
        else:
            self.__createAD__()
            print("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode")
        self.FM = CC.corClust(self.n) # 用于特征映射的增量式相关性聚类

    # 如果 FM_grace_period + AD_grace_period 已经过去，就对 x 做检测；否则拿 x 继续学习。
    # x: 长度为 n 的 numpy 数组
    # 注意：KitNET 会自动对所有属性做 0-1 归一化
    def process(self,x):
        if self.n_trained > self.FM_grace_period + self.AD_grace_period: # 特征映射和检测器都已进入"检测模式"
            return self.execute(x)
        else:
            self.train(x)
            return 0.0

    # 强制用 x 训练 KitNET
    # 返回训练期间 x 的异常分数（这个值不要用来报警）
    def train(self,x):
        if self.n_trained <= self.FM_grace_period and self.v is None: # 特征映射还在训练期，且用户没有自己提供映射
            # 更新增量式的相关性矩阵
            self.FM.update(x)
            if self.n_trained == self.FM_grace_period: # 时间到：该把特征映射固定下来了
                self.v = self.FM.cluster(self.m)
                self.__createAD__()
                print("The Feature-Mapper found a mapping: "+str(self.n)+" features to "+str(len(self.v))+" autoencoders.")
                print("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode")
        else: # 训练
            ## 集成层：每个自编码器各自训练
            S_l1 = np.zeros(len(self.ensembleLayer))
            for a in range(len(self.ensembleLayer)):
                # 取出第 a 个自编码器负责的那几个特征，拼成子向量
                xi = x[self.v[a]]
                S_l1[a] = self.ensembleLayer[a].train(xi)
            ## 输出层
            self.outputLayer.train(S_l1)
            if self.n_trained == self.AD_grace_period+self.FM_grace_period:
                print("Feature-Mapper: execute-mode, Anomaly-Detector: execute-mode")
        self.n_trained += 1

    # 强制对 x 做一次检测（只前向传播，不更新参数）
    def execute(self,x):
        if self.v is None:
            raise RuntimeError('KitNET Cannot execute x, because a feature mapping has not yet been learned or provided. Try running process(x) instead.')
        else:
            self.n_executed += 1
            ## 集成层
            S_l1 = np.zeros(len(self.ensembleLayer))
            for a in range(len(self.ensembleLayer)):
                # 取出这一组对应的子向量
                xi = x[self.v[a]]
                S_l1[a] = self.ensembleLayer[a].execute(xi)
            ## 输出层
            return self.outputLayer.execute(S_l1)

    def __createAD__(self):
        # 构建集成层：特征映射里每一组特征对应一个自编码器
        for map in self.v:
            params = AE.dA_params(n_visible=len(map), n_hidden=0, lr=self.lr, corruption_level=0, gracePeriod=0, hiddenRatio=self.hr)
            self.ensembleLayer.append(AE.dA(params))

        # 构建输出层：它的输入是"各集成层自编码器的误差"，输出是最终异常分数
        params = AE.dA_params(len(self.v), n_hidden=0, lr=self.lr, corruption_level=0, gracePeriod=0, hiddenRatio=self.hr)
        self.outputLayer = AE.dA(params)

# Copyright (c) 2017 Yisroel Mirsky
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
