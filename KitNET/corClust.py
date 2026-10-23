import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, to_tree

# KitNET 的辅助类：对数据集的各个维度做"基于相关性的增量聚类"
# n: 数据集的维度数
# 详细说明和引用请见我们的 NDSS'18 论文：
#   Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection
class corClust:
    def __init__(self,n):
        # 参数
        self.n = n
        # 变量
        self.c = np.zeros(n) # 每个特征值的累加和
        self.c_r = np.zeros(n) # 残差（特征值减当前均值）的累加和
        self.c_rs = np.zeros(n) # 残差平方的累加和
        self.C = np.zeros((n,n)) # 两个特征残差乘积的累加和（用来算相关系数）
        self.N = 0 # 已经更新过多少次

    # x: 长度为 n 的 numpy 向量
    def update(self,x):
        self.N += 1
        self.c += x
        c_rt = x - self.c/self.N
        self.c_r += c_rt
        self.c_rs += c_rt**2
        self.C += np.outer(c_rt,c_rt)

    # 计算当前特征之间的"相关性距离"矩阵
    def corrDist(self):
        c_rs_sqrt = np.sqrt(self.c_rs)
        C_rs_sqrt = np.outer(c_rs_sqrt,c_rs_sqrt)
        C_rs_sqrt[C_rs_sqrt==0] = 1e-100 # 防止除以 0（某个特征一直不变时会出现）
        D = 1-self.C/C_rs_sqrt # 相关性距离矩阵：用 1 减去相关系数
        D[D<0] = 0 # 均值是增量更新的，会引入很小的负值，这里统一修正为 0
        return D

    # 把特征聚类，保证每一簇的特征个数不超过 maxClust
    def cluster(self,maxClust):
        D = self.corrDist()
        Z = linkage(D[np.triu_indices(self.n, 1)])  # 用距离矩阵的上三角做层次聚类
        if maxClust < 1:
            maxClust = 1
        if maxClust > self.n:
            maxClust = self.n
        map = self.__breakClust__(to_tree(Z),maxClust)
        return map

    # 递归辅助函数：不断切分聚类树的分支，直到每一簇的元素数都不超过 maxClust
    def __breakClust__(self,dendro,maxClust):
        if dendro.count <= maxClust: # 递归出口：这一簇已经足够小，可以收下了
            return [dendro.pre_order()] # 返回这一簇里各特征在原始特征向量中的下标
        return self.__breakClust__(dendro.get_left(),maxClust) + self.__breakClust__(dendro.get_right(),maxClust)

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
