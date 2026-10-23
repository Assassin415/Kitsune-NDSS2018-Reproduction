import numpy as np
## 准备 AfterImage 的 Cython 包（pyximport 让 .pyx 文件可以像普通模块一样导入）
import os
import subprocess
import pyximport
pyximport.install()
import AfterImage as af
#import AfterImage_NDSS as af   # 备用：论文原版实现（本仓库没有这个文件）

#
# ↓ 以下为 MIT 许可证原文（法律文本，保留英文，请勿修改）
# MIT License
#
# Copyright (c) 2018 Yisroel mirsky
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class netStat:
    # 用于高效查询网络统计量的数据结构
    # HostLimit: 最多跟踪多少个主机标识
    # HostSimplexLimit: 每个主机最多跟踪多少条对外通道（超出的会被定期清理）
    # Lambdas: 每个流要跟踪的"时间窗"列表（其实是衰减因子 λ）。传 nan 时用默认值 [5,3,1,.1,.01]
    def __init__(self, Lambdas = np.nan, HostLimit=255,HostSimplexLimit=1000):
        # 时间窗（衰减因子 λ）
        if np.isnan(Lambdas):
            self.Lambdas = [5,3,1,.1,.01]
        else:
            self.Lambdas = Lambdas

        # 各哈希表（HT）的容量上限
        self.HostLimit = HostLimit
        self.SessionLimit = HostSimplexLimit*self.HostLimit*self.HostLimit # 每条双向会话会在内存里生成 2 条记录，所以这里已经放大
        self.MAC_HostLimit = self.HostLimit*10

        # 四张统计表（对应四个通道）
        self.HT_jit = af.incStatDB(limit=self.HostLimit*self.HostLimit)# 主机对之间的到达间隔（jitter）统计
        self.HT_MI = af.incStatDB(limit=self.MAC_HostLimit)# 源 MAC 与源 IP 的绑定关系统计
        self.HT_H = af.incStatDB(limit=self.HostLimit) # 源 IP 的发送统计 / 主机对之间的双向统计
        self.HT_Hp = af.incStatDB(limit=self.SessionLimit)# 源-目的 socket 对的统计


    def findDirection(self,IPtype,srcIP,dstIP,eth_src,eth_dst): # C++ 版里方向信息直接写在实例字符串里，这个函数用不到
        if IPtype==0: # 是 IPv4
            lstP = srcIP.rfind('.')
            src_subnet = srcIP[0:lstP:]
            lstP = dstIP.rfind('.')
            dst_subnet = dstIP[0:lstP:]
        elif IPtype==1: # 是 IPv6
            src_subnet = srcIP[0:round(len(srcIP)/2):]
            dst_subnet = dstIP[0:round(len(dstIP)/2):]
        else: # 没有网络层，用 MAC 地址代替
            src_subnet = eth_src
            dst_subnet = eth_dst

        return src_subnet, dst_subnet

    def updateGetStats(self, IPtype, srcMAC,dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, datagramSize, timestamp):
        # ⚠️ 下面这段是「源 IP 单独统计」通道，被作者注释掉了（所以特征只有 100 维而不是论文的 115 维）
        # 作者在 issue #12 中解释：这一通道与下面的 MAC-IP 通道高度冗余，所以移除
        # Hstat = np.zeros((3*len(self.Lambdas,)))
        # for i in range(len(self.Lambdas)):
        #     Hstat[(i*3):((i+1)*3)] = self.HT_H.update_get_1D_Stats(srcIP, timestamp, datagramSize, self.Lambdas[i])

        # MI 通道：源 MAC + 源 IP 这个"设备"的发送统计（权重、均值、方差 × 每个 λ）
        MIstat =  np.zeros((3*len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            MIstat[(i*3):((i+1)*3)] = self.HT_MI.update_get_1D_Stats(srcMAC+srcIP, timestamp, datagramSize, self.Lambdas[i])

        # HH 通道：源 IP ↔ 目的 IP 这对主机之间的流量行为（7 个统计量：w、均值、方差、幅度、半径、协方差、相关系数）
        HHstat =  np.zeros((7*len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            HHstat[(i*7):((i+1)*7)] = self.HT_H.update_get_1D2D_Stats(srcIP, dstIP,timestamp,datagramSize,self.Lambdas[i])

        # HH_jit 通道：源 IP → 目的 IP 的包到达时间间隔（本身也是一个统计对象）
        HHstat_jit =  np.zeros((3*len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            HHstat_jit[(i*3):((i+1)*3)] = self.HT_jit.update_get_1D_Stats(srcIP+dstIP, timestamp, 0, self.Lambdas[i],isTypeDiff=True)

        # HpHp 通道：源 socket ↔ 目的 socket 这一对之间的流量行为
        HpHpstat =  np.zeros((7*len(self.Lambdas,)))
        if srcProtocol == 'arp':
            for i in range(len(self.Lambdas)):
                HpHpstat[(i*7):((i+1)*7)] = self.HT_Hp.update_get_1D2D_Stats(srcMAC, dstMAC, timestamp, datagramSize, self.Lambdas[i])
        else:  # 其他协议（例如 TCP/UDP）：用 "IP+端口" 作为 socket 的标识
            for i in range(len(self.Lambdas)):
                HpHpstat[(i*7):((i+1)*7)] = self.HT_Hp.update_get_1D2D_Stats(srcIP + srcProtocol, dstIP + dstProtocol, timestamp, datagramSize, self.Lambdas[i])

        return np.concatenate((MIstat, HHstat, HHstat_jit, HpHpstat))  # 把这几个统计量拼成一个特征向量

    def getNetStatHeaders(self):
        MIstat_headers = []
        Hstat_headers = []
        HHstat_headers = []
        HHjitstat_headers = []
        HpHpstat_headers = []

        for i in range(len(self.Lambdas)):
            MIstat_headers += ["MI_dir_"+h for h in self.HT_MI.getHeaders_1D(Lambda=self.Lambdas[i],ID=None)]
            HHstat_headers += ["HH_"+h for h in self.HT_H.getHeaders_1D2D(Lambda=self.Lambdas[i],IDs=None,ver=2)]
            HHjitstat_headers += ["HH_jit_"+h for h in self.HT_jit.getHeaders_1D(Lambda=self.Lambdas[i],ID=None)]
            HpHpstat_headers += ["HpHp_" + h for h in self.HT_Hp.getHeaders_1D2D(Lambda=self.Lambdas[i], IDs=None, ver=2)]
        return MIstat_headers + Hstat_headers + HHstat_headers + HHjitstat_headers + HpHpstat_headers
