from Kitsune import Kitsune
import numpy as np
import time

##############################################################################
# Kitsune：基于自编码器集成（KitNET）的轻量级在线网络入侵检测系统。
# 详细说明和引用请见我们的 NDSS'18 论文：
#   Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection

# 这个脚本演示 Kitsune 如何在 Mirai 恶意软件的抓包上增量学习并检测异常。
# 演示数据是一个 m×n 的矩阵：n = 115 维特征，m = 100,000 个观测（数据包）。
# 每个观测都是网络状态的一张快照，由增量阻尼统计量构成（详见 NDSS 论文）。

# 注意：论文里给出的运行速度来自 C++ 实现，比这个 Python 实现大约快 100 倍。
###################  最后测试环境：Anaconda 3.6.3   #######################

# 加载 Mirai 抓包（记录了 Mirai 僵尸网络被激活的过程）
# 前 70,000 个观测是干净流量（注：本仓库自带的抓包实际从第 12 万多个包才开始异常）
print("Unzipping Sample Capture...")
import zipfile
with zipfile.ZipFile("mirai.zip","r") as zip_ref:
    zip_ref.extractall()


# 文件位置
path = "mirai.pcap" # 要处理的文件，支持 pcap / pcapng / tsv
packet_limit = np.inf # 最多处理多少个包

# KitNET 参数：
maxAE = 10 # 集成层中每个自编码器的最大输入维度（也就是论文里的 m）
FMgrace = 5000 # 学习特征映射（决定集成结构）所用的包数
ADgrace = 50000 # 训练异常检测器（集成本身）所用的包数

# 构建 Kitsune 对象
K = Kitsune(path,packet_limit,maxAE,FMgrace,ADgrace)

print("Running Kitsune:")
RMSEs = []
i = 0
start = time.time()
# 这里逐个处理每个数据包（训练或检测）。
# 每个观测在调用 process() 之后就被丢弃，不会保存下来。
while True:
    i+=1
    if i % 1000 == 0:
        print(i)
    rmse = K.proc_next_packet()
    if rmse == -1:
        break
    RMSEs.append(rmse)
stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))


# 这里演示如何把 RMSE 分数拟合到对数正态分布，用来选取报警阈值 φ
from scipy.stats import norm
benignSample = np.log(RMSEs[FMgrace+ADgrace+1:100000])
logProbs = norm.logsf(np.log(RMSEs), np.mean(benignSample), np.std(benignSample))

# 画出 RMSE 异常分数
print("Plotting results")
from matplotlib import pyplot as plt
from matplotlib import cm
plt.figure(figsize=(10,5))
fig = plt.scatter(range(FMgrace+ADgrace+1,len(RMSEs)),RMSEs[FMgrace+ADgrace+1:],s=0.1,c=logProbs[FMgrace+ADgrace+1:],cmap='RdYlGn')
plt.yscale("log")
plt.title("Anomaly Scores from Kitsune's Execution Phase")
plt.ylabel("RMSE (log scaled)")
plt.xlabel("Time elapsed [min]")
figbar=plt.colorbar()
figbar.ax.set_ylabel('Log Probability\n ', rotation=270)
plt.show()
