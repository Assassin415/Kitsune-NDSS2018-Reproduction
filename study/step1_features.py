# -*- coding: utf-8 -*-
"""
第 1 步：亲眼看见 AfterImage 提取出来的特征

这一节解决三个问题：
  1. 100 个特征分别是什么（按通道分组看）
  2. 一个包进来，哪些数字变了
  3. 这些数字的含义（包数、平均包大小……）

为什么要自己造一个小 pcap：
  mirai.pcap 有 76 万个包、读一次要 1 分钟，不适合用来理解原理。
  这里用 6 个手工构造的包，数字小到可以心算核对。

运行：python study/step1_features.py   （约 3 秒）
"""

import os
import sys

# 让脚本无论从哪里运行，都能找到项目根目录下的 netStat.py / FeatureExtractor.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import netStat as ns

# AfterImage 的 5 个时间窗（论文里的 λ：5、3、1、0.1、0.01，对应约 100ms~1min）
LAMBDAS = [5, 3, 1, .1, .01]

# 演示用的 3 台主机
MAC_A, IP_A = "aa:aa:aa:aa:aa:aa", "10.0.0.1"
MAC_B, IP_B = "bb:bb:bb:bb:bb:bb", "10.0.0.2"
MAC_C, IP_C = "cc:cc:cc:cc:cc:cc", "10.0.0.3"


def which_channel(header):
    """把特征名归类到它所属的通道。"""
    if header.startswith("MI_dir_"):
        return "MI   (源 MAC+IP 的发送统计)"
    if header.startswith("HH_jit_"):
        return "HH_jit (源IP->目的IP 的到达间隔)"
    if header.startswith("HH_"):
        return "HH   (源IP<->目的IP 的双向统计)"
    if header.startswith("HpHp_"):
        return "HpHp (源socket<->目的socket 的双向统计)"
    return "未知"


def part1_headers():
    print("=" * 78)
    print("第 1 部分：100 个特征分别是什么")
    print("=" * 78)
    stat = ns.netStat(np.nan, 1000, 1000)
    headers = stat.getNetStatHeaders()
    print("特征总数：%d 个（论文写的是 115 个，差的 15 个是被注释掉的 srcIP 通道）\n" % len(headers))

    counts = {}
    examples = {}
    for h in headers:
        ch = which_channel(h)
        counts[ch] = counts.get(ch, 0) + 1
        examples.setdefault(ch, h)
    for ch, n in counts.items():
        print("  %-36s %3d 个   例：%s" % (ch, n, examples[ch]))

    print("\n  每个时间窗(λ)分别贡献多少维：")
    for lam in LAMBDAS:
        per_window = [h for h in headers if ("_%s_" % lam) in h]
        print("    λ=%-5s -> %2d 维" % (lam, len(per_window)))

    print("\n  同一个通道里，每个时间窗的数字长这样（以 MI 为例）：")
    for h in headers[:6]:
        print("    " + h)
    print("    ...（weight=包数, mean=平均包大小, std=包大小标准差）")


def make_demo_pcap(path):
    """造 6 个包的小 pcap：既能看懂计数，又能看懂方向性。"""
    from scapy.all import Ether, IP, TCP, UDP, ARP, wrpcap

    def tcp_pkt(mac_s, ip_s, sport, mac_d, ip_d, dport, payload):
        return (Ether(src=mac_s, dst=mac_d) /
                IP(src=ip_s, dst=ip_d) /
                TCP(sport=sport, dport=dport) /
                (b"x" * payload))

    packets = [
        tcp_pkt(MAC_A, IP_A, 1234, MAC_B, IP_B, 80, 100),   # 1: A->B 正常请求
        tcp_pkt(MAC_A, IP_A, 1234, MAC_B, IP_B, 80, 200),   # 2: A->B 又来一个
        tcp_pkt(MAC_B, IP_B, 80, MAC_A, IP_A, 1234, 50),    # 3: B->A 回包
        tcp_pkt(MAC_A, IP_A, 1234, MAC_B, IP_B, 80, 300),   # 4: A->B 一个很大的包
        (Ether(src=MAC_A, dst=MAC_C) /                       # 5: A->C 换了个目的地
         IP(src=IP_A, dst=IP_C) / UDP(sport=5000, dport=53) / (b"y" * 20)),
        (Ether(src=MAC_A, dst="ff:ff:ff:ff:ff:ff") /         # 6: A 发 ARP 广播
         ARP(psrc=IP_A, pdst=IP_C)),
    ]
    for i, p in enumerate(packets):            # 给每个包一个时间戳，间隔 0.1 秒
        p.time = 1000.0 + i * 0.1
    wrpcap(path, packets)


def part2_trace():
    print()
    print("=" * 78)
    print("第 2 部分：6 个包进来，特征怎么变")
    print("=" * 78)
    pcap = os.path.join(ROOT, "study", "demo6.pcap")
    make_demo_pcap(pcap)
    print("已生成小样本：study/demo6.pcap（6 个包，A=10.0.0.1, B=10.0.0.2, C=10.0.0.3）\n")

    from FeatureExtractor import FE
    fe = FE(pcap, np.inf)
    names = fe.nstat.getNetStatHeaders()

    descriptions = [
        "A->B  TCP 1234->80    100 字节负载",
        "A->B  TCP 1234->80    200 字节负载",
        "B->A  TCP 80->1234     50 字节负载",
        "A->B  TCP 1234->80    300 字节负载",
        "A->C  UDP 5000->53     20 字节负载",
        "A     ARP 广播（问 C 的 MAC）",
    ]

    idx = 0
    while True:
        x = fe.get_next_vector()
        if len(x) == 0:
            break
        x = np.asarray(x, dtype=float)
        # 只挑最短时间窗(λ=5)的非零特征，输出才好读
        picked = [(names[i], x[i]) for i in range(len(x))
                  if "_5_" in names[i] and abs(x[i]) > 1e-12]
        print("包 %d：%s" % (idx + 1, descriptions[idx]))
        for name, value in picked:
            print("      %-30s = %.4g" % (name, value))
        print("      （这一包非零特征 %d 个 / 全向量 %d 维）\n" % (len(picked), len(x)))
        idx += 1


def part3_notes():
    print("=" * 78)
    print("第 3 部分：怎么读这些数字")
    print("=" * 78)
    print("""
  MI_dir_5_weight  -> 这个「源MAC+源IP」一共发过几个包（每来一个包就 +1）
  MI_dir_5_mean    -> 这个源的平均包大小（字节）
  MI_dir_5_std     -> 包大小的波动程度（标准差）

  HH_5_*           -> 看的是「A<->B 这一对」之间的流量，只统计 A->B 方向
  HH_jit_5_*       -> A->B 的包到达时间间隔（抖动）
  HpHp_5_*         -> 看的是「A:1234 <-> B:80 这一对 socket」

  前缀里的 5、3、1、0.1、0.01 是 5 个时间窗：数字越大看得越久远。
  所以同一个统计量有 5 个版本：通道 x 统计量 x 5 个时间窗 = 100 维。

  重点体会：每处理一个包，AfterImage 只做「读数 + 累加」，
  不保存任何包的内容，这是 Kitsune 能在线跑、内存占用小的原因。
""")


if __name__ == "__main__":
    part1_headers()
    part2_trace()
    part3_notes()
