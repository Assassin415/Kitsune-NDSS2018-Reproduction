# -*- coding: utf-8 -*-
"""
追查：为什么攻击期还有约 11% 的包分数很低？

做法：读第 250000~450000 个包（攻击已经稳定），把每个包的 Kitsune 分数
和它的五元组（源/目的 IP、协议、端口、包长）对起来，看看低分的包是谁发的。

运行：python study/analyze_low_scores.py   （约 1~2 分钟，需要先跑完 step3）
"""

import collections
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
from scapy.all import PcapReader, IP, IPv6, TCP, UDP, ARP

LO, HI = 250000, 450000        # 要抽查的包区间
CLEAN_MAX = 0.7433             # 干净流量（55001~70000）里出现过的最大分数


def load_scores():
    scores = {}
    with open("results/full_rmse.csv") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            i = int(row[0])
            if LO < i <= HI:
                scores[i] = float(row[1])
    return scores


def describe(pkt):
    """把一个包浓缩成 (源, 协议, 包长)。"""
    src = dst = "?"
    if pkt.haslayer(IP):
        src, dst = pkt[IP].src, pkt[IP].dst
    elif pkt.haslayer(IPv6):
        src, dst = pkt[IPv6].src, pkt[IPv6].dst
    elif pkt.haslayer(ARP):
        return pkt[ARP].psrc, "ARP", len(pkt)
    else:
        return (pkt.src if hasattr(pkt, "src") else "?"), "L2", len(pkt)

    if pkt.haslayer(TCP):
        proto = "TCP:%d->%d" % (pkt[TCP].sport, pkt[TCP].dport)
    elif pkt.haslayer(UDP):
        proto = "UDP:%d->%d" % (pkt[UDP].sport, pkt[UDP].dport)
    else:
        proto = pkt.lastlayer().name
    return src, proto, len(pkt)


def main():
    scores = load_scores()
    print("已载入第 %d~%d 个包的分数：%d 个" % (LO + 1, HI, len(scores)))

    low = []      # 分数低于干净流量最大值的包
    high = []     # 分数明显很高的包
    with PcapReader("mirai.pcap") as reader:
        for i, pkt in enumerate(reader, start=1):
            if i <= LO:
                continue
            if i > HI:
                break
            s = scores.get(i)
            if s is None:
                continue
            item = (describe(pkt), s)
            if s <= CLEAN_MAX:
                low.append(item)
            elif s > 50:
                high.append(item)

    def top(items, key_index, n=5):
        c = collections.Counter(it[0][key_index] for it in items)
        return c.most_common(n)

    print("\n本次抽查：低分包 %d 个（%.1f%%），高分包 %d 个"
          % (len(low), 100.0 * len(low) / len(scores), len(high)))

    print("\n【低分包】最常见的来源主机：")
    for k, v in top(low, 0):
        print("   %-22s %6d 个 (%.1f%%)" % (k, v, 100.0 * v / len(low)))
    print("【低分包】最常见的协议/端口：")
    for k, v in top(low, 1):
        print("   %-22s %6d 个 (%.1f%%)" % (k, v, 100.0 * v / len(low)))
    print("   低分包平均长度：%.0f 字节" % np.mean([it[0][2] for it in low]))

    print("\n【高分包】最常见的来源主机：")
    for k, v in top(high, 0):
        print("   %-22s %6d 个 (%.1f%%)" % (k, v, 100.0 * v / len(high)))
    print("【高分包】最常见的协议/端口：")
    for k, v in top(high, 1):
        print("   %-22s %6d 个 (%.1f%%)" % (k, v, 100.0 * v / len(high)))
    if high:
        print("   高分包平均长度：%.0f 字节" % np.mean([it[0][2] for it in high]))

    print("\n结论：如果低分包集中在某个固定的源主机和固定端口上，")
    print("      那说明它不是攻击包，而是同时段在跑的正常业务流量（被时间标注误判成攻击）。")


if __name__ == "__main__":
    main()
