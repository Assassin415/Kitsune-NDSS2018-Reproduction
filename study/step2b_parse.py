# -*- coding: utf-8 -*-
"""
第 2 课配套：看看"解析器"到底从数据包里取出了什么

论文架构图最左边的两个方框是：
  1. Packet Capturer（抓包）：把网线上的二进制流记录下来
  2. Packet Parser（解析）：按协议一层层剥开，把有用的字段取出来

这个脚本只做第 2 步，不涉及任何特征提取和神经网络。
它会打印 study/demo6.pcap 里 6 个包被解析出来的全部字段。

运行：python study/step2b_parse.py   （1 秒）
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from scapy.all import rdpcap, Ether, IP, IPv6, TCP, UDP, ARP

PCAP = "study/demo6.pcap"


def parse_one(pkt):
    """模拟解析器的工作：把二进制包拆成一层层字段。"""
    info = {
        "时间": float(pkt.time) - 1000.0,   # 演示数据的时间戳从 1000 开始，这里转成相对时间
        "包长": len(pkt),
        "源MAC": pkt[Ether].src if pkt.haslayer(Ether) else "",
        "目的MAC": pkt[Ether].dst if pkt.haslayer(Ether) else "",
        "源IP": "", "目的IP": "", "协议": "", "源端口": "", "目的端口": "",
    }

    # 第三层：网络层，取 IP 地址
    if pkt.haslayer(IP):
        info["源IP"], info["目的IP"] = pkt[IP].src, pkt[IP].dst
    elif pkt.haslayer(IPv6):
        info["源IP"], info["目的IP"] = pkt[IPv6].src, pkt[IPv6].dst
    elif pkt.haslayer(ARP):
        info["源IP"], info["目的IP"] = pkt[ARP].psrc, pkt[ARP].pdst
        info["协议"] = "ARP"
        return info

    # 第四层：传输层，取端口和协议名
    if pkt.haslayer(TCP):
        info["协议"] = "TCP"
        info["源端口"], info["目的端口"] = pkt[TCP].sport, pkt[TCP].dport
    elif pkt.haslayer(UDP):
        info["协议"] = "UDP"
        info["源端口"], info["目的端口"] = pkt[UDP].sport, pkt[UDP].dport
    else:
        info["协议"] = pkt.lastlayer().name
    return info


def main():
    print("读取文件：" + PCAP)
    packets = rdpcap(PCAP)
    print("抓到 %d 个包。下面是被解析出来的字段（每个包一行）：\n" % len(packets))

    header = ["时间(s)", "包长", "源MAC", "目的MAC", "源IP", "目的IP", "协议", "源端口", "目的端口"]
    print("  ".join("%-9s" % h for h in header[:2]) + "  " +
          "  ".join("%-17s" % h for h in header[2:4]) + "  " +
          "  ".join("%-12s" % h for h in header[4:6]) + "  " +
          "  ".join("%-8s" % h for h in header[6:]))

    for i, pkt in enumerate(packets, 1):
        info = parse_one(pkt)
        print("%-9.1f  %-9d  %-17s  %-17s  %-12s  %-12s  %-8s  %-8s  %-8s"
              % (info["时间"], info["包长"], info["源MAC"], info["目的MAC"],
                 info["源IP"], info["目的IP"], info["协议"],
                 info["源端口"], info["目的端口"]))

    print("""
请注意这三件事：

  1. 解析出来的都是"客观事实"：几点、多大、谁发给谁。
     它还没有任何判断能力，也不是神经网络要的特征。

  2. 这些字段正好就是 AfterImage 需要的原料：
     源MAC + 源IP  → 用于 MI 通道
     源IP + 目的IP → 用于 HH 通道
     源端口 + 目的端口 → 用于 HpHp 通道
     时间 → 用于计算衰减和抖动
     包长 → 用于统计带宽

  3. 解析这一步是可替换的：用 Wireshark 的 tshark 解析（快），
     或者用 Python 的 scapy 解析（慢），结果一样，只是速度差很多。
""" % ())


if __name__ == "__main__":
    main()
