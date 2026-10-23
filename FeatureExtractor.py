# 检查 Cython 代码是否已经编译过
import os
import subprocess

use_extrapolation=False # 实验性的"相关性外推"代码，默认关闭
if use_extrapolation:
    print("Importing AfterImage Cython Library")
    if not os.path.isfile("AfterImage.c"): # 还没编译过，就先编译一次
        cmd = "python setup.py build_ext --inplace"
        subprocess.call(cmd,shell=True)
# 导入依赖
import netStat as ns
import csv
import numpy as np
print("Importing Scapy Library")
from scapy.all import *
import os.path
import platform
import subprocess


# 从给定的 pcap 文件中逐个数据包地提取 Kitsune 特征，用 get_next_vector() 取用
# 装了 Wireshark（tshark）就优先用它解析（更快），否则退回 scapy（慢很多）
# 用 tshark 解析时会顺手生成一个同名 .tsv 文本文件，下次可以直接拿它当输入
class FE:
    def __init__(self,file_path,limit=np.inf):
        self.path = file_path
        self.limit = limit
        self.parse_type = None # 解析方式未知（待判断）
        self.curPacketIndx = 0
        self.tsvin = None # 用 csv.reader 逐行读 tsv 时使用
        self.scapyin = None # 用 scapy 一次性读入 pcap 时使用

        ### 准备阶段：判断文件类型、选择解析方式 ###
        self.__prep__()

        ### 准备特征提取器（AfterImage）###
        maxHost = 100000000000
        maxSess = 100000000000
        self.nstat = ns.netStat(np.nan, maxHost, maxSess)

    def _get_tshark_path(self):
        if platform.system() == 'Windows':
            return r'C:\Program Files\Wireshark\tshark.exe'
        else:
            system_path = os.environ['PATH']
            for path in system_path.split(os.pathsep):
                filename = os.path.join(path, 'tshark')
                if os.path.isfile(filename):
                    return filename
        return ''

    def __prep__(self):
        ### 找文件 ###
        if not os.path.isfile(self.path):  # 文件不存在
            print("File: " + self.path + " does not exist")
            raise Exception()

        ### 判断文件类型 ###
        type = self.path.split('.')[-1]

        self._tshark = self._get_tshark_path()
        ## 如果文件本身就是 tsv（事先用 wireshark 解析好的）
        if type == "tsv":
            self.parse_type = "tsv"

        ## 如果是 pcap
        elif type == "pcap" or type == 'pcapng':
            # 优先调用 wireshark 的 tshark 来解析（更快）
            if os.path.isfile(self._tshark):
                self.pcap2tsv_with_tshark()  # 生成同名的 tsv 文件
                self.path += ".tsv"
                self.parse_type = "tsv"
            else: # 找不到 tshark，只能用 scapy 解析（慢）
                print("tshark not found. Trying scapy...")
                self.parse_type = "scapy"
        else:
            print("File: " + self.path + " is not a tsv or pcap file")
            raise Exception()

        ### 打开读取器 ###
        if self.parse_type == "tsv":
            maxInt = sys.maxsize
            decrement = True
            while decrement:
                # csv 模块对单行长度有限制，这里不断把上限缩小，
                # 直到不再抛出 OverflowError 为止
                decrement = False
                try:
                    csv.field_size_limit(maxInt)
                except OverflowError:
                    maxInt = int(maxInt / 10)
                    decrement = True

            print("counting lines in file...")
            num_lines = sum(1 for line in open(self.path))
            print("There are " + str(num_lines) + " Packets.")
            self.limit = min(self.limit, num_lines-1)
            self.tsvinf = open(self.path, 'rt', encoding="utf8")
            self.tsvin = csv.reader(self.tsvinf, delimiter='\t')
            row = self.tsvin.__next__() # 跳过表头那一行

        else: # 用 scapy 读取
            print("Reading PCAP file via Scapy...")
            self.scapyin = rdpcap(self.path)
            self.limit = len(self.scapyin)
            print("Loaded " + str(len(self.scapyin)) + " Packets.")

    def get_next_vector(self):
        if self.curPacketIndx == self.limit:
            if self.parse_type == 'tsv':
                self.tsvinf.close()
            return []

        ### 解析下一个数据包 ###
        if self.parse_type == "tsv":
            row = self.tsvin.__next__()
            IPtype = np.nan
            timestamp = row[0]
            framelen = row[1]
            srcIP = ''
            dstIP = ''
            if row[4] != '':  # 这条记录里IPv4 地址非空
                srcIP = row[4]
                dstIP = row[5]
                IPtype = 0
            elif row[17] != '':  # IPv6 地址非空
                srcIP = row[17]
                dstIP = row[18]
                IPtype = 1
            srcproto = row[6] + row[
                8]  # UDP 或 TCP 的源端口：端口字符串拼接后天然区分了协议
            dstproto = row[7] + row[9]  # UDP 或 TCP 的目的端口
            srcMAC = row[2]
            dstMAC = row[3]
            if srcproto == '':  # 没有传输层端口，说明是 L1/L2 层的协议
                if row[12] != '':  # 是 ARP 包
                    srcproto = 'arp'
                    dstproto = 'arp'
                    srcIP = row[14]  # ARP 报文里的源 IP
                    dstIP = row[16]  # ARP 报文里的目的 IP
                    IPtype = 0
                elif row[10] != '':  # 是 ICMP 包
                    srcproto = 'icmp'
                    dstproto = 'icmp'
                    IPtype = 0
                elif srcIP + srcproto + dstIP + dstproto == '':  # 其他协议
                    srcIP = row[2]  # 退而求其次：用源 MAC 当作源地址
                    dstIP = row[3]  # 用目的 MAC 当作目的地址

        elif self.parse_type == "scapy":
            packet = self.scapyin[self.curPacketIndx]
            IPtype = np.nan
            timestamp = packet.time
            framelen = len(packet)
            if packet.haslayer(IP):  # IPv4 包
                srcIP = packet[IP].src
                dstIP = packet[IP].dst
                IPtype = 0
            elif packet.haslayer(IPv6):  # IPv6 包
                srcIP = packet[IPv6].src
                dstIP = packet[IPv6].dst
                IPtype = 1
            else:
                srcIP = ''
                dstIP = ''

            if packet.haslayer(TCP):
                srcproto = str(packet[TCP].sport)
                dstproto = str(packet[TCP].dport)
            elif packet.haslayer(UDP):
                srcproto = str(packet[UDP].sport)
                dstproto = str(packet[UDP].dport)
            else:
                srcproto = ''
                dstproto = ''

            srcMAC = packet.src
            dstMAC = packet.dst
            if srcproto == '':  # 没有传输层端口，说明是 L1/L2 层的协议
                if packet.haslayer(ARP):  # 是 ARP 包
                    srcproto = 'arp'
                    dstproto = 'arp'
                    srcIP = packet[ARP].psrc  # ARP 报文里的源 IP
                    dstIP = packet[ARP].pdst  # ARP 报文里的目的 IP
                    IPtype = 0
                elif packet.haslayer(ICMP):  # 是 ICMP 包
                    srcproto = 'icmp'
                    dstproto = 'icmp'
                    IPtype = 0
                elif srcIP + srcproto + dstIP + dstproto == '':  # 其他协议
                    srcIP = packet.src  # 用源 MAC
                    dstIP = packet.dst  # 用目的 MAC
        else:
            return []

        self.curPacketIndx = self.curPacketIndx + 1


        ### 提取特征：把上面解析出的字段交给 AfterImage 计算统计量
        try:
            return self.nstat.updateGetStats(IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto,
                                                 int(framelen),
                                                 float(timestamp))
        except Exception as e:
            print(e)
            return []


    def pcap2tsv_with_tshark(self):
        print('Parsing with tshark...')
        fields = "-e frame.time_epoch -e frame.len -e eth.src -e eth.dst -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -e icmp.type -e icmp.code -e arp.opcode -e arp.src.hw_mac -e arp.src.proto_ipv4 -e arp.dst.hw_mac -e arp.dst.proto_ipv4 -e ipv6.src -e ipv6.dst"
        cmd =  '"' + self._tshark + '" -r '+ self.path +' -T fields '+ fields +' -E header=y -E occurrence=f > '+self.path+".tsv"
        subprocess.call(cmd,shell=True)
        print("tshark parsing complete. File saved as: "+self.path +".tsv")

    def get_num_features(self):
        return len(self.nstat.getNetStatHeaders())
