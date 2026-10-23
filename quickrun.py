# -*- coding: utf-8 -*-
"""
Kitsune 快速运行脚本（复现用）

作用：
  1. 用 AfterImage 提取 mirai.pcap 的特征
  2. 交给 KitNET（自编码器集成）在线训练 + 检测
  3. 把每个包的 RMSE 异常分数存成 CSV，并画一张异常分数图

与 example.py 的区别：
  - 可以指定只处理前 N 个包（example.py 的 scapy 分支会读完整文件）
  - 不弹窗，直接把图保存到 results/ 目录，方便写报告

用法：
  python quickrun.py                      # 默认 20000 个包
  python quickrun.py --limit 50000        # 处理 50000 个包
"""

import argparse
import csv
import os
import platform
import time

# matplotlib 需要可写的缓存目录，否则在某些机器上会报警告
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "matplotlib"))

import numpy as np

from FeatureExtractor import FE
from KitNET.KitNET import KitNET


def find_tshark():
    """沿用原仓库逻辑：Windows 找 Wireshark 默认安装路径，其他系统在 PATH 里找 tshark。"""
    if platform.system() == "Windows":
        path = r"C:\Program Files\Wireshark\tshark.exe"
        return path if os.path.isfile(path) else ""
    for path in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(path, "tshark")
        if os.path.isfile(candidate):
            return candidate
    return ""


def main():
    parser = argparse.ArgumentParser(description="Kitsune 快速运行脚本")
    parser.add_argument("--pcap", default="mirai.pcap", help="要处理的 pcap 文件")
    parser.add_argument("--limit", type=int, default=20000, help="最多处理多少个包")
    parser.add_argument("--fm-grace", type=int, default=2000, help="特征映射阶段包数（越少越快）")
    parser.add_argument("--ad-grace", type=int, default=8000, help="异常检测器训练阶段包数")
    parser.add_argument("--max-ae", type=int, default=10, help="单个自编码器的最大输入维度")
    parser.add_argument("--outdir", default="results", help="结果输出目录")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    tshark = find_tshark()
    print("tshark: " + (tshark if tshark else "未安装，将使用 scapy 解析（较慢）"))

    t0 = time.time()
    fe = FE(args.pcap, np.inf)
    n_features = fe.get_num_features()
    print("AfterImage 就绪：%d 个特征，耗时 %.1f 秒" % (n_features, time.time() - t0))

    detector = KitNET(n_features, args.max_ae, args.fm_grace, args.ad_grace, 0.1, 0.75)
    warmup = args.fm_grace + args.ad_grace

    print("开始处理前 %d 个包（前 %d 个包用于训练，之后才输出异常分数）..." % (args.limit, warmup))
    rmse_list = []
    t0 = time.time()
    for i in range(args.limit):
        x = fe.get_next_vector()
        if len(x) == 0:
            print("数据用完了，共读到 %d 个包" % i)
            break
        rmse_list.append(detector.process(x))
        if (i + 1) % 2000 == 0:
            print("  已处理 %d 个包，用时 %.1f 秒" % (i + 1, time.time() - t0))

    elapsed = time.time() - t0
    n = len(rmse_list)
    print("处理完成：%d 个包，用时 %.1f 秒（%.0f 包/秒）" % (n, elapsed, n / max(elapsed, 1e-9)))

    if n == 0:
        return

    # 训练阶段的分数恒为 0，只对训练结束后的包做统计
    scores = np.array(rmse_list[warmup:]) if n > warmup else np.array([])
    if scores.size:
        print("检测阶段 %d 个包：RMSE 均值 %.4f，中位数 %.4f，95 分位 %.4f，最大 %.4f（第 %d 个包）"
              % (scores.size, scores.mean(), np.median(scores), np.percentile(scores, 95),
                 scores.max(), int(np.argmax(scores)) + warmup + 1))

    csv_path = os.path.join(args.outdir, "rmse.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["packet_index", "rmse"])
        for i, v in enumerate(rmse_list):
            writer.writerow([i + 1, v])
    print("分数已保存：" + csv_path)

    if scores.size:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.scatter(range(warmup + 1, n + 1), rmse_list[warmup:], s=0.5)
        plt.yscale("log")
        plt.title("Kitsune anomaly scores (first %d packets of mirai.pcap)" % n)
        plt.ylabel("RMSE (log scale)")
        plt.xlabel("packet index")
        png_path = os.path.join(args.outdir, "rmse.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        print("图表已保存：" + png_path)


if __name__ == "__main__":
    main()
