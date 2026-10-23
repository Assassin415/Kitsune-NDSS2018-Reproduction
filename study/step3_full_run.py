# -*- coding: utf-8 -*-
"""
第 3 步：在完整的 Mirai 数据集上跑 Kitsune，看能不能复现论文的效果

论文里的 Mirai 数据集就是仓库里的 mirai.pcap：
  764,137 个包，前 70,000 个是干净流量，之后是 Mirai 僵尸网络的扫描攻击。

参数与论文/官方 example.py 一致：
  maxAE = 10（每个自编码器最多 10 维）
  FMgrace = 5000（先学特征映射）
  ADgrace = 50000（再训练异常检测器），第 55001 个包开始才输出异常分数

脚本会每 1 万个包打印一次进度、实时速度和内存占用，并且边跑边写 CSV，
所以中途被中断也能拿到已完成部分的分数。

运行：
  python study/step3_full_run.py                    # 最多跑 25 分钟
  python study/step3_full_run.py --max-minutes 60   # 放宽到 60 分钟
输出：results/full_rmse.csv、results/full_rmse.png、results/full_metrics.txt
"""

import argparse
import csv
import os
import resource
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, ".cache", "matplotlib"))

import numpy as np

from Kitsune import Kitsune

MAX_AE = 10
FM_GRACE = 5000
AD_GRACE = 50000
FIRST_ATTACK_PACKET = 70000      # 从第 70001 个包开始算攻击流量（论文数据集的标注）


def curr_rss_mb():
    """当前进程占用的物理内存（MB）。macOS 上 ru_maxrss 单位是字节。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def auc_score(scores, labels):
    """手算 ROC AUC（Mann-Whitney U 统计量），避免额外依赖 sklearn。"""
    n = len(scores)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    s_sorted = scores[order]
    i = 0
    while i < n:                      # 处理并列分数，取平均名次
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def tpr_at_fpr(scores, labels, fpr_target):
    """在给定误报率下算检测率 TPR：阈值取干净流量分数的分位数。"""
    clean = scores[labels == 0]
    attack = scores[labels == 1]
    threshold = np.quantile(clean, 1 - fpr_target)
    return (attack > threshold).mean(), threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap", default="mirai.pcap")
    parser.add_argument("--max-minutes", type=float, default=25.0, help="最多跑多少分钟")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个包（0=全量）")
    parser.add_argument("--max-ae", type=int, default=MAX_AE,
                        help="集成层每个自编码器的最大输入维度 m（默认 10）")
    parser.add_argument("--out-prefix", default="full",
                        help="输出文件名前缀，用于参数敏感性实验（默认 full）")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    limit = args.limit if args.limit > 0 else np.inf
    max_ae = args.max_ae
    prefix = args.out_prefix
    csv_path = "results/%s_rmse.csv" % prefix
    metrics_path = "results/%s_metrics.txt" % prefix
    png_path = "results/%s_rmse.png" % prefix

    print("第 1/3 阶段：读取 pcap 并初始化（约 1 分钟）...", flush=True)
    t0 = time.time()
    K = Kitsune(args.pcap, limit, max_ae, FM_GRACE, AD_GRACE)
    print("  初始化完成，耗时 %.1f 秒，内存 %.0f MB" % (time.time() - t0, curr_rss_mb()), flush=True)
    print("  参数：maxAE=%d, FM_grace=%d, AD_grace=%d" % (max_ae, FM_GRACE, AD_GRACE), flush=True)

    print("第 2/3 阶段：逐包处理（每 1 万个包报一次进度）...", flush=True)
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["packet_index", "rmse"])

    scores = []
    t0 = time.time()
    last_report_n = 0
    last_report_t = t0
    stopped_by_budget = False

    while True:
        rmse = K.proc_next_packet()
        if rmse == -1:
            break
        scores.append(rmse)
        writer.writerow([len(scores), rmse])

        if len(scores) % 10000 == 0:
            now = time.time()
            rate = 10000 / (now - last_report_t)
            last_report_n = len(scores)
            last_report_t = now
            print("  %7d 包 | 本段 %6.0f 包/秒 | 累计 %.1f 分钟 | 内存 %.0f MB"
                  % (len(scores), rate, (now - t0) / 60.0, curr_rss_mb()), flush=True)
            csv_file.flush()
            if (now - t0) / 60.0 > args.max_minutes:
                print("  达到时间上限，提前停止（已处理 %d 个包）" % len(scores), flush=True)
                stopped_by_budget = True
                break

    csv_file.close()
    elapsed = time.time() - t0
    scores = np.array(scores)
    n = len(scores)
    print("处理结束：%d 个包，用时 %.1f 分钟（整体 %.0f 包/秒，峰值内存 %.0f MB）"
          % (n, elapsed / 60.0, n / elapsed, curr_rss_mb()), flush=True)

    print("第 3/3 阶段：计算指标和画图...", flush=True)
    start = FM_GRACE + AD_GRACE
    if n <= start:
        print("还没跑过训练期，无法评估，请调大 --max-minutes", flush=True)
        return

    idx = np.arange(start + 1, n + 1)
    test_scores = scores[start:]
    labels = (idx > FIRST_ATTACK_PACKET).astype(int)

    lines = []
    lines.append("数据集：%s%s" % (args.pcap, "（未跑完）" if stopped_by_budget else "（全量跑完）"))
    lines.append("已处理 %d / 764137 个包" % n)
    lines.append("用于评估的区间：第 %d~%d 个包，共 %d 个" % (start + 1, n, len(test_scores)))
    lines.append("其中干净流量 %d 个，攻击流量 %d 个" % ((labels == 0).sum(), (labels == 1).sum()))
    lines.append("")

    if (labels == 0).sum() > 0 and (labels == 1).sum() > 0:
        clean_scores = test_scores[labels == 0]
        attack_scores = test_scores[labels == 1]
        lines.append("干净流量 RMSE：均值 %.4f，中位数 %.4f，99.9%% 分位 %.4f，最大 %.4f"
                     % (clean_scores.mean(), np.median(clean_scores),
                        np.quantile(clean_scores, 0.999), clean_scores.max()))
        lines.append("攻击流量 RMSE：均值 %.4f，中位数 %.4f，最小 %.4f，最大 %.4f"
                     % (attack_scores.mean(), np.median(attack_scores),
                        attack_scores.min(), attack_scores.max()))
        lines.append("")
        lines.append("ROC AUC：%.4f" % auc_score(test_scores, labels))
        for fpr in (0.0, 0.001, 0.01):
            tpr, thr = tpr_at_fpr(test_scores, labels, fpr)
            lines.append("FPR<=%.3f 时：TPR = %.4f（阈值 RMSE > %.4f）" % (fpr, tpr, thr))
    else:
        lines.append("当前区间只有单一类别（攻击还没开始或还没跑到），无法算 AUC。")

    text = "\n".join(lines)
    print("\n" + text + "\n", flush=True)
    with open(metrics_path, "w") as f:
        f.write(text + "\n")

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.scatter(idx, test_scores, s=0.2, c=np.where(labels == 1, "tab:red", "tab:blue"), alpha=0.5)
    if idx.max() > FIRST_ATTACK_PACKET:
        plt.axvline(FIRST_ATTACK_PACKET, color="k", ls="--", lw=1, label="Mirai attack starts")
        plt.legend(loc="upper left")
    plt.yscale("log")
    plt.xlabel("packet index")
    plt.ylabel("RMSE (log scale)")
    plt.title("Kitsune anomaly scores on mirai.pcap (maxAE=%d, up to packet %d)" % (max_ae, n))
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    print("图已保存：%s" % png_path, flush=True)


if __name__ == "__main__":
    main()
