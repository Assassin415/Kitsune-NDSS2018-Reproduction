# -*- coding: utf-8 -*-
"""汇总参数 m（maxAE）敏感性实验：读取各档指标文件，输出汇总表与速度-精度图。

运行前需先跑完四档主实验：
    python study/step3_full_run.py --max-ae 1  --out-prefix ae1
    python study/step3_full_run.py --max-ae 5  --out-prefix ae5
    python study/step3_full_run.py --max-ae 10 --out-prefix ae10
    python study/step3_full_run.py --max-ae 20 --out-prefix ae20

指标文件不记录处理速率，因此速率、耗时与内存由下方 RATE 表显式登记。
"""

from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, ".cache", "matplotlib"))

from matplotlib import pyplot as plt

# 让图里的中文正常显示（macOS 自带字体）
plt.rcParams["font.sans-serif"] = ["Songti SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# maxAE -> (子空间个数 k, 处理速率 包/秒, 总耗时 分钟, 峰值内存 MB)
RATE = {
    1:  (100, 598, 21.3, 2749),
    5:  (31, 885, 14.4, 2817),
    10: (16, 1032, 12.3, 2958),
    20: (12, 1013, 12.6, 2904),
}


def parse_metrics(path):
    if not os.path.exists(path):
        return None
    text = open(path, encoding="utf-8").read()
    out = {"auc": None, "tpr0": None, "thr0": None}
    m = re.search(r"AUC：([0-9.]+)", text)
    if m:
        out["auc"] = float(m.group(1))
    m = re.search(r"FPR<=0\.000 时：TPR = ([0-9.]+)（阈值 RMSE > ([0-9.]+)）", text)
    if m:
        out["tpr0"] = float(m.group(1))
        out["thr0"] = float(m.group(2))
    return out


def main():
    prefix = {1: "ae1", 5: "ae5", 10: "ae10", 20: "ae20"}
    rows = []
    for mae in sorted(prefix):
        metrics = parse_metrics("results/%s_metrics.txt" % prefix[mae])
        if metrics is None or metrics["auc"] is None:
            print("跳过 maxAE=%d：缺少 results/%s_metrics.txt" % (mae, prefix[mae]))
            continue
        k, rate, minutes, mem = RATE[mae]
        rows.append((mae, k, rate, minutes, mem, metrics["auc"], metrics["tpr0"]))

    if not rows:
        print("没有可汇总的结果")
        return

    lines = ["%-6s %-8s %-12s %-9s %-9s %-9s %-10s" %
             ("maxAE", "子空间k", "速率(包/秒)", "耗时(分)", "内存(MB)", "AUC", "TPR@FPR=0")]
    for mae, k, rate, minutes, mem, auc, tpr0 in rows:
        lines.append("%-6d %-8d %-12d %-9.1f %-9d %-9.4f %-10.4f" %
                     (mae, k, rate, minutes, mem, auc, tpr0))
    table = "\n".join(lines)
    print(table)
    with open("results/maxae_summary.txt", "w", encoding="utf-8") as f:
        f.write(table + "\n")

    rates = [r[2] for r in rows]
    aucs = [r[5] for r in rows]
    labels = ["m=%d\n(k=%s)" % (r[0], r[1]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    bars = ax1.bar(range(len(rows)), rates, width=0.5, color="#8FC6E0")
    ax1.set_ylabel("处理速率（包/秒）", color="#2C7A9E")
    ax1.tick_params(axis="y", labelcolor="#2C7A9E")
    ax1.set_xticks(range(len(rows)))
    ax1.set_xticklabels(labels)
    top = max(rates) * 1.28 if max(rates) else 1
    ax1.set_ylim(0, top)
    for b, r in zip(bars, rates):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + top * 0.03,
                 "%d" % r, ha="center", va="bottom", fontsize=9, color="#2C7A9E")

    ax2 = ax1.twinx()
    ax2.plot(range(len(rows)), aucs, "o-", color="#0056FF", lw=2, ms=7)
    ax2.set_ylabel("ROC AUC", color="#0056FF")
    ax2.tick_params(axis="y", labelcolor="#0056FF")
    lo, hi = min(aucs), max(aucs)
    pad = max(0.004, (hi - lo) * 0.6)
    ax2.set_ylim(lo - pad, hi + pad)
    for i, a in enumerate(aucs):
        ax2.text(i, a + pad * 0.3, "%.4f" % a, ha="center", fontsize=9, color="#0056FF")

    ax1.set_xlabel("集成层每个自编码器的最大输入维度 m")
    plt.title("参数 m 的速度-精度权衡（Mirai 全量 764,137 包）")
    fig.tight_layout()
    fig.savefig("results/maxae_sweep.png", dpi=150)
    print("图已保存：results/maxae_sweep.png")


if __name__ == "__main__":
    main()
