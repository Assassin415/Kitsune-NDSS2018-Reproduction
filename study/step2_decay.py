# -*- coding: utf-8 -*-
"""
第 2 课补充：阻尼增量统计（decay）的代码演示

这个脚本直接用 AfterImage.py 里的 incStat 类（和 Kitsune 跑实验时是同一个类），
把「每来一个包，内部变量怎么变」一步步打印出来。

运行：python study/step2_decay.py   （1 秒）
"""

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from AfterImage import incStat


def line(s):
    """把一个 incStat 对象的内部状态格式化成一行。"""
    return ("w=%9.5f  CF1=%11.3f  CF2=%13.3f  平均=%9.3f  方差=%13.3f"
            % (s.w, s.CF1, s.CF2, s.mean(), s.var()))


def scenario1_steady_state():
    print("=" * 100)
    print("场景 1：正常流量。每 0.2 秒来 1 个 100 字节的包，λ=5（半衰期 0.2 秒）")
    print("=" * 100)
    s = incStat(5, "demo")
    t = 0.0
    for i in range(1, 7):
        t += 0.1
        factor = math.pow(2, -5 * 0.1)          # 每一步的衰减系数，手算出来对照
        s.insert(100, t)
        print("第 %d 个包 t=%.1f  (插入前先乘衰减系数 %.4f)  %s" % (i, t, factor, line(s)))
    print("-> 结论：w 不会一直变大，它收敛到 2.0 左右（= 1/(1-0.5)）。")
    print("   这就是「阻尼」的意义：老包的贡献被不断打折，最终只保留最近的活跃程度。\n")


def scenario2_burst():
    print("=" * 100)
    print("场景 2：同一条流上突然来了一个大包（模拟突发）")
    print("=" * 100)
    s = incStat(5, "demo")
    t = 0.0
    for _ in range(5):                          # 先跑几个正常包，让统计量进入稳态
        t += 0.2
        s.insert(100, t)
    print("突发前：       %s" % line(s))
    t += 0.2
    s.insert(5000, t)                           # 一个 5000 字节的大包
    print("5000 字节的包：%s" % line(s))
    print("-> 平均包大小从 100 跳到 2589，这个跳变就是 KitNET 要检测的信号。\n")


def scenario3_idle():
    print("=" * 100)
    print("场景 3：空闲 10 秒。注意：代码里没有定时器，衰减只在「新包到来时」补算一次")
    print("=" * 100)
    s = incStat(5, "demo")
    t = 0.0
    for _ in range(5):
        t += 0.2
        s.insert(100, t)
    print("空闲前：          %s" % line(s))
    t += 10.0
    factor = math.pow(2, -5 * 10.0)
    s.insert(100, t)
    print("空闲 10 秒后来包：%s" % line(s))
    print("   （这一次性乘上的衰减系数 = %.20f）" % factor)
    print("-> 历史几乎被清零，等于「这条流已经凉了」。懒执行的好处：")
    print("   不跑定时器、不遍历数据，只在需要时算一次。\n")


def scenario4_compare_lambda():
    print("=" * 100)
    print("场景 4：同一次突发，在 5 个不同 λ 下分别看到什么（关键对比）")
    print("=" * 100)
    print("每个 λ 都先跑 5 个正常包（100 字节），再插入一个 5000 字节的大包：\n")
    print("%-8s %-12s %-32s %-32s" % ("λ", "半衰期", "突发前", "突发后"))
    for lam in [5, 3, 1, 0.1, 0.01]:
        s = incStat(lam, "demo")
        t = 0.0
        for _ in range(5):
            t += 0.2
            s.insert(100, t)
        before = "w=%.3f  平均=%.1f" % (s.w, s.mean())
        t += 0.2
        s.insert(5000, t)
        after = "w=%.3f  平均=%.1f" % (s.w, s.mean())
        print("%-8s %-12s %-32s %-32s" % (lam, "%.2f 秒" % (1.0 / lam), before, after))
    print("\n-> 结论：λ 越大（半衰期越短），平均值被突发拉得越狠；")
    print("   λ=0.01 的平均值几乎不动，因为那个 5000 字节的包被摊进了很长的历史。\n")


def scenario5_variance_bug():
    print("=" * 100)
    print("场景 5：第三个数是「方差」还是「标准差」？（用第 2 课的两个包核对）")
    print("=" * 100)
    s = incStat(5, "demo")
    s.insert(154, 0.1)
    print("包 1（154 字节）：%s" % line(s))
    s.insert(254, 0.2)
    print("包 2（254 字节）：%s" % line(s))
    print("-> 手动核对公式：CF2/w - 平均^2 = %.3f/%.5f - %.3f^2 = %.3f"
          % (s.CF2, s.w, s.mean(), s.var()))
    print("   如果它是标准差，两个数 154 和 254 的波动应该在 50 上下；实际是 2426，所以是方差。")
    print("   注意 std() 这个函数是自己开过根的，但特征向量用的是 allstats_1D()，")
    print("   它返回 [w, mean, cur_var]，也就是未开根的方差。")


if __name__ == "__main__":
    scenario1_steady_state()
    scenario2_burst()
    scenario3_idle()
    scenario4_compare_lambda()
    scenario5_variance_bug()
