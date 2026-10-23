# Kitsune NDSS 2018 论文复现

本项目是 NDSS 2018 论文 Kitsune 的课程复现。Kitsune 面向在线、无监督的网络入侵检测，用一组小型自编码器学习正常流量，并以重构误差判断异常。

实现基于作者公开的
[Kitsune-py](https://github.com/ymirsky/Kitsune-py)，遵循 MIT License。
原始论文信息：

> Yisroel Mirsky, Tomer Doitshman, Yuval Elovici, and Asaf Shabtai.
> *Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection.*
> NDSS 2018.

![Kitsune 系统架构](Kitsune_fig.png)

## 复现内容

- 使用 AfterImage 从数据包中提取 100 维统计特征。
- 使用增量特征映射器，把强相关特征划分为若干子空间。
- 在线训练 KitNET 自编码器集成，不使用人工标签。
- 在公开的 764,137 个 Mirai 数据包上完成主实验。
- 对 `maxAE = 1、5、10、20` 进行参数扫描，观察速度和精度的权衡。

## 复现结果

主实验参数为 `maxAE=10`、`FM_grace=5000`、`AD_grace=50000`。

| 指标 | 结果 |
|---|---:|
| ROC AUC | 0.9042 |
| TPR @ FPR = 0 | 0.8159 |
| TPR @ FPR = 0.001 | 0.8160 |
| 干净流量 RMSE 中位数 | 0.0096 |
| 攻击流量 RMSE 中位数 | 17.9590 |
| 处理速率 | 1,032 包/秒 |

参数扫描结果：

| maxAE | 特征子空间数 | 处理速率（包/秒） | AUC |
|---:|---:|---:|---:|
| 1 | 100 | 598 | 0.9119 |
| 5 | 31 | 885 | 0.9022 |
| 10 | 16 | 1,032 | 0.9042 |
| 20 | 12 | 1,013 | 0.9089 |

完整指标见 `results/*_metrics.txt` 和 `results/maxae_summary.txt`，实验图见
`results/*.png`。

## 目录说明

| 路径 | 说明 |
|---|---|
| `Kitsune.py` | 在线训练与检测主流程 |
| `AfterImage.py` | 阻尼增量统计实现 |
| `netStat.py` | 网络通道划分 |
| `FeatureExtractor.py` | 特征提取封装 |
| `KitNET/` | 特征映射与自编码器集成 |
| `quickrun.py` | 小样本快速运行脚本 |
| `study/` | 完整实验与分析脚本 |
| `results/` | 已完成实验的指标和图表 |
| `original_code_backup/` | 兼容性修改前的原始代码 |

## 运行环境

已在 macOS 和 Python 3.13.9 上验证。安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 数据集

仓库不附带 Mirai 抓包文件，避免仓库体积过大。请从作者公开仓库下载数据集，
解压后把 `mirai.pcap` 放在项目根目录：

<https://github.com/ymirsky/Kitsune-py>

## 运行复现

运行主实验：

```bash
python study/step3_full_run.py --pcap mirai.pcap --max-minutes 25
```

主实验会生成：

- `results/full_rmse.csv`
- `results/full_rmse.png`
- `results/full_metrics.txt`

运行参数扫描并汇总结果：

```bash
python study/step3_full_run.py --max-ae 1  --out-prefix ae1
python study/step3_full_run.py --max-ae 5  --out-prefix ae5
python study/step3_full_run.py --max-ae 10 --out-prefix ae10
python study/step3_full_run.py --max-ae 20 --out-prefix ae20
python study/step5_maxae_summary.py
```

运行小样本演示：

```bash
python study/step1_features.py
python quickrun.py --pcap study/demo6.pcap --limit 6 --fm-grace 2 --ad-grace 2
```

## 实现差异

公开 Python 实现与论文描述有三处差异，会影响复现数值与论文结果的直接比较：

1. 实际提取 100 维特征，论文为 115 维，原因是单独统计源 IP 的通道被注释掉。
2. 代码中的第三个一维统计量命名为 `std`，实际返回的是方差。
3. HH 和 HpHp 通道的聚合范围与论文描述不完全一致。

公开抓包没有逐包真实标签，因此本次评估按时间区间划分干净流量和攻击流量。

为兼容当前 NumPy 版本，代码只做了 `np.Inf`/`numpy.Inf` 替换和 tshark 路径适配。
修改前文件保存在 `original_code_backup/`。

## 许可证

上游代码和本次复现均遵循 MIT License。详见 [LICENSE](LICENSE) 和
[KitNET/LICENSE.txt](KitNET/LICENSE.txt)。
