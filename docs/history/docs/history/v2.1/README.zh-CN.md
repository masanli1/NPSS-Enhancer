# NPSS-Enhancer（中文说明）

作者：[masanli1](https://github.com/masanli1) · 仓库：[NPSS-Enhancer](https://github.com/masanli1/NPSS-Enhancer)

[English](README.md) · [历史文档](docs/history/)

**扰动稳定支点降噪：语音/音乐通用，传统算法，CPU 可跑，带 GUI/CLI。**

> **无需训练、无需 GPU，比 SpecSub 高 2.7 dB SI-SDR。**

## 结果（VoiceBank-DEMAND 30 对 held-out，16 kHz）

| 模式 | SI-SDR | PESQ | STOI |
|---|---|---|---|
| noisy | 8.29 | 2.056 | 0.9156 |
| SpecSub | 11.05 | 2.353 | 0.9168 |
| **音乐 16k**（多统计 blend） | **13.77** | 2.487 | 0.9071 |
| **声音 16k**（多统计 blend + 40% NMF） | **13.92** | 2.480 | **0.9110** |
| 音乐 48k（旧 NPSS） | 13.02 | 2.421 | 0.9026 |

- 对比 v2.0：音乐 +0.38 dB SI（PESQ −0.027）；声音 +0.22 dB SI（PESQ 持平）。
- STOI 仍低于 noisy/经典基线；这是质量/可懂度的 trade-off。
- 不是 SOTA；深度模型约 18–20 SI-SDR / PESQ 3+。
- 30 对、一个数据集；数字随数据变化。

## 多统计 Gmin

- 扰动库：white、Student-t(3)、impulsive、AM(4 Hz)、bursty、traffic。
- **不用人声/babble**；换统计，不换颜色。
- 每个 family 独立 3-pass 稳定性 → per-family gate → **min（交集）**。
- 与单扰动 **50/50 混合**；声音再叠 40% NMF。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 可选 PESQ/STOI
pip install -r requirements-metrics.txt
```

## 使用

```bash
# 合成 demo（不含第三方音频）
python scripts/make_demo.py

# CLI（默认多统计）
python npss_parallel.py samples/demo_noisy.wav samples/demo_denoised.wav --workers 4

# 旧单扰动模式
python npss_parallel.py samples/demo_noisy.wav samples/demo_old.wav --single --workers 4

# GUI
./run_gui.command
# 或
python GUI/npss_gui.py
```

- GUI 有“多统计 Gmin（v2.1）”复选框，默认开启。
- 16k 多统计；48k 自动回旧 NPSS。
- 旧循环只对非多统计模式生效；多统计固定 3-pass/2-loop。
- 多统计约 6× 单扰动耗时。

## 评估

```bash
python eval_official.py --clean data/val16/clean --noisy data/val16/noisy
python eval_baselines.py --clean data/val16/clean --noisy data/val16/noisy
```

- 数据不放仓库；`data/val16/` 自己放 16k clean/noisy。
- PESQ/STOI 依赖 scipy/pesq/pystoi；建议普通 venv/conda。

## 双协议

- **学术/非商业：免费**，见 `LICENSE-ACADEMIC.md`。
- **商业使用：收费**，见 `LICENSE-COMMERCIAL.md`，联系 `2459551532@qq.com`。
- NMF 字典来自 VoiceBank-DEMAND（CC BY 4.0），需保留署名；见 `NOTICE.md`。
- 不是法律意见；商用前找律师。

## 限制

- 传统单通道，离深度 SOTA 很远。
- STOI 没超过 noisy；论文/产品里要如实写。
- 多统计约 6× 计算；长音频明显变慢。
- 30 对 held-out；traffic 是合成，不是真实环境库。
- 音乐 PESQ 比 v2.0 略低（2.487 vs 2.514），SI/STOI 更高。

## 历史

v2.0 文档在 [`docs/history/v2.0/`](docs/history/v2.0/)；当前为 v2.1 多统计 Gmin。
