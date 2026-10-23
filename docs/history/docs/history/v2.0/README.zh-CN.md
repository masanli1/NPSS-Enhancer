# NPSS-Enhancer（中文说明）

作者：[masanli1](https://github.com/masanli1) · 仓库：[NPSS-Enhancer](https://github.com/masanli1/NPSS-Enhancer)

**扰动稳定支点降噪：语音/音乐通用，传统算法，CPU 可跑，带 GUI/CLI。**

## 结果（VoiceBank-DEMAND 30 对 held-out，16 kHz）

| 模式 | SI-SDR | PESQ | STOI |
|---|---|---|---|
| noisy | 8.29 | 2.056 | 0.9156 |
| SpecSub | 11.05 | 2.353 | 0.9168 |
| **音乐 16k**（互补 k5，floor 0.3） | **13.39** | **2.514** | 0.9055 |
| **声音 16k**（互补 k5，floor 0.3 + NMF） | **13.70** | 2.480 | **0.9104** |
| 音乐 48k（旧 NPSS） | 13.02 | 2.421 | 0.9026 |

- 不是 SOTA；深度模型约 18–20 SI-SDR / PESQ 3+。
- STOI 仍低于 noisy/经典基线；这是质量/可懂度的 trade-off。
- 30 对、一个数据集；数字随数据变化。

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
# 生成合成 demo（不含第三方音频）
python scripts/make_demo.py

# CLI（音乐模式）
python npss_parallel.py samples/demo_noisy.wav samples/demo_denoised.wav --workers 4 --loops 5

# GUI
./run_gui.command
# 或
python GUI/npss_gui.py
```

- 音乐模式：16k 自动互补；48k 走旧 NPSS。
- 声音模式：16k 互补 + 40% NMF；48k 只跑 NPSS。
- 循环默认 5；NMF 字典为 60 条英文 clean，中文/音乐未验证。

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

## 注意

- 传统单通道，离深度 SOTA 很远。
- STOI 没提升；论文/产品里要如实写。
- 5 次循环约 5× NPSS 时间。
- 只有 30 对 held-out；PESQ/STOI 随数据集变化。