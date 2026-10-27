# v2.7 协议复测（同一指标、同一流程）

- 语料：8 对 16 kHz 语音对（macOS `say` 真人声库合成 + 白/粉/交通噪声，SNR 0–15 dB），**不是** VoiceBank-DEMAND，绝对值不可与 2.505 直接比。
- 指标：SI-SDR / STOI，与 v2.7 的 `metrics.py` 同一份代码；PESQ 列留空（本机 `pesq` C 扩展无法加载，且没有 VoiceBank 语料），到有语料的机器上运行同一脚本会自动填上。
- 流程：`scripts/eval_protocol_v27.py`，方法名、输入布局、CSV 列名与 `eval_official.py` 一致。

| 方法 | SI-SDR | PESQ | STOI |
|---|---|---|---|
| noisy | 7.59 | — | 0.9040 |
| SpecSub | 11.02 | — | 0.9155 |
| NPSS (single) | 11.51 | — | 0.9058 |
| **Music 16k (v2.7 默认)** | 12.44 | — | 0.9160 |
| **Voice 16k** | 12.92 | — | 0.9147 |
| **target 15 dB** | 12.55 | — | 0.9153 |
| target 18 dB | 12.49 | — | 0.9117 |

## 复现的方法排序

v2.7 表里的顺序是 noisy < SpecSub < Music 16k，本机复测一致：7.59 < 11.02 < 12.44。说明这套替代语料能复现**方法间相对关系**，这也是唯一可靠的用法。

