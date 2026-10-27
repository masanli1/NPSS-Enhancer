# NPSS-Enhancer v2.7.0 完整技术文档

> 版本：v2.7.0（算法 v2.7）  
> 日期：2026-10-25  
> 平台：macOS 11+，Apple Silicon（M1/M2/M3/M4）  
> 作者：[masanli1](https://github.com/masanli1)  
> 仓库：https://github.com/masanli1/NPSS-Enhancer  
> 联系：2459551532@qq.com

---

## 目录

1. 摘要
2. 结果摘要
3. 算法
4. 与历史版本对比
5. 客观评测
6. 图形界面（App）
7. 命令行
8. 构建与打包
9. 限制与边界
10. 许可
11. 版本历史
12. 文件与哈希

---

## 1. 摘要

NPSS-Enhancer 是一个**无需训练、CPU 可跑、可解释**的传统语音/音乐降噪器。

核心思想：

- 对信号施加不同统计特性的扰动；
- 观察局部极值位置（支点）在扰动下是否稳定；
- 用稳定支点控制谱减/维纳类增益；
- 多统计扰动取交集，再与单扰动结果混合；
- 用跨区域拓扑、支点补全、高/低贡献角色和能量/SNR 加权进一步修正；
- 16k 语音额外叠加 40% NMF；48k 音乐保留全带与立体声。

它不是深度学习模型，不依赖 GPU，也不使用云端。

---

## 2. 结果摘要

### 2.1 16k 语音/音乐（30 对 held-out，VoiceBank-DEMAND）

| 方法 | SI-SDR | PESQ | STOI |
|---|---|---|---|
| noisy | 8.29 | 2.056 | 0.9156 |
| SpecSub | 11.05 | 2.353 | 0.9168 |
| v2.7 音乐 16k | **13.92** | **2.505** | 0.9082 |
| v2.7 声音 16k | **14.03** | **2.489** | **0.9116** |

- 比 SpecSub：SI **+2.87/+2.98 dB**，PESQ **+0.15/+0.14**。
- STOI 仍低于 noisy/经典基线；这是**质量/可懂度 trade-off**。
- 深度模型参考：SI 18–20，PESQ 2.8–3.5，STOI 0.93–0.97。

### 2.2 48k 音乐

- **全带**：16k 多统计 + 8k 以上柔和（×0.70，约 −2.15 dB）；
- **立体声**：左右声道分别处理；
- **低频保护**：20–250 Hz 约 **−1.01 dB**；
- **电平匹配**：输出 RMS 对齐输入；
- 8k+ 只做柔和，不降噪；
- 48k 没有 clean reference，因此不报 PESQ/STOI。

### 2.3 结论

- **传统单通道：T1 / A−**；
- **工程/产品：A−**（CPU、离线、可解释、GUI/CLI、48k 立体声）；
- **对比深度：差一档**；
- **三区论文：证据仍不足**（30 对、无深度基线、无全量测试/统计）。

---

## 3. 算法

### 3.1 单扰动稳定性

对信号 `x` 施加相关噪声：

```text
x_k = x + a_k * sigma * n_k
```

在幅度 STFT 上计算 2×2 局部极值位置 `P`：

```text
P(M)(f,t) =
  argmax_{i,j in {0,1}} M(f+i,t+j)   if mean2x2 > mean4x4
  argmin_{i,j in {0,1}} M(f+i,t+j)   otherwise
```

稳定性：

```text
S(f,t) = (1/K) * sum_k 1[ P(x_k)(f,t) == P(x)(f,t) ]
```

`S` 越高，越可能是结构而不是噪声/细节。

### 3.2 多统计 Gmin

扰动库（无 voice/babble）：

```text
white, student_t(df=3), impulsive, AM(4 Hz),
bursty(20–80 ms on), traffic(low-pass + 60 Hz + rumble)
```

每个 family 独立做 3-pass 稳定性，得到 `S_f`，再算 per-family gate：

```text
g_f = contrast_gate(S_f, 30, 90, 0.30)
g_min = min_f g_f
y_gmin = ISTFT(g_specsub * g_min * X)
```

`min` 表示“所有统计下都稳定才保留”，去掉 family-specific 假支点。

### 3.3 单扰动 + Gmin 混合

```text
y_music = 0.5 * y_single + 0.5 * y_gmin
y_voice = 0.6 * y_music + 0.4 * y_nmf
```

- 50/50 保留单扰动细节，避免 Gmin 过度保守；
- 声音模式叠加 40% NMF。

### 3.4 跨区域拓扑

- 时间轴切成 **0.5 s 区域**（16 帧，hop 512）；
- 每区域取 **8 个频带的平均稳定性**；
- 相邻区域余弦相似度 → 图 degree；
- soft 权重：

```text
w = 0.8 + 0.2 * degree / max_degree
S = S * w
```

- 使用前一/后一区域的分布，但权重很软，避免过度平滑；
- 硬拓扑权重和时间中值滤波被验证为更差，未采用。

### 3.5 高/低贡献角色补全

- **高贡献**：`S >= 0.95` 且前后至少一帧也是支点 → **主干锚点**；
- **低贡献**：`0.65 <= S < 0.95` → **辅助**；
- 补全：
  - 主干锚点补到 `THETA = 0.95`；
  - 低贡献只在两侧都有主干锚点时补到 **0.82**；
- **gate 仍连续**，不做硬二值；
- 只补“应该有但没发现”的位置，不做音频恢复。

### 3.6 能量/SNR 加权（v2.7）

对每个时频点：

```text
snr = 10 * log10(|X|^2 / noise)
e   = sigmoid((snr - 6) / 4)
gate = gate * (0.8 + 0.2 * e)
```

- 高 SNR 保留更多，低 SNR 压一点；
- **只用于 16k**；
- 48k 全带路径关闭能量加权，避免高频被相对压低。

### 3.7 48k 全带 + 立体声

```text
x16   = FFT_resample(x, 48000 -> 16000)
y16   = multi_stat(x16)            # 0-8k
ylow  = FFT_resample(y16, 16000 -> 48000)
xh    = highpass(x, 8000 Hz)       # 8k+ 柔和 ×0.70
y     = (ylow + xh) * (rms_in / rms_out)
```

- 8k 以上不再被 16k 带限砍掉；
- 左右声道分别处理；左右完全相同时只算一次再复制；
- 真立体声约 **2×** 计算。

### 3.8 最终参数

```text
single : family=correlated, K=10, passes=3, loops=2,
         theta=0.95, support=0.60, gate_floor=0.30,
         n_fft=2048, hop=512, backend=specsub
multi  : 6 families, K=10, 3-pass, 2-loop, gate_floor=0.30
topology: 0.5 s regions, 8 bands, w=0.8+0.2*degree
roles  : high anchors fill 0.95; low auxiliary fill 0.82
energy : 16k only, gate *= 0.8+0.2*sigmoid((snr-6)/4)
blend  : 0.5 single + 0.5 gmin
voice  : 0.6 blend + 0.4 NMF
48k    : 16k gate + 8k+ preserved; low floor 0.70; stereo L/R
```

---

## 4. 与历史版本对比

| 版本 | 音乐 16k | 声音 16k | 关键变化 |
|---|---|---|---|
| v2.0 | 13.39 / 2.514 / 0.9055 | 13.70 / 2.480 / 0.9104 | 单扰动 + 互补 |
| v2.1 | 13.77 / 2.487 / 0.9071 | 13.92 / 2.480 / 0.9110 | 多统计 Gmin |
| v2.2 | 13.77 | 13.92 | 48k 全带 |
| v2.3 | 13.77 | 13.92 | 立体声 + 低频保护 |
| v2.4 | 13.79 / 2.492 / 0.9085 | 13.94 / 2.482 / 0.9116 | 跨区拓扑 |
| v2.5 | 13.81 / 2.497 / 0.9086 | 13.95 / 2.484 / 0.9117 | 支点补全 |
| v2.6 | 13.83 / 2.498 / 0.9086 | 13.96 / 2.484 / 0.9117 | 高/低贡献角色 |
| **v2.7** | **13.92 / 2.505 / 0.9082** | **14.03 / 2.489 / 0.9116** | 能量/SNR 加权 |

---

## 5. 客观评测

### 5.1 指标

- **SI-SDR**：尺度不变信噪比；
- **PESQ-WB**：16k 感知质量；
- **STOI**：可懂度。

### 5.2 经典基线

| 方法 | PESQ | STOI |
|---|---|---|
| noisy | 2.056 | 0.9156 |
| Wiener | 2.178 | 0.9168 |
| SpecSub | 2.353 | 0.9168 |
| MMSE-LSA | 2.176 | 0.9168 |
| OM-LSA | 2.202 | 0.9166 |

### 5.3 解释

- SI/PESQ：传统第一梯队；
- STOI：仍低于 noisy/经典基线；
- 深度：SI 18–20，PESQ 2.8–3.5，STOI 0.93–0.97；
- **不是 SOTA**。

### 5.4 评测限制

- 30 对、一个数据集、英语；
- 没有全量 824 对；
- 没有深度基线；
- 48k 没有 clean reference；
- PESQ/STOI 依赖实现，跨仓库比较需谨慎。

---

## 6. 图形界面（App）

### 6.1 安装

1. 打开 `NPSS-Enhancer-v2.7.0-macOS-arm64.dmg`；
2. 把 `NPSS-Enhancer.app` 拖到 Applications；
3. 右键 → 打开（未公证，首次需确认）。

系统要求：macOS 11+，Apple Silicon。

### 6.2 使用

1. 输入：WAV / MP3 / FLAC / M4A / MP4 / MOV；
2. 输出：WAV 或 MP4；
3. 内容：
   - 音乐为主：采样率 **48000**；
   - 声音为主：采样率 **16000**；
4. “多统计 Gmin（v2.7）”默认开启；
5. 开始处理；
6. 完成后播放/查看波形声谱。

### 6.3 说明

- 音乐 48k：全带 + 立体声 + 低频保护；
- 声音 16k：多统计 + 拓扑 + 角色补全 + 能量 + 40% NMF；
- 8k+ 保留但不降噪；
- 16k 处理音乐会带限到 8k；
- 本机处理，不上传云端。

---

## 7. 命令行

```bash
cd /Users/ma/Downloads/降噪/最终算法_多统计
PY="/Users/ma/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/bin/python3"
"$PY" npss_parallel.py input.wav output.wav --workers 8
```

- 默认多统计；
- `--single` 回旧单扰动模式；
- 16k/48k 自动；
- CLI 只读 WAV；MP4 用 GUI。

---

## 8. 构建与打包

- PyInstaller 6.22.3；
- Python 3.12（conda）；
- 打包内容：Python、Tcl/Tk、NumPy、Pillow、imageio-ffmpeg、`nmf_dict.npz`、全部算法；
- App 约 96 MB；
- DMG 约 48 MB，ZIP 约 42 MB；
- ad-hoc 签名，未公证；
- `multiprocessing.freeze_support()` + 串行兜底，修复进程池子进程退出。

---

## 9. 限制与边界

- 传统单通道，不是深度 SOTA；
- STOI 未超过 noisy/经典；
- 30 对、一个数据集；
- 48k 8k+ 保留但不降噪；
- 48k 低频仍约 −1 dB；
- 真立体声约 2×，多统计约 6×；
- 音乐 PESQ 仍低于 v2.0（2.505 vs 2.514）；
- NMF 仅英语、16k；
- 不支持 Intel Mac；Windows/Linux 需单独打包；
- App 未公证，Gatekeeper 需右键打开。

---

## 10. 许可

双协议：

- **学术/非商业**：免费，见 `LICENSE-ACADEMIC.md`；
- **商业**：付费授权，见 `LICENSE-COMMERCIAL.md`；
- 联系：2459551532@qq.com。

NMF 字典来自 VoiceBank-DEMAND（CC BY 4.0），需保留署名；见 `NOTICE.md`。

不是法律意见；商用前请咨询律师。

---

## 11. 版本历史

- **v2.0**：单扰动 + 互补；音乐 13.39 / 声音 13.70。
- **v2.1**：多统计 Gmin；音乐 13.77 / 声音 13.92。
- **v2.2**：48k 全带。
- **v2.3**：立体声 + 48k 低频保护。
- **v2.4**：soft 跨区拓扑。
- **v2.5**：支点补全。
- **v2.6**：高/低贡献角色补全。
- **v2.7**：能量/SNR 加权；音乐 13.92 / 声音 14.03。
- **App v2.7.0**：macOS 打包；修复进程池子进程退出。

---

## 12. 文件与哈希

### 安装包

```text
发行版/NPSS-Enhancer-v2.7.0-macOS-arm64.dmg
SHA256: b580b4a35f08e90a033ac28ec922e6a448789a4359243c9d4cdbf2d42f46899f

发行版/NPSS-Enhancer-v2.7.0-macOS-arm64.zip
SHA256: f65a9fac381d1588bc588883d385a619904c75bbcee0c316eef70686c38523f9
```

### 程序封存

```text
归档/封存/最终算法_多统计_v2.7_2026-10-25.zip
SHA256: 83baaa0ac005c6e76a412ff37e06bbb3f49571008966cb9cbc2f9ad78c725438
```

### 关键目录

```text
最终算法_多统计/
├── GUI/
│   ├── npss_gui.py
│   ├── engine.py
│   ├── multi_stat.py
│   ├── common.py
│   ├── common_win.py
│   ├── nmf_blend.py
│   └── nmf_dict.npz
├── npss_parallel.py
├── metrics.py
├── eval_official.py
├── eval_baselines.py
├── results_pesq_stoi.csv
└── README.md
```

---

## 13. 联系与引用

- 作者：masanli1
- 仓库：https://github.com/masanli1/NPSS-Enhancer
- 邮箱：2459551532@qq.com
- 引用：见 `CITATION.cff`

---

*文档结束。*
