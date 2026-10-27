# 定果寻因：15 dB 降噪目标下的四视角不变特征

由 `scripts/report_target15.py` 自动生成，数据来自 `scripts/eval_target15.py`。

## 0. 一句话

本文件的数字来自**合成素材**上的控制研究（噪声底、剂量、四视角消融），不能当成 v2.7 那套 SI-SDR/PESQ/STOI 的替代；协议一致的复测见 `results_v28_protocol.md`。

把“降 15 dB”从一个**涌现结果**变成一个**可指定、可复现的输入**：8/8 个样本命中 15.00 ± 0.01 dB；原固定阈值管线在同样的样本上落在 14.42 ± 2.79 dB。在同一个 15 dB 上，四视角保护图再换来 SI-SDR +0.32 dB（8/8 样本同向）、ΔSNR +0.29 dB。

## 1. 总览

> `噪声底降幅` 是**被控变量**（控制环里反解的那个量），不是效果指标。效果仍只用 SI-SDR / STOI 衡量，与 v2.7 同一套指标；见 `results_v28_protocol.md`。

| 方法 | 控制量：噪声底降幅 (dB) | ΔSNR (dB) | SI-SDR (dB) | 四视角复合 |
|---|---|---|---|---|
| baseline（固定阈值） | 14.42 ± 2.79 | +5.88 | 10.38 | 0.635 |
| target 15 dB + 四视角 | 15.00 ± 0.01 | +5.63 | 10.23 | 0.631 |
| target 15 dB，均匀施加 | 15.00 ± 0.01 | +5.35 | 9.91 | 0.621 |

样本 8 对：3 种信号形态 × 3 种噪声 × 3 个输入 SNR，每段 8 s，均含真实静音段。

`baseline` 的方差说明问题：同一套固定阈值，在不同内容上落点从 11.9 到 19.2 dB。目标模式把它钉死在 15.00。

## 2. 同一 15 dB 下，四视角 vs 均匀

| 样本 | SI-SDR 差 (dB) | ΔSNR 差 (dB) | 四视角 comp | 均匀 comp |
|---|---|---|---|---|
| harmonic_white_+05 | +0.62 | +0.59 | 0.560 | 0.534 |
| harmonic_white_+00 | +0.42 | +0.38 | 0.475 | 0.455 |
| harmonic_pink_+05 | +0.05 | +0.13 | 0.676 | 0.669 |
| harmonic_traffic_+10 | +0.09 | -0.06 | 0.832 | 0.833 |
| music_white_+05 | +0.37 | +0.36 | 0.636 | 0.617 |
| music_pink_+00 | +0.13 | +0.10 | 0.541 | 0.528 |
| formant_white_+05 | +0.71 | +0.74 | 0.549 | 0.555 |
| formant_traffic_+05 | +0.14 | +0.05 | 0.780 | 0.780 |

SI-SDR 是主判据（标准、与感知相关、不依赖我自造的权重）；composite 是四个视角的几何平均，作为诊断：**它并不总是偏向四视角**——选择性衰减会在低能量格打出更深的洞，因而压低包络/频带形状分，但同时抬高 SI-SDR。两者冲突时以 SI-SDR 与 ΔSNR 为准。

## 3. 逐样本全部指标

| case | method | dose | nr_blind | nr_ref | dsnr | si_sdr | energy | freq | form | struct | composite |
|---|---|---|---|---|---|---|---|---|---|---|---|
| harmonic_white_+05 | baseline | 0.0 | 12.728 | 12.663 | 3.937 | 9.755 | 0.481 | 0.65 | 0.655 | 0.459 | 0.554 |
| harmonic_white_+05 | target | 0.66 | 15.001 | 14.921 | 3.821 | 9.908 | 0.462 | 0.703 | 0.637 | 0.475 | 0.56 |
| harmonic_white_+05 | uniform | 0.22 | 15.009 | 14.925 | 3.227 | 9.292 | 0.416 | 0.685 | 0.611 | 0.468 | 0.534 |
| harmonic_white_+00 | baseline | 0.0 | 12.379 | 12.32 | 6.398 | 5.771 | 0.38 | 0.604 | 0.626 | 0.388 | 0.486 |
| harmonic_white_+00 | target | 0.824 | 15.011 | 14.939 | 6.476 | 6.365 | 0.341 | 0.611 | 0.588 | 0.416 | 0.475 |
| harmonic_white_+00 | uniform | 0.259 | 15.002 | 14.951 | 6.091 | 5.945 | 0.303 | 0.601 | 0.568 | 0.413 | 0.455 |
| harmonic_pink_+05 | baseline | 0.0 | 14.172 | 13.912 | 6.792 | 12.4 | 0.71 | 0.563 | 0.775 | 0.64 | 0.667 |
| harmonic_pink_+05 | target | 0.15 | 14.996 | 14.585 | 6.856 | 12.55 | 0.706 | 0.592 | 0.773 | 0.647 | 0.676 |
| harmonic_pink_+05 | uniform | 0.067 | 14.995 | 14.687 | 6.721 | 12.497 | 0.692 | 0.586 | 0.769 | 0.643 | 0.669 |
| harmonic_traffic_+10 | baseline | 0.0 | 19.155 | 17.493 | 9.563 | 19.965 | 0.937 | 0.637 | 0.911 | 0.991 | 0.857 |
| harmonic_traffic_+10 | target | -0.489 | 15.008 | 13.903 | 8.953 | 19.206 | 0.94 | 0.559 | 0.912 | 1.0 | 0.832 |
| harmonic_traffic_+10 | uniform | -0.243 | 14.996 | 13.809 | 9.011 | 19.121 | 0.952 | 0.553 | 0.919 | 0.997 | 0.833 |
| music_white_+05 | baseline | 0.0 | 12.603 | 12.711 | 0.521 | 6.204 | 0.424 | 0.745 | 0.88 | 0.661 | 0.655 |
| music_white_+05 | target | 0.746 | 15.002 | 15.062 | 0.07 | 5.658 | 0.39 | 0.693 | 0.845 | 0.717 | 0.636 |
| music_white_+05 | uniform | 0.235 | 14.993 | 15.132 | -0.287 | 5.291 | 0.353 | 0.705 | 0.829 | 0.702 | 0.617 |
| music_pink_+00 | baseline | 0.0 | 13.322 | 13.676 | 4.551 | 2.928 | 0.477 | 0.356 | 0.709 | 0.686 | 0.536 |
| music_pink_+00 | target | 0.387 | 14.988 | 15.062 | 4.513 | 2.901 | 0.462 | 0.385 | 0.732 | 0.66 | 0.541 |
| music_pink_+00 | uniform | 0.147 | 14.991 | 15.165 | 4.412 | 2.77 | 0.438 | 0.373 | 0.724 | 0.656 | 0.528 |
| formant_white_+05 | baseline | 0.0 | 11.891 | 11.712 | 3.685 | 9.213 | 0.551 | 0.336 | 0.875 | 0.654 | 0.571 |
| formant_white_+05 | target | 1.168 | 14.998 | 14.679 | 3.392 | 9.177 | 0.507 | 0.292 | 0.885 | 0.693 | 0.549 |
| formant_white_+05 | uniform | 0.354 | 14.996 | 14.738 | 2.652 | 8.465 | 0.446 | 0.355 | 0.859 | 0.698 | 0.555 |
| formant_traffic_+05 | baseline | 0.0 | 19.093 | 18.42 | 11.615 | 16.807 | 0.915 | 0.416 | 0.895 | 0.937 | 0.751 |
| formant_traffic_+05 | target | -0.596 | 14.992 | 14.513 | 10.994 | 16.044 | 0.923 | 0.487 | 0.894 | 0.922 | 0.78 |
| formant_traffic_+05 | uniform | -0.255 | 14.992 | 14.447 | 10.946 | 15.908 | 0.938 | 0.47 | 0.903 | 0.932 | 0.78 |

## 4. 四视角证据

每个文件独立测量：`stability` = 特征在 4 组随机扰动下的稳定度；`auc` = 它区分信号格/噪声格的能力（干净参考为标签）。权重由这两项算出，不是常数。

| case | view | feature | stability | auc |
|---|---|---|---|---|
| harmonic_white_+05 | 能量 | cell_snr_db | 0.842 | 0.975 |
| harmonic_white_+05 | 能量 | frame_energy_db | 0.901 | 0.699 |
| harmonic_white_+05 | 频率 | band_snr_db | 0.933 | 0.971 |
| harmonic_white_+05 | 频率 | peak_persistence | 0.672 | 0.41 |
| harmonic_white_+05 | 形态 | stability_S | 0.679 | 0.948 |
| harmonic_white_+05 | 形态 | pivot_density | 0.747 | 0.624 |
| harmonic_white_+05 | 结构 | harmonic_comb | 0.942 | 0.878 |
| harmonic_white_+05 | 结构 | onset_protect | 0.968 | 0.502 |
| harmonic_white_+00 | 能量 | cell_snr_db | 0.863 | 0.969 |
| harmonic_white_+00 | 能量 | frame_energy_db | 0.916 | 0.704 |
| harmonic_white_+00 | 频率 | band_snr_db | 0.949 | 0.969 |
| harmonic_white_+00 | 频率 | peak_persistence | 0.702 | 0.449 |
| harmonic_white_+00 | 形态 | stability_S | 0.754 | 0.926 |
| harmonic_white_+00 | 形态 | pivot_density | 0.891 | 0.554 |
| harmonic_white_+00 | 结构 | harmonic_comb | 0.944 | 0.897 |
| harmonic_white_+00 | 结构 | onset_protect | 0.937 | 0.511 |
| harmonic_pink_+05 | 能量 | cell_snr_db | 0.802 | 0.978 |
| harmonic_pink_+05 | 能量 | frame_energy_db | 0.904 | 0.697 |
| harmonic_pink_+05 | 频率 | band_snr_db | 0.938 | 0.972 |
| harmonic_pink_+05 | 频率 | peak_persistence | 0.613 | 0.405 |
| harmonic_pink_+05 | 形态 | stability_S | 0.563 | 0.985 |
| harmonic_pink_+05 | 形态 | pivot_density | 0.365 | 0.827 |
| harmonic_pink_+05 | 结构 | harmonic_comb | 0.874 | 0.886 |
| harmonic_pink_+05 | 结构 | onset_protect | 0.959 | 0.501 |
| harmonic_traffic_+10 | 能量 | cell_snr_db | 0.737 | 0.978 |
| harmonic_traffic_+10 | 能量 | frame_energy_db | 0.875 | 0.665 |
| harmonic_traffic_+10 | 频率 | band_snr_db | 0.87 | 0.954 |
| harmonic_traffic_+10 | 频率 | peak_persistence | 0.568 | 0.4 |
| harmonic_traffic_+10 | 形态 | stability_S | 0.347 | 0.952 |
| harmonic_traffic_+10 | 形态 | pivot_density | 0.075 | 0.945 |
| harmonic_traffic_+10 | 结构 | harmonic_comb | 0.852 | 0.71 |
| harmonic_traffic_+10 | 结构 | onset_protect | 0.861 | 0.5 |
| music_white_+05 | 能量 | cell_snr_db | 0.84 | 0.98 |
| music_white_+05 | 能量 | frame_energy_db | 0.908 | 0.597 |
| music_white_+05 | 频率 | band_snr_db | 0.933 | 0.937 |
| music_white_+05 | 频率 | peak_persistence | 0.671 | 0.588 |
| music_white_+05 | 形态 | stability_S | 0.681 | 0.883 |
| music_white_+05 | 形态 | pivot_density | 0.765 | 0.601 |
| music_white_+05 | 结构 | harmonic_comb | 0.939 | 0.717 |
| music_white_+05 | 结构 | onset_protect | 0.96 | 0.509 |
| music_pink_+00 | 能量 | cell_snr_db | 0.818 | 0.975 |
| music_pink_+00 | 能量 | frame_energy_db | 0.919 | 0.571 |
| music_pink_+00 | 频率 | band_snr_db | 0.952 | 0.93 |
| music_pink_+00 | 频率 | peak_persistence | 0.633 | 0.594 |
| music_pink_+00 | 形态 | stability_S | 0.652 | 0.959 |
| music_pink_+00 | 形态 | pivot_density | 0.656 | 0.672 |
| music_pink_+00 | 结构 | harmonic_comb | 0.814 | 0.633 |
| music_pink_+00 | 结构 | onset_protect | 0.913 | 0.516 |
| formant_white_+05 | 能量 | cell_snr_db | 0.839 | 0.943 |
| formant_white_+05 | 能量 | frame_energy_db | 0.899 | 0.704 |
| formant_white_+05 | 频率 | band_snr_db | 0.936 | 0.954 |
| formant_white_+05 | 频率 | peak_persistence | 0.675 | 0.502 |
| formant_white_+05 | 形态 | stability_S | 0.687 | 0.864 |
| formant_white_+05 | 形态 | pivot_density | 0.752 | 0.608 |
| formant_white_+05 | 结构 | harmonic_comb | 0.936 | 0.598 |
| formant_white_+05 | 结构 | onset_protect | 0.855 | 0.556 |
| formant_traffic_+05 | 能量 | cell_snr_db | 0.69 | 0.972 |
| formant_traffic_+05 | 能量 | frame_energy_db | 0.898 | 0.719 |
| formant_traffic_+05 | 频率 | band_snr_db | 0.832 | 0.844 |
| formant_traffic_+05 | 频率 | peak_persistence | 0.592 | 0.516 |
| formant_traffic_+05 | 形态 | stability_S | 0.477 | 0.837 |
| formant_traffic_+05 | 形态 | pivot_density | 0.24 | 0.83 |
| formant_traffic_+05 | 结构 | harmonic_comb | 0.862 | 0.485 |
| formant_traffic_+05 | 结构 | onset_protect | 0.854 | 0.567 |

## 5. 怎么读

- **果**：`nr_floor` = 输入最安静 20% 帧的中位电平 − 输出同帧的中位电平。盲测，不需要干净参考。与参考掩码版本的差异通常在 0.2 dB 内；噪声非平稳（traffic）时可达约 1 dB。
- **因**：`dose c` 由单调二分反解到 ±0.03 dB；`I` 决定这份衰减落在哪些时频格。
- **四视角**：能量/频率/形态/结构各出一个保护分，加权共识成 `I`。
- **均匀施加**：同样的 15 dB，但 `I≡0`。对照组。
- **未命中**：`info.hit=False`。目标超出可达范围时控制器会报告，而不是偷偷输出一个别的数。
- **代价**：额外开销约 5–8%（8 s 音频实测：四视角建图 0.63 s + 反解 0.08 s，相对 10.9 s 的 loads）；多块文件只在中间切片上标定一次，所有块共用同一个 dose。

## 5.5 前沿：固定目标买到什么，花掉什么

| target | 实测 NR | SI-SDR（四视角） | SI-SDR（均匀） | 差 |
|---|---|---|---|---|
| baseline | 14.42 | 10.38 | 10.38 | — |
| 6 dB | 8.89 | 9.19 | 8.06 | +1.13 |
| 9 dB | 9.42 | 9.35 | 9.22 | +0.13 |
| 12 dB | 12.00 | 9.89 | 9.81 | +0.08 |
| 15 dB | 15.00 | 10.23 | 9.91 | +0.32 |
| 18 dB | 18.00 | 10.29 | 9.66 | +0.63 |
| 21 dB | 21.00 | 10.17 | 9.19 | +0.98 |
| 24 dB | 24.00 | 9.94 | 8.62 | +1.32 |

每个素材自己的最优目标分散在 9–24 dB；15 dB 只在 2/8 个素材上恰好最优，但离均值最优（18 dB）只差约 0.06 dB。见 `target15_frontier.png`。

## 6. 目标不可能时

把“15 dB”读成 **+15 dB SNR**（而不是噪声底 -15 dB）再跑一遍：同一文件上 ΔSNR 的可达上限只有 **+3.94 dB**（c=0.0），控制器返回 `hit=False` 并给出最接近的点，而不是伪造一个 15。

原因是残差被**信号失真**而不是残余噪声主导：噪声底可以压到 -24 dB，但把谐波/瞬态削掉的失真会把 SNR 拉回 3–4 dB 附近。这也说明两个“15 dB”不是同一件事，选错指标会导致一个永远达不到的目标。
