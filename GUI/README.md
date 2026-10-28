# GUI（v2.7）

```bash
./run_gui.command
# 或
python npss_gui.py
```

- 内容：音乐为主 / 声音为主。
- **多统计 Gmin（v2.7）**：默认开启；拓扑 + 高/低贡献补全 + 能量/SNR 加权。
- 16k/48k 都走多统计；48k = 16k gate + 8k+ 柔和（×0.70）。
- 输出保留**立体声**；左右声道分别处理。
- 音乐建议处理采样率选 **48k**；16k 会带限到 8k。
- 结果（16k）：音乐 13.92 / PESQ 2.505 / STOI 0.9082；声音 14.03 / PESQ 2.489 / STOI 0.9116。
- 能量加权只用于 16k；48k 全带路径关闭，避免高频被相对压低。
- 48k：8k+ 柔和（×0.70，约 −2.15 dB）；真立体声约 2× 计算。

## 视频一体化降噪（画面 + 声音）

`av_denoise.py` 把图像降噪（NPSS-Image）和音频降噪（本仓库的多统计 Gmin）
封装进同一个视频处理入口：输入视频，音轨和画面分别降噪，再合成输出。

```bash
python3 av_denoise.py in.mp4 out.mp4                  # 自动
python3 av_denoise.py in.mp4 out.mp4 --sigma 20 --fast
python3 av_denoise.py in.mp4 out.mp4 --audio-target 12 --content voice
```

或在代码里：

```python
import engine
engine.process_video_av("in.mp4", "out.mp4", sigma=20, fast=True)
```

| 参数 | 说明 |
|---|---|
| `--sigma` | 画面噪声 std（0-255）；缺省自动估计 |
| `--audio-target` | 音频目标降噪 dB（定果寻因）；缺省用默认档 |
| `--content` | 音轨类型 `voice`（默认）/ `music` |
| `--fast` | 画面单相位，快约 4 倍 |
| `--spins` | 画面相位密度 1/2/4（默认 fast?1:2） |

实测（768×512、12 fps、2 s 测试片）：

- 画面：带噪帧 27.18 dB → 降噪后 30.53 dB（首帧，PSNR vs 干净帧）
- 声音：6.83 dB → 9.51 dB（SNR vs 干净音轨）
- 耗时：`--fast` 下约 0.2 s/帧（768×512），1080p 约 0.8–1.2 s/帧

注意：逐帧处理是 CPU 密集操作，长视频建议先用 `--fast` 试片，确认效果后再全量。
