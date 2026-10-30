# 人声提取试听素材（v3.3）

同一段 约 3 s 英文语音（VoiceBank-DEMAND `p232_249`）与合成音乐按 0 dB 混合：

| 文件 | 内容 |
|---|---|
| `mix_speech_music.wav` | 混合输入 |
| `mix_npss_v27.wav` | NPSS v2.7 多统计 Gmin |
| `mix_voice_sep_v33.wav` | v3.3 人声提取（强度 0.8） |
| `pure_music_in.wav` / `pure_music_out.wav` | 纯音乐进/出（整段落到地板，−32 dB） |
| `compare.png` | 三层声谱 + 人声门（最下，白=保留） |

听点：最后一段音乐在语音结束后仍在响。v2.7 会留下（它是有结构的），
v3.3 的人声门在语音段结束后关闭，把这段音乐压掉。
