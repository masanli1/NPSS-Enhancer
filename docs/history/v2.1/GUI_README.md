# GUI（v2.1）

```bash
./run_gui.command
# 或
python npss_gui.py
```

- 内容：音乐为主 / 声音为主。
- **多统计 Gmin（v2.1）**：默认开启；16k 走多统计，48k 自动回旧 NPSS。
- 旧循环只对非多统计模式生效；多统计固定 3-pass/2-loop。
- 结果：音乐 13.77 / PESQ 2.487 / STOI 0.9071；声音 13.92 / PESQ 2.480 / STOI 0.9110。
- 多统计约 6× 单扰动耗时。
