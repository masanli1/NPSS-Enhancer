#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NPSS GUI: denoise audio files and replace the audio track of videos."""
import os
import sys
import queue
import threading
import subprocess
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine
import common as C

FONT = ("PingFang SC", 13)
SMALL = ("PingFang SC", 11)


class App:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.busy = False
        self._plot_photo = None
        self._plot_img = None
        self._last_plot_path = None
        self._spec_lut = None
        root.title("NPSS 降噪 — 多统计 Gmin 最佳版 v2.3")
        root.geometry("900x790")
        root.minsize(760, 620)

        style = ttk.Style()
        try:
            style.configure("TButton", font=FONT)
            style.configure("TLabel", font=FONT)
            style.configure("TRadiobutton", font=FONT)
            style.configure("TCombobox", font=FONT)
        except Exception:
            pass

        main = ttk.Frame(root, padding=14)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)

        ttk.Label(main, text="NPSS 降噪 — 多统计 Gmin 最佳版 v2.3",
                  font=("PingFang SC", 16, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 12))

        self.in_var = tk.StringVar()
        self.out_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="auto")
        self.family_var = tk.StringVar(value="correlated")
        self.passes_var = tk.StringVar(value="3")
        self.chunk_var = tk.StringVar(value="20")
        self.proc_var = tk.StringVar(value="16000")
        self.iter_var = tk.StringVar(value="1")
        self.inject_var = tk.StringVar(value="0")
        self.theta_var = tk.StringVar(value="0.95")
        self.support_var = tk.StringVar(value="0.60")
        self.backend_var = tk.StringVar(value="specsub")
        self.content_var = tk.StringVar(value="音乐为主")
        self.loops_var = tk.StringVar(value="5")
        self.multi_var = tk.BooleanVar(value=True)
        self.workers_var = tk.StringVar(value=str(min(os.cpu_count() or 4, 8)))
        self.auto_var = tk.BooleanVar(value=True)

        ttk.Label(main, text="输入").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(main, textvariable=self.in_var).grid(
            row=1, column=1, columnspan=2, sticky="ew", padx=6)
        ttk.Button(main, text="浏览…", command=self.browse_input).grid(
            row=1, column=3, sticky="ew")

        ttk.Label(main, text="输出").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(main, textvariable=self.out_var).grid(
            row=2, column=1, columnspan=2, sticky="ew", padx=6)
        ttk.Button(main, text="另存…", command=self.browse_output).grid(
            row=2, column=3, sticky="ew")

        mode = ttk.Frame(main)
        mode.grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 2))
        ttk.Label(mode, text="模式").pack(side="left", padx=(0, 8))
        for text, val in [("自动", "auto"), ("音频 WAV", "audio"),
                          ("视频 MP4（替换音轨）", "video")]:
            ttk.Radiobutton(mode, text=text, value=val,
                            variable=self.mode_var,
                            command=self.on_mode).pack(side="left", padx=4)

        content = ttk.Frame(main)
        content.grid(row=4, column=0, columnspan=4, sticky="w", pady=(2, 0))
        ttk.Label(content, text="内容").pack(side="left", padx=(0, 8))
        ttk.Radiobutton(content, text="音乐为主（通用）", value="音乐为主",
                        variable=self.content_var,
                        command=self.on_content).pack(side="left", padx=4)
        ttk.Radiobutton(content, text="声音为主（语音增强）", value="声音为主",
                        variable=self.content_var,
                        command=self.on_content).pack(side="left", padx=4)
        ttk.Label(content, text="旧循环").pack(side="left", padx=(10, 2))
        ttk.Spinbox(content, from_=2, to=8, width=3,
                    textvariable=self.loops_var).pack(side="left")
        ttk.Checkbutton(content, text="多统计 Gmin（v2.3）",
                        variable=self.multi_var,
                        command=self.on_content).pack(side="left", padx=(8, 2))
        self.content_hint = ttk.Label(content, text="16k/48k 多统计 + 混合；立体声", font=SMALL)
        self.content_hint.pack(side="left", padx=6)

        opts = ttk.Frame(main)
        opts.grid(row=5, column=0, columnspan=4, sticky="w", pady=8)
        ttk.Label(opts, text="噪声类型").pack(side="left")
        ttk.Combobox(opts, textvariable=self.family_var, width=11,
                     values=["white", "correlated", "impulsive"],
                     state="readonly").pack(side="left", padx=(4, 14))
        ttk.Label(opts, text="passes").pack(side="left")
        ttk.Spinbox(opts, from_=1, to=5, width=4,
                    textvariable=self.passes_var).pack(side="left", padx=(4, 14))
        ttk.Label(opts, text="分块(s)").pack(side="left")
        ttk.Spinbox(opts, from_=5, to=60, width=5,
                    textvariable=self.chunk_var).pack(side="left", padx=(4, 14))
        ttk.Label(opts, text="迭代").pack(side="left")
        ttk.Spinbox(opts, from_=1, to=4, width=4,
                    textvariable=self.iter_var).pack(side="left", padx=(4, 14))
        ttk.Label(opts, text="回注 β").pack(side="left")
        ttk.Combobox(opts, textvariable=self.inject_var, width=5,
                     values=["0", "0.05", "0.1", "0.2"],
                     state="readonly").pack(side="left", padx=(4, 14))
        ttk.Checkbutton(opts, text="自动高 SNR 保护",
                        variable=self.auto_var).pack(side="left", padx=(14, 0))
        ttk.Label(opts, text="处理采样率").pack(side="left")
        ttk.Combobox(opts, textvariable=self.proc_var, width=7,
                     values=["16000", "48000"],
                     state="readonly").pack(side="left", padx=4)

        btns = ttk.Frame(main)
        btns.grid(row=6, column=0, columnspan=4, sticky="w", pady=(6, 8))
        self.start_btn = ttk.Button(btns, text="开始处理", command=self.start)
        self.start_btn.pack(side="left")
        ttk.Button(btns, text="播放原文件", command=lambda: self.play(self.in_var.get())).pack(side="left", padx=6)
        ttk.Button(btns, text="播放输出", command=lambda: self.play(self.out_var.get())).pack(side="left", padx=6)
        ttk.Button(btns, text="打开输出目录", command=self.open_folder).pack(side="left", padx=6)

        self.bar = ttk.Progressbar(main, mode="determinate")
        self.bar.grid(row=7, column=0, columnspan=4, sticky="ew", pady=4)
        self.status = ttk.Label(main, text="就绪", font=SMALL)
        self.status.grid(row=8, column=0, columnspan=4, sticky="w")

        self.nb = ttk.Notebook(main)
        self.nb.grid(row=9, column=0, columnspan=4, sticky="nsew", pady=(8, 0))
        main.rowconfigure(9, weight=1)

        log_tab = ttk.Frame(self.nb)
        self.nb.add(log_tab, text="日志")
        self.log = ScrolledText(log_tab, height=13, font=("Menlo", 11))
        self.log.pack(fill="both", expand=True)

        self.plot_tab = ttk.Frame(self.nb)
        self.nb.add(self.plot_tab, text="波形 / 声谱")
        self.plot_label = ttk.Label(
            self.plot_tab, text="处理完成后显示 Original / Denoised 波形与声谱",
            anchor="center")
        self.plot_label.pack(fill="both", expand=True)
        self.plot_path_label = ttk.Label(self.plot_tab, text="", font=SMALL)
        self.plot_path_label.pack(fill="x")

        self.on_content()
        ff = engine.find_ffmpeg()
        self.log_line("ffmpeg: " + ff if ff else
                      "警告：未找到 ffmpeg，请 pip install imageio-ffmpeg")
        self.root.after(120, self.poll)

    def log_line(self, s):
        self.log.insert("end", str(s) + "\n")
        self.log.see("end")

    def browse_input(self):
        p = filedialog.askopenfilename(
            title="选择音频或视频",
            filetypes=[("媒体文件", "*.*"),
                       ("音频", "*.wav *.mp3 *.flac *.m4a *.aac *.ogg"),
                       ("视频", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v")])
        if not p:
            return
        self.in_var.set(p)
        base, ext = os.path.splitext(p)
        if ext.lower() in engine.VIDEO_EXTS:
            self.mode_var.set("video")
            self.out_var.set(base + "_denoised.mp4")
        else:
            self.mode_var.set("audio")
            self.out_var.set(base + "_denoised.wav")

    def browse_output(self):
        video = self.mode_var.get() == "video"
        ext = ".mp4" if video else ".wav"
        p = filedialog.asksaveasfilename(
            title="保存为", defaultextension=ext,
            filetypes=[("MP4 视频", "*.mp4")] if video else [("WAV 音频", "*.wav")])
        if p:
            self.out_var.set(p)

    def on_content(self, event=None):
        multi = bool(self.multi_var.get())
        if self.content_var.get() == "声音为主":
            self.content_hint.config(
                text="16k/48k 多统计 + NMF；立体声" if multi
                else "16k 自动互补 + NMF；48k 旧 NPSS")
        else:
            self.content_hint.config(
                text="16k/48k 多统计 + 混合；立体声" if multi
                else "16k 自动互补；48k 旧 NPSS")

    def on_mode(self):
        p = self.out_var.get()
        if not p:
            return
        base, ext = os.path.splitext(p)
        if self.mode_var.get() == "video" and ext.lower() not in (".mp4", ".mov", ".mkv"):
            self.out_var.set(base + ".mp4")
        elif self.mode_var.get() == "audio" and ext.lower() != ".wav":
            self.out_var.set(base + ".wav")

    def start(self):
        if self.busy:
            return
        src = self.in_var.get().strip()
        dst = self.out_var.get().strip()
        if not src or not os.path.exists(src):
            messagebox.showwarning("提示", "请选择存在的输入文件")
            return
        if not dst:
            messagebox.showwarning("提示", "请选择输出文件")
            return
        self._clear_last_plot()
        self.busy = True
        self.start_btn.config(state="disabled")
        self.bar["value"] = 0
        self.status.config(text="处理中…")
        self.log_line("—— 开始 ——")
        mode = self.mode_var.get()
        fam = self.family_var.get()
        try:
            passes = max(1, min(5, int(self.passes_var.get())))
            chunk = max(5.0, min(60.0, float(self.chunk_var.get())))
            proc_sr = int(self.proc_var.get())
            iterations = max(1, min(4, int(self.iter_var.get())))
            inject = max(0.0, min(0.5, float(self.inject_var.get())))
            auto = bool(self.auto_var.get())
            theta = float(self.theta_var.get())
            support = float(self.support_var.get())
            backend = self.backend_var.get()
            workers = max(1, min(16, int(self.workers_var.get())))
            content = "voice" if self.content_var.get() == "声音为主" else "music"
            complement = True
            loops = max(2, min(8, int(self.loops_var.get())))
            multi = bool(self.multi_var.get())
            if multi and proc_sr > 16000:
                self.log_line("48k 全带 + 立体声：16k gate + 8k+ 保留")
            elif content == "music" and proc_sr <= 16000:
                self.log_line("提示：音乐会带限到 8k；建议处理采样率 48k")
            if content == "voice" and proc_sr != 16000:
                self.log_line("提示：声音模式的 NMF 字典是 16k；当前 %d Hz 只跑 NPSS" % proc_sr)
        except Exception:
            passes, chunk, proc_sr, iterations, inject, auto, theta, support, backend, workers = 3, 20.0, 16000, 1, 0.0, True, 0.95, 0.60, "specsub", 4
            content = "music"
            complement = True
            loops = 2
            multi = True
        t = threading.Thread(target=self.worker,
                             args=(src, dst, mode, fam, passes, chunk, proc_sr,
                                   iterations, inject, auto, theta, support,
                                   backend, workers, content, complement, loops, multi),
                             daemon=True)
        t.start()

    def worker(self, src, dst, mode, fam, passes, chunk, proc_sr, iterations, inject, auto, theta, support, backend, workers, content, complement, loops, multi):
        try:
            engine.process_file(
                src, dst, mode=mode, family=fam, passes=passes,
                chunk_sec=chunk, proc_sr=proc_sr, out_sr=proc_sr,
                iterations=iterations, inject=inject, auto=auto,
                theta=theta, support=support, backend=backend, workers=workers,
                content_mode=content, complement=complement, loops=loops,
                multi_stat=multi,
                progress=lambda d, t: self.q.put(("progress", d, t)),
                log=lambda s: self.q.put(("log", s)))
            self.q.put(("done", dst))
            threading.Thread(target=self._plot_worker, args=(src, dst, proc_sr),
                             daemon=True).start()
        except Exception as e:
            self.q.put(("error", str(e)))

    def poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "log":
                    self.log_line(msg[1])
                elif msg[0] == "progress":
                    d, t = msg[1], msg[2]
                    self.bar["maximum"] = max(1, t)
                    self.bar["value"] = d
                    self.status.config(text="处理中… %d/%d" % (d, t))
                elif msg[0] == "plot":
                    self._show_plot(msg[1], msg[2] if len(msg) > 2 else None)
                elif msg[0] == "done":
                    self.busy = False
                    self.start_btn.config(state="normal")
                    self.status.config(text="完成")
                    self.log_line("完成：" + msg[1])
                    messagebox.showinfo("完成", "已生成：\n" + msg[1])
                elif msg[0] == "error":
                    self.busy = False
                    self.start_btn.config(state="normal")
                    self.status.config(text="失败")
                    self.log_line("错误：" + msg[1])
                    messagebox.showerror("失败", msg[1][:1500])
        except queue.Empty:
            pass
        self.root.after(120, self.poll)

    def _load_audio(self, path, plot_sr):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"):
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            try:
                engine.extract_audio(path, tmp, plot_sr, None)
                x, sr = C.read_wav(tmp)
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            return x, sr
        x, sr = C.read_wav(path)
        if len(x) and sr != plot_sr:
            t = np.arange(int(len(x) * plot_sr / sr)) / float(plot_sr)
            x = np.interp(t, np.arange(len(x)) / float(sr), x).astype(np.float32)
            sr = plot_sr
        return x, sr

    def _plot_worker(self, src, dst, proc_sr):
        try:
            plot_sr = 48000 if int(proc_sr) == 48000 else 16000
            o, sr = self._load_audio(src, plot_sr)
            d, _ = self._load_audio(dst, plot_sr)
            img = self._make_plot_image(o, d, sr)
            base = os.path.splitext(dst)[0]
            plot_path = base + "_plot.png"
            try:
                img.save(plot_path)
            except Exception as e:
                plot_path = None
                self.q.put(("log", "波形图保存失败：" + str(e)))
            self.q.put(("plot", img, plot_path))
        except Exception as e:
            self.q.put(("log", "绘图失败：" + str(e)))

    def _show_plot(self, img, path=None):
        self._plot_img = img
        self._plot_photo = ImageTk.PhotoImage(img)
        self.plot_label.configure(image=self._plot_photo, text="")
        self._last_plot_path = path
        self.plot_path_label.config(
            text=("已保存：" + path) if path else "波形图未保存")
        self.nb.select(self.plot_tab)
        if path:
            self.log_line("波形图已保存：" + path)

    def _clear_last_plot(self):
        p = getattr(self, "_last_plot_path", None)
        if p:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        self._last_plot_path = None
        self._plot_img = None
        self._plot_photo = None
        try:
            self.plot_label.configure(
                image="", text="处理完成后显示 Original / Denoised 波形与声谱")
            self.plot_path_label.config(text="")
        except Exception:
            pass

    def _lut(self):
        if self._spec_lut is None:
            stops = [(0, 0, 0), (14, 0, 40), (40, 0, 90), (90, 10, 90),
                     (150, 40, 60), (210, 90, 20), (245, 170, 40),
                     (255, 240, 180)]
            out = []
            for i in range(256):
                t = i / 255.0 * (len(stops) - 1)
                k = int(t)
                f = t - k
                if k >= len(stops) - 1:
                    c = stops[-1]
                else:
                    c = tuple(int(stops[k][j] * (1 - f) + stops[k + 1][j] * f)
                              for j in range(3))
                out.append(c)
            self._spec_lut = np.array(out, dtype=np.uint8)
        return self._spec_lut

    def _spec_rgb(self, x, sr, w, h):
        nfft, hop = 1024, 512
        if len(x) < nfft:
            x = np.pad(x, (0, nfft - len(x)))
        win = np.hanning(nfft).astype(np.float32)
        fr = np.lib.stride_tricks.sliding_window_view(x, nfft)[::hop] * win
        S = np.abs(np.fft.rfft(fr, axis=1)).T
        Sdb = 20.0 * np.log10(S / (S.max() + 1e-8) + 1e-8)
        Sdb = np.clip(Sdb, -80, 0)[::-1]
        im = Image.fromarray(Sdb.astype(np.float32), mode="F").resize(
            (w, h), Image.BILINEAR)
        a = np.array(im)
        t = np.clip((a + 80.0) / 80.0, 0, 1)
        idx = (t * 255).astype(np.uint8)
        return self._lut()[idx]

    def _make_plot_image(self, o, d, sr):
        n = min(len(o), len(d))
        if n == 0:
            raise ValueError("空音频")
        o, d = o[:n], d[:n]
        if n > 12 * sr:
            win, step = 12 * sr, sr
            best, start = -1.0, 0
            for s in range(0, n - win, step):
                e = float(np.mean(o[s:s + win] ** 2))
                if e > best:
                    best, start = e, s
            o = o[start:start + win]
            d = d[start:start + win]
        n = len(o)
        W, H = 860, 420
        L, R, T, B = 62, 16, 26, 26
        gap = 12
        col = (W - L - R - gap) // 2
        wave_h, spec_h = 78, 230
        spec_top = T + wave_h + 34
        img = Image.new("RGB", (W, H), (250, 250, 250))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 13)
            fsmall = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 10)
        except Exception:
            font = ImageFont.load_default()
            fsmall = font
        peak = max(float(np.max(np.abs(o))), float(np.max(np.abs(d))), 1e-6)
        mid = T + wave_h // 2
        half = wave_h // 2 - 2
        xs = [L, L + col + gap]
        for k, (arr, color, title) in enumerate([
                (o, (50, 110, 210), "Original"),
                (d, (20, 150, 80), "Denoised")]):
            x0 = xs[k]
            draw.rectangle([x0, T, x0 + col, T + wave_h], outline=(200, 200, 200))
            draw.line([(x0, mid), (x0 + col, mid)], fill=(225, 225, 225))
            draw.text((x0, 2), title + " waveform", fill=(20, 20, 20), font=font)
            for i in range(col):
                s0 = int(i * n / col)
                s1 = max(s0 + 1, int((i + 1) * n / col))
                seg = arr[s0:s1]
                mn, mx = float(np.min(seg)), float(np.max(seg))
                y0 = mid - int(mx / peak * half)
                y1 = mid - int(mn / peak * half)
                draw.line([(x0 + i, y0), (x0 + i, y1)], fill=color)
            # spectrogram
            rgb = self._spec_rgb(arr, sr, col, spec_h)
            img.paste(Image.fromarray(rgb, "RGB"), (x0, spec_top))
            draw.rectangle([x0, spec_top, x0 + col, spec_top + spec_h],
                           outline=(200, 200, 200))
            draw.text((x0, spec_top - 18), title + " spectrogram",
                      fill=(20, 20, 20), font=font)
        # frequency axis (left of first spectrogram)
        for hz in (0, 4000, 8000, 16000, 24000):
            if hz > sr / 2:
                continue
            y = spec_top + spec_h - int(hz / (sr / 2.0) * spec_h)
            draw.line([(L - 4, y), (L, y)], fill=(80, 80, 80))
            draw.text((4, y - 6), "%dk" % (hz // 1000), fill=(40, 40, 40), font=fsmall)
        # time axis under first spectrogram
        dur = n / float(sr)
        for sec in range(0, int(dur) + 1, 2):
            x = L + int(sec / max(dur, 1e-6) * col)
            draw.line([(x, spec_top + spec_h), (x, spec_top + spec_h + 4)],
                      fill=(80, 80, 80))
            draw.text((x - 5, spec_top + spec_h + 5), str(sec),
                      fill=(40, 40, 40), font=fsmall)
        draw.text((4, spec_top + spec_h // 2), "Hz", fill=(40, 40, 40), font=fsmall)
        draw.text((L + col // 2, H - 12), "time (s)", fill=(40, 40, 40), font=fsmall)
        return img

    def play(self, path):
        if not path or not os.path.exists(path):
            return
        try:
            if engine.is_video(path):
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["afplay", path])
        except Exception as e:
            messagebox.showerror("播放失败", str(e))

    def open_folder(self):
        p = self.out_var.get().strip()
        if p and os.path.exists(p):
            subprocess.Popen(["open", "-R", p])
        elif p:
            folder = os.path.dirname(os.path.abspath(p))
            if os.path.isdir(folder):
                subprocess.Popen(["open", folder])


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
