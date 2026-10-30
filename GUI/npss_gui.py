#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NPSS GUI: denoise audio files and replace the audio track of videos."""
import os
import sys
import queue
import threading
import subprocess
import tempfile
import os as _os
# 单线程 BLAS：小矩阵/小批量下反而更快（Accelerate 多线程开销大）
for _k in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

HERE = os.path.dirname(os.path.abspath(__file__))

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp",
            ".ppm", ".pgm", ".jp2"}
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
        root.title("NPSS 降噪 v3.1")
        root.geometry("1000x880")
        root.minsize(900, 720)

        style = ttk.Style()
        try:
            style.configure("TButton", font=FONT)
            style.configure("TLabel", font=FONT)
            style.configure("TRadiobutton", font=FONT)
            style.configure("TCheckbutton", font=FONT)
            style.configure("TCombobox", font=FONT)
            style.configure("Section.TLabelframe.Label", font=(FONT[0], FONT[1], "bold"))
        except Exception:
            pass

        main = ttk.Frame(root, padding=(18, 14))
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)

        ttk.Label(main, text="NPSS 降噪", font=("PingFang SC", 20, "bold")).grid(
            row=0, column=0, sticky="w")
        ttk.Label(main, text="音频 · 视频 · 图片 一体化降噪", font=SMALL,
                  foreground="#666666").grid(row=1, column=0, sticky="w", pady=(0, 12))

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
        self.vstr_var = tk.StringVar(value="0.8")
        self.loops_var = tk.StringVar(value="5")
        self.multi_var = tk.BooleanVar(value=True)
        self.fast_var = tk.BooleanVar(value=True)
        self.workers_var = tk.StringVar(value=str(min(os.cpu_count() or 4, 8)))
        self.auto_var = tk.BooleanVar(value=True)
        self.profile_var = tk.StringVar(value="频率优先（PESQ/STOI）")
        self.frames_var = tk.BooleanVar(value=True)

        # ---------------- 文件 ----------------
        files = ttk.LabelFrame(main, text=" 文件 ", padding=(12, 8), style="Section.TLabelframe")
        files.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        files.columnconfigure(1, weight=1)
        ttk.Label(files, text="输入").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(files, textvariable=self.in_var).grid(
            row=0, column=1, sticky="ew", padx=8)
        ttk.Button(files, text="浏览…", command=self.browse_input, width=8).grid(
            row=0, column=2, sticky="e")
        ttk.Label(files, text="输出").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(files, textvariable=self.out_var).grid(
            row=1, column=1, sticky="ew", padx=8)
        ttk.Button(files, text="另存…", command=self.browse_output, width=8).grid(
            row=1, column=2, sticky="e")

        # ---------------- 模式 ----------------
        modebox = ttk.LabelFrame(main, text=" 模式 ", padding=(12, 8), style="Section.TLabelframe")
        modebox.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        row1 = ttk.Frame(modebox)
        row1.pack(fill="x")
        for text, val in [("自动", "auto"), ("音频 WAV", "audio"),
                          ("视频 MP4", "video"), ("图片 PNG", "image")]:
            ttk.Radiobutton(row1, text=text, value=val, variable=self.mode_var,
                            command=self.on_mode).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(row1, text="视频画面也降噪", variable=self.frames_var).pack(
            side="left", padx=(16, 0))

        row2 = ttk.Frame(modebox)
        row2.pack(fill="x", pady=(8, 0))
        ttk.Radiobutton(row2, text="音乐为主（通用）", value="音乐为主",
                        variable=self.content_var,
                        command=self.on_content).pack(side="left", padx=(0, 16))
        ttk.Radiobutton(row2, text="声音为主（语音增强）", value="声音为主",
                        variable=self.content_var,
                        command=self.on_content).pack(side="left", padx=(0, 16))
        ttk.Radiobutton(row2, text="人声提取（只留人声）", value="人声提取",
                        variable=self.content_var,
                        command=self.on_content).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(row2, text="多统计 Gmin", variable=self.multi_var,
                        command=self.on_content).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(row2, text="快速模式", variable=self.fast_var).pack(side="left")

        self.content_hint = ttk.Label(modebox, text="", font=SMALL, foreground="#666666")
        self.content_hint.pack(fill="x", pady=(8, 0))

        # ---------------- 参数 ----------------
        opts = ttk.LabelFrame(main, text=" 参数 ", padding=(12, 8), style="Section.TLabelframe")
        opts.grid(row=4, column=0, sticky="ew", pady=(0, 10))

        def field(parent, r, c, label, widget):
            ttk.Label(parent, text=label).grid(row=r, column=c * 2, sticky="e",
                                               padx=(0, 6), pady=4)
            widget.grid(row=r, column=c * 2 + 1, sticky="w", padx=(0, 22), pady=4)

        field(opts, 0, 0, "噪声类型",
              ttk.Combobox(opts, textvariable=self.family_var, width=11,
                           values=["white", "correlated", "impulsive"], state="readonly"))
        field(opts, 0, 1, "passes",
              ttk.Spinbox(opts, from_=1, to=5, width=5, textvariable=self.passes_var))
        field(opts, 0, 2, "分块 (s)",
              ttk.Spinbox(opts, from_=5, to=60, width=5, textvariable=self.chunk_var))
        field(opts, 0, 3, "迭代",
              ttk.Spinbox(opts, from_=1, to=4, width=5, textvariable=self.iter_var))

        field(opts, 1, 0, "回注 β",
              ttk.Combobox(opts, textvariable=self.inject_var, width=9,
                           values=["0", "0.05", "0.1", "0.2"], state="readonly"))
        field(opts, 1, 1, "处理采样率",
              ttk.Combobox(opts, textvariable=self.proc_var, width=9,
                           values=["16000", "48000"], state="readonly"))
        field(opts, 1, 2, "旧循环",
              ttk.Spinbox(opts, from_=2, to=8, width=5, textvariable=self.loops_var))
        ttk.Checkbutton(opts, text="自动高 SNR 保护", variable=self.auto_var).grid(
            row=1, column=6, columnspan=2, sticky="w", padx=(0, 0), pady=4)

        field(opts, 2, 2, "人声强度",
              ttk.Spinbox(opts, from_=0.0, to=1.0, increment=0.05, width=5,
                          textvariable=self.vstr_var))
        field(opts, 2, 0, "四视角 profile",
              ttk.Combobox(opts, textvariable=self.profile_var, width=18,
                           values=["频率优先（PESQ/STOI）", "均衡（频率+结构）", "能量均衡",
                                   "结构优先（SI-SDR）", "关闭（v2.8）"], state="readonly"))

        # ---------------- 操作 ----------------
        btns = ttk.Frame(main)
        btns.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        self.start_btn = ttk.Button(btns, text="开始处理", command=self.start, width=12)
        self.start_btn.pack(side="left")
        for text, cmd in (("播放原文件", lambda: self.play(self.in_var.get())),
                          ("播放输出", lambda: self.play(self.out_var.get())),
                          ("打开输出目录", self.open_folder)):
            ttk.Button(btns, text=text, command=cmd).pack(side="left", padx=(8, 0))

        self.bar = ttk.Progressbar(main, mode="determinate")
        self.bar.grid(row=6, column=0, sticky="ew", pady=(2, 4))
        self.status = ttk.Label(main, text="就绪", font=SMALL, foreground="#666666")
        self.status.grid(row=7, column=0, sticky="w")

        self.nb = ttk.Notebook(main)
        self.nb.grid(row=8, column=0, sticky="nsew", pady=(8, 0))
        main.rowconfigure(8, weight=1)

        log_tab = ttk.Frame(self.nb)
        self.nb.add(log_tab, text="日志")
        self.log = ScrolledText(log_tab, height=12, font=("Menlo", 11))
        self.log.pack(fill="both", expand=True)

        self.plot_tab = ttk.Frame(self.nb)
        self.nb.add(self.plot_tab, text="波形 / 声谱")
        self.plot_label = ttk.Label(
            self.plot_tab, text="处理完成后显示 Original / Denoised 波形与声谱",
            anchor="center")
        self.plot_label.pack(fill="both", expand=True)
        self.plot_path_label = ttk.Label(self.plot_tab, text="", font=SMALL)
        self.plot_path_label.pack(fill="x")

        # ---------------- 音色 ----------------
        vtab = ttk.Frame(self.nb, padding=12)
        self.nb.add(vtab, text="音色")
        vtab.columnconfigure(1, weight=1)

        self.vref_var = tk.StringVar()
        self.vsrc_var = tk.StringVar()
        self.vout_var = tk.StringVar()

        for r, (label, var, dlg) in enumerate((
                ("参考音色", self.vref_var, self._pick_ref),
                ("源朗读", self.vsrc_var, self._pick_src),
                ("输出", self.vout_var, self._pick_out))):
            ttk.Label(vtab, text=label).grid(row=r, column=0, sticky="w", pady=4)
            ttk.Entry(vtab, textvariable=var).grid(row=r, column=1, sticky="ew", padx=8)
            ttk.Button(vtab, text="浏览…", width=8, command=dlg).grid(row=r, column=2)

        row = ttk.Frame(vtab)
        row.grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 6))
        ttk.Button(row, text="提取特征", width=10, command=self._analyze_ref).pack(side="left")
        ttk.Button(row, text="转换", width=10, command=self._do_convert).pack(side="left", padx=8)

        self.vfeat = tk.Text(vtab, height=8, font=("Menlo", 11), wrap="word")
        self.vfeat.grid(row=4, column=0, columnspan=3, sticky="nsew")
        vtab.rowconfigure(4, weight=1)

        self.on_content()
        ff = engine.find_ffmpeg()
        self.log_line("ffmpeg: " + ff if ff else
                      "警告：未找到 ffmpeg，请 pip install imageio-ffmpeg")
        self.root.after(120, self.poll)

    def log_line(self, s):
        self.log.insert("end", str(s) + "\n")
        self.log.see("end")

    def browse_input(self):
        ap = " ".join("*" + e for e in sorted(engine.AUDIO_EXTS))
        vp = " ".join("*" + e for e in sorted(engine.VIDEO_EXTS))
        ip = " ".join("*" + e for e in sorted(IMG_EXTS))
        p = filedialog.askopenfilename(
            title="选择音频 / 视频 / 图片",
            filetypes=[("所有支持的媒体", ip + " " + ap + " " + vp),
                       ("图片", ip), ("音频", ap), ("视频", vp),
                       ("所有文件", "*.*")])
        if not p:
            return
        self.in_var.set(p)
        base, ext = os.path.splitext(p)
        if ext.lower() in IMG_EXTS:
            self.mode_var.set("image")
            self.out_var.set(base + "_denoised.png")
        elif ext.lower() in engine.VIDEO_EXTS:
            self.mode_var.set("video")
            self.out_var.set(base + "_denoised.mp4")
        else:
            self.mode_var.set("audio")
            self.out_var.set(base + "_denoised.wav")

    def browse_output(self):
        video = self.mode_var.get() == "video"
        if video:
            pat = " ".join("*" + e for e in sorted(engine.VIDEO_OUT_EXTS))
            types = [("视频", pat), ("MP4", "*.mp4"), ("MKV", "*.mkv"),
                     ("WebM", "*.webm"), ("MOV", "*.mov"), ("AVI", "*.avi")]
            ext = ".mp4"
        else:
            pat = " ".join("*" + e for e in sorted(engine.AUDIO_OUT_EXTS))
            types = [("音频", pat), ("WAV", "*.wav"), ("FLAC", "*.flac"),
                     ("MP3", "*.mp3"), ("M4A", "*.m4a"), ("OGG", "*.ogg"),
                     ("Opus", "*.opus"), ("AIFF", "*.aiff")]
            ext = ".wav"
        p = filedialog.asksaveasfilename(
            title="保存为", defaultextension=ext, filetypes=types)
        if p:
            self.out_var.set(p)

    def on_content(self, event=None):
        multi = bool(self.multi_var.get())
        if self.content_var.get() == "人声提取":
            self.content_hint.config(
                text="基频 + 谐波梳 + 共振峰 + 音节调制 → 只留人声；"
                     "停顿时接近静音，音乐/机器/风噪按人声特征压掉")
        elif self.content_var.get() == "声音为主":
            self.content_hint.config(
                text="16k/48k 多统计 + 拓扑 + 角色补全 + 能量 + NMF；立体声" if multi
                else "16k 自动互补 + NMF；48k 旧 NPSS")
        else:
            self.content_hint.config(
                text="16k/48k 多统计 + 拓扑 + 角色补全 + 高频柔和；立体声" if multi
                else "16k 自动互补；48k 旧 NPSS")

    def on_mode(self):
        p = self.out_var.get()
        if not p:
            return
        base, ext = os.path.splitext(p)
        if self.mode_var.get() == "image":
            if ext.lower() not in IMG_EXTS:
                self.out_var.set(base + "_denoised.png")
            return
        if self.mode_var.get() == "video" and ext.lower() not in engine.VIDEO_OUT_EXTS:
            self.out_var.set(base + ".mp4")
        elif self.mode_var.get() == "audio" and ext.lower() not in engine.AUDIO_OUT_EXTS:
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
        if mode == "image":
            threading.Thread(target=self.worker_image, args=(src, dst),
                             daemon=True).start()
            return
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
            vstr = max(0.0, min(1.0, float(self.vstr_var.get())))
            content = {"声音为主": "voice", "人声提取": "speech"}.get(
                self.content_var.get(), "music")
            complement = True
            loops = max(2, min(8, int(self.loops_var.get())))
            multi = bool(self.multi_var.get())
            fast = bool(self.fast_var.get())
            profile = {"频率优先（PESQ/STOI）": "freq",
                       "均衡（频率+结构）": "blend",
                       "能量均衡": "energy",
                       "结构优先（SI-SDR）": "struct",
                       "关闭（v2.8）": "off"}.get(self.profile_var.get(), "freq")
            target = None
            if multi and proc_sr > 16000:
                self.log_line("48k 全带 + 立体声 + 跨区拓扑：16k gate + 8k+ 保留")
            elif content == "music" and proc_sr <= 16000:
                self.log_line("提示：音乐会带限到 8k；建议处理采样率 48k")
            if content == "speech":
                self.log_line("人声提取：基频/谐波/共振峰/音节调制 → 只保留人声，强度 %.2f" % vstr)
            if content == "voice" and proc_sr != 16000:
                self.log_line("提示：声音模式的 NMF 字典是 16k；当前 %d Hz 只跑 NPSS" % proc_sr)
        except Exception:
            passes, chunk, proc_sr, iterations, inject, auto, theta, support, backend, workers = 3, 20.0, 16000, 1, 0.0, True, 0.95, 0.60, "specsub", 4
            vstr = 0.8
            profile = "freq"
            target = None
            content = "music"
            complement = True
            loops = 2
            multi = True
            fast = True
        t = threading.Thread(target=self.worker,
                             args=(src, dst, mode, fam, passes, chunk, proc_sr,
                                   iterations, inject, auto, theta, support,
                                   backend, workers, content, complement, loops, multi, fast, profile, vstr),
                             daemon=True)
        t.start()

    def worker_image(self, src, dst):
        try:
            import numpy as np
            from PIL import Image
            import npss_image as M
            if os.path.splitext(src)[1].lower() in engine.VIDEO_EXTS:
                import still_merge as SM
                self.q.put(("log", "视频取帧：融合连续帧生成一张干净图片 …"))
                img, info = SM.still(src, t_sec=None, count=None,
                                     log=lambda s: self.q.put(("log", s)))
                Image.fromarray(M.to_uint8(img)).save(dst)
                self.q.put(("done", dst))
                return
            self.q.put(("log", "图片降噪：自动估计噪声 …"))
            img = Image.open(src).convert("RGB")
            a = M.to_float(img)
            sig = float(np.median(np.abs(M.luma(a) - M.gaussian_blur(M.luma(a), 1.2))) / 0.6745)
            self.q.put(("log", "噪声 σ ≈ %.1f/255" % (sig * 255)))
            out, _ = M.denoise(a, sigma=sig)
            Image.fromarray(M.to_uint8(out)).save(dst)
            self.q.put(("done", dst))
        except Exception as e:
            self.q.put(("error", str(e)))

    def worker(self, src, dst, mode, fam, passes, chunk, proc_sr, iterations, inject, auto, theta, support, backend, workers, content, complement, loops, multi, fast, profile="freq", vstr=0.8, target=None):
        try:
            if mode == "video" and self.frames_var.get():
                import av_denoise
                self.q.put(("log", "视频：音轨 + 画面同时降噪（轨迹累积）"))
                try:
                    av_denoise.denoise_video_traj(
                        src, dst, sigma=None, audio_target=target, fast=fast,
                        radius=(2 if fast else 3), content=content,
                        voice_strength=vstr,
                        log=lambda s: self.q.put(("log", s)))
                except Exception as e:
                    self.q.put(("log", "轨迹累积失败，退回逐帧降噪：%s" % e))
                    av_denoise.denoise_video(
                        src, dst, sigma=None, audio_target=target, fast=fast,
                        content=content, voice_strength=vstr,
                        log=lambda s: self.q.put(("log", s)))
                self.q.put(("done", dst))
                return
            engine.process_file(
                src, dst, mode=mode, family=fam, passes=passes,
                chunk_sec=chunk, proc_sr=proc_sr, out_sr=proc_sr,
                iterations=iterations, inject=inject, auto=auto,
                theta=theta, support=support, backend=backend, workers=workers,
                content_mode=content, complement=complement, loops=loops,
                multi_stat=multi, fast=fast, target_nr_db=target,
                view_profile=profile, voice_strength=vstr,
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



    # ---------------- 音色 ----------------
    def _pick_ref(self):
        p = filedialog.askopenfilename(title="选择参考音色",
                                       filetypes=[("音频", "*.wav *.mp3 *.m4a *.flac *.aac *.mp4 *.mov"), ("全部", "*.*")])
        if p:
            self.vref_var.set(p)

    def _pick_src(self):
        p = filedialog.askopenfilename(title="选择要转换的朗读",
                                       filetypes=[("音频", "*.wav *.mp3 *.m4a *.flac *.aac *.mp4 *.mov"), ("全部", "*.*")])
        if p:
            self.vsrc_var.set(p)

    def _pick_out(self):
        p = filedialog.asksaveasfilename(title="保存为", defaultextension=".wav",
                                         filetypes=[("WAV", "*.wav")])
        if p:
            self.vout_var.set(p)

    def _load_any(self, path):
        import av_denoise
        import common as C
        if os.path.splitext(path)[1].lower() in (".wav", ".flac", ".aiff"):
            x, sr = C.read_wav(path)
            return x, sr
        import subprocess
        ff = av_denoise._ffmpeg()
        tmp = tempfile.mktemp(suffix=".wav")
        subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-i", path,
                        "-vn", "-ac", "1", "-ar", "16000", tmp], check=False)
        x, sr = C.read_wav(tmp)
        return x, sr

    def _analyze_ref(self):
        path = self.vref_var.get().strip()
        if not path:
            self._vlog("先选一个参考音色文件。")
            return
        try:
            import voice_transform as V
            x, sr = self._load_any(path)
            A = V.analyze(x, sr)
            lines = [
                "参考音色：%s" % os.path.basename(path),
                "时长        %.1f s" % A["duration"],
                "基频 F0     中位 %.0f Hz（%.0f–%.0f）" % (A["f0_median"], A["f0_low"], A["f0_high"]),
                "共振峰      " + "  ".join("%.0f Hz" % f for f, _ in A["formants"]),
                "谱质心      %.0f Hz" % A["centroid"],
                "语速        %.1f 段/秒" % A["rate"],
                "停顿        %.0f%%" % (100 * A["pause"]),
            ]
            self._vlog("\n".join(lines))
        except Exception as e:
            self._vlog("分析失败：%s" % e)

    def _do_convert(self):
        ref, src, out = (self.vref_var.get().strip(), self.vsrc_var.get().strip(),
                         self.vout_var.get().strip())
        if not (ref and src and out):
            self._vlog("参考、源朗读、输出三项都要填。")
            return
        def work():
            try:
                import voice_transform as V
                import common as C
                xr, sr = self._load_any(ref)
                xs, _ = self._load_any(src)
                tr = tempfile.mktemp(suffix="_ref.wav")
                ts = tempfile.mktemp(suffix="_src.wav")
                C.write_wav(tr, xr, sr)
                C.write_wav(ts, xs, sr)
                self._vlog("转换中…")
                V.convert(tr, ts, out)
                self._vlog("完成：%s" % out)
            except Exception as e:
                self._vlog("转换失败：%s" % e)
        threading.Thread(target=work, daemon=True).start()

    def _vlog(self, text):
        self.vfeat.delete("1.0", "end")
        self.vfeat.insert("1.0", text)

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
