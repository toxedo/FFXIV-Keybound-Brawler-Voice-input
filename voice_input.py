#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FF14 语音输入 —— 按住热键说话，松开自动打进游戏输入框。

流程：按住说话 -> faster-whisper 识别 -> 吸附到 1210 条词表 -> WM_CHAR 投进游戏 -> 回车

用法：
    python voice_input.py                  # 启动（默认按住说话）
    python voice_input.py --model small    # 换更大的模型
    python voice_input.py --no-send        # 只填字，不自动回车提交
    python voice_input.py keyscan          # 看哪些热键没被占用

依赖：pip install sounddevice faster-whisper pypinyin
前置：vocab_clean.txt（词表，一行一个）
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import re
import sys
import threading
import time
import unicodedata
from pathlib import Path

SR = 16000
VOCAB_FILE = Path("vocab_clean.txt")
HOTKEY_FILE = Path("hotkey.txt")

# ---------------------------------------------------------------- 词表与对齐

PUNCT = re.compile(r"[\s\u3000，。、；：？！,.;:?!\"'“”‘’（）()\[\]【】《》<>—\-…·~`]+")


def norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    return PUNCT.sub("", s).strip().lower()


def phon(text):
    """把文本压成一条无声调的拼音串。

    汉字取拼音，字母数字原样保留——这样 ASR 输出 "Mina" 和词表里的
    「蜜娜」("mina") 能直接对上，不需要额外的语言判断。
    """
    from pypinyin import lazy_pinyin, Style
    out = []
    for ch in text or "":
        if "\u4e00" <= ch <= "\u9fff":
            py = lazy_pinyin(ch, style=Style.NORMAL, errors=lambda x: list(x))
            out.append(py[0] if py else ch)
        elif ch.isalnum():
            out.append(ch.lower())
    return "".join(out)


def lev(a, b, sub=1.0, ind=0.8):
    """加权编辑距离。插入删除比替换便宜，贴近'多听/漏听一个字'的情况。"""
    prev = [j * ind for j in range(len(b) + 1)]
    for i, ca in enumerate(a, 1):
        cur = [i * ind]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + ind,
                cur[j - 1] + ind,
                prev[j - 1] + (0.0 if ca == cb else sub),
            ))
        prev = cur
    return prev[-1]


def load_vocab():
    if not VOCAB_FILE.exists():
        raise SystemExit(f"找不到 {VOCAB_FILE}（一行一个词）")
    words = [l.strip() for l in
             VOCAB_FILE.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    return words, {w: phon(w) for w in words}


def align(text, vocab, cache, topk=3):
    """把任意文本吸附到词表里最近的一条，永远返回最像的那个。"""
    q = phon(text)
    if not q:
        return []
    scored = []
    for w in vocab:
        c = cache[w]
        scored.append((1.0 - lev(q, c) / max(len(q), len(c)), w))
    scored.sort(reverse=True)
    return scored[:topk]

# ---------------------------------------------------------------- 录音


class Recorder:
    def __init__(self):
        self.stream = None
        self.frames = []
        self.active = False

    def start(self):
        import sounddevice as sd
        self.frames = []
        self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="int16",
                                     callback=self._cb)
        self.stream.start()
        self.active = True

    def _cb(self, indata, frames, time_info, status):
        self.frames.append(indata.copy())

    def stop(self):
        import numpy as np
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.active = False
        if not self.frames:
            return np.zeros((0, 1), dtype="int16")
        return np.concatenate(self.frames, axis=0)

# ---------------------------------------------------------------- 识别


def trim_silence(audio, rel=0.02, pad_ms=120):
    """掐掉首尾静音，短词最需要这个。"""
    import numpy as np
    x = audio.reshape(-1)
    peak = float(np.abs(x).max()) if x.size else 0.0
    if peak <= 0:
        return audio
    idx = np.where(np.abs(x) > peak * rel)[0]
    if idx.size == 0:
        return audio
    pad = int(SR * pad_ms / 1000)
    return audio.reshape(-1, 1)[max(0, int(idx[0]) - pad):
                                min(len(x), int(idx[-1]) + pad)]


def audio_level(audio):
    import numpy as np
    x = audio.astype(np.float32).reshape(-1) / 32768.0
    if x.size == 0:
        return 0.0, 0.0
    return float(np.abs(x).max()), float(np.sqrt((x ** 2).mean()))


_models = {}


def _cuda_ready():
    """ctranslate2 能不能看到 CUDA 设备。"""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _build(size, threads, device):
    from faster_whisper import WhisperModel
    if device == "cuda":
        last = None
        # 显卡算力不同，float16 不一定支持，逐级降
        for ct in ("float16", "int8_float16", "float32"):
            try:
                return WhisperModel(size, device="cuda", compute_type=ct,
                                    num_workers=1), f"GPU / {ct}"
            except Exception as e:
                last = e
        raise RuntimeError(f"GPU 加载失败: {last}")

    kw = {"device": "cpu", "compute_type": "int8", "num_workers": 1}
    if threads:
        kw["cpu_threads"] = threads
    return WhisperModel(size, **kw), "CPU / int8"


def get_model(size, threads=0, device="auto"):
    import numpy as np
    key = (size, threads, device)
    if key in _models:
        return _models[key]

    print(f"加载模型 {size} ...")
    use = device
    if device == "auto":
        use = "cuda" if _cuda_ready() else "cpu"
        if use == "cpu":
            print("没检测到可用的 CUDA 设备，用 CPU")

    try:
        m, desc = _build(size, threads, use)
    except Exception as e:
        if use != "cuda":
            raise
        print(f"GPU 不可用：{str(e)[:160]}")
        print("常见原因：缺 cuDNN 9 或 cuBLAS 12，或显卡驱动太旧。退回 CPU。")
        m, desc = _build(size, threads, "cpu")

    try:
        # 分词器要等第一次 transcribe 才初始化，先用半秒静音跑一遍，顺便预热
        list(m.transcribe(np.zeros(SR // 2, dtype=np.float32),
                          language="zh", beam_size=1)[0])
    except Exception:
        pass

    _models[key] = m
    print(f"模型就绪（{desc}）。")
    return m


_latin_ids = {}


def _tokenizer(model):
    for name in ("tokenizer", "hf_tokenizer"):
        tok = getattr(model, name, None)
        if tok is not None and getattr(tok, "encoding", None) is not None:
            return tok
    return None


def latin_token_ids(model, limit=8000):
    """找出分词表里"纯拉丁字母/数字"的 token，全部屏蔽，逼模型只能吐汉字。"""
    key = id(model)
    if key in _latin_ids:
        return _latin_ids[key]

    ids = []
    tok = _tokenizer(model)
    if tok is None:
        print("[警告] 拿不到分词器，无法屏蔽英文输出")
        _latin_ids[key] = ids
        return ids

    enc = tok.encoding
    try:
        for i in range(enc.n_vocab):
            try:
                s = enc.decode_single_token_bytes(i).decode("utf-8", "ignore")
            except Exception:
                continue
            s = s.strip()
            if s and all(c.isascii() and (c.isalnum() or c in " '-.") for c in s):
                ids.append(i)
                if len(ids) >= limit:
                    break
    except Exception:
        ids = []

    _latin_ids[key] = ids
    if ids:
        print(f"已屏蔽 {len(ids)} 个拉丁 token（只输出汉字）")
    else:
        print("[警告] 一个拉丁 token 都没找到，英文输出不会被拦住")
    return ids


def _run_transcribe(model, x, beam, prompt, suppress):
    segments, _ = model.transcribe(
        x, language="zh", beam_size=beam, condition_on_previous_text=False,
        without_timestamps=True,
        # 三档温度兜底：0 度卡死时自动升温重来，这是 whisper 自带的防复读机制
        # 只保留两档：温度越高越"猜"，短词上容易越试越离谱，还慢好几倍
        temperature=[0.0, 0.2],
        # 这两个阈值就是"复读了就重来"的触发器，不能关
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=1.0,
        initial_prompt=prompt or None,
        suppress_tokens=suppress,
    )
    return "".join(s.text for s in segments).strip()


def collapse_repeats(s, max_period=4, min_reps=4):
    """把「宝宝宝宝宝宝…」这种复读压成一个。"""
    out = s
    for p in range(1, max_period + 1):
        res, i = [], 0
        while i < len(out):
            unit = out[i:i + p]
            k, j = 1, i + p
            while out[j:j + p] == unit:
                k += 1
                j += p
            if k >= min_reps:
                res.append(unit)
                i = j
            else:
                res.append(out[i])
                i += 1
        out = "".join(res)
    return out


def build_prompt(words, max_chars=90):
    """用词表里的短词做提示，把解码往"游戏名词"上带。"""
    shorts = [w for w in words if len(w) <= 3]
    picked, n = [], 0
    for w in shorts:
        if n + len(w) + 1 > max_chars:
            break
        picked.append(w)
        n += len(w) + 1
    return "以下是最终幻想14游戏里的名词：" + "，".join(picked)


def transcribe(audio, size="base", beam=1, threads=0, prompt=None,
               no_latin=False, short_beam=3, device="auto"):
    """短词的坑都在这里处理：静音占比高、声音小、复读、无语音误判。"""
    import numpy as np
    model = get_model(size, threads, device)
    if audio.size < SR * 0.2:
        return ""

    x = trim_silence(audio).astype(np.float32).reshape(-1) / 32768.0
    peak = float(np.abs(x).max())
    if 1e-6 < peak < 0.5:                 # 太小声，放大
        x = np.clip(x * min(8.0, 0.9 / peak), -1.0, 1.0)
    elif peak > 0.95:                     # 太大声，压一点
        x = x * (0.9 / peak)

    # 短语音上 beam=1 太容易错，加宽搜索；这会让耗时涨 3~5 倍
    if len(x) / SR < 1.0:
        beam = max(beam, short_beam)

    if no_latin:
        ids = latin_token_ids(model)
        if ids:
            try:
                text = _run_transcribe(model, x, beam, prompt, ids)
                text = collapse_repeats(text)
                if text:
                    return text
                print("  （屏蔽字母后没输出，退回普通模式重试）")
            except Exception:
                pass

    return collapse_repeats(_run_transcribe(model, x, beam, prompt, None))

def transcribe_timeout(audio, timeout, **kw):
    """把识别丢到后台线程，超过 timeout 秒就放弃。

    超时返回 None。线程无法强杀，但温度兜底只剩两档，它很快会自己结束。
    """
    box = {}

    def run():
        try:
            box["text"] = transcribe(audio, **kw)
        except Exception as e:      # noqa: BLE001
            box["err"] = e

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return None
    if "err" in box:
        raise box["err"]
    return box.get("text", "")

# ---------------------------------------------------------------- Win32

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

KEYEVENTF_KEYUP = 0x0002
VK_RETURN, VK_CONTROL, VK_MENU, VK_SHIFT = 0x0D, 0x11, 0x12, 0x10
WM_CHAR, WM_KEYDOWN, WM_KEYUP = 0x0102, 0x0100, 0x0101
WM_HOTKEY = 0x0312
MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = (
    0x0001, 0x0002, 0x0004, 0x0008, 0x4000)


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ctypes.c_void_p)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD),
                ("dwFlags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", ctypes.c_void_p)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wt.DWORD), ("wParamL", wt.WORD), ("wParamH", wt.WORD)]


class _U(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wt.DWORD), ("u", _U)]


def _init():
    u, k = _user32, _kernel32
    u.SendInput.argtypes = [wt.UINT, ctypes.c_void_p, ctypes.c_int]
    u.SendInput.restype = wt.UINT
    u.RegisterHotKey.argtypes = [wt.HWND, ctypes.c_int, wt.UINT, wt.UINT]
    u.RegisterHotKey.restype = wt.BOOL
    u.UnregisterHotKey.argtypes = [wt.HWND, ctypes.c_int]
    u.GetAsyncKeyState.argtypes = [ctypes.c_int]
    u.GetAsyncKeyState.restype = ctypes.c_short
    u.GetForegroundWindow.restype = wt.HWND
    u.SetForegroundWindow.argtypes = [wt.HWND]
    u.AttachThreadInput.argtypes = [wt.DWORD, wt.DWORD, wt.BOOL]
    u.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.c_void_p]
    u.GetWindowThreadProcessId.restype = wt.DWORD
    u.EnumWindows.argtypes = [ctypes.c_void_p, wt.LPARAM]
    u.GetWindowTextLengthW.argtypes = [wt.HWND]
    u.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
    u.IsWindowVisible.argtypes = [wt.HWND]
    u.ShowWindow.argtypes = [wt.HWND, ctypes.c_int]
    u.BringWindowToTop.argtypes = [wt.HWND]
    u.PostMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
    k.GetCurrentThreadId.restype = wt.DWORD
    k.GetConsoleWindow.restype = wt.HWND

    size = ctypes.sizeof(INPUT)
    if size != 40:      # 64 位必须是 40，否则 SendInput 静默失败
        raise RuntimeError(f"INPUT 结构体 {size} 字节，应为 40")


_init()


def _key(vk, up=False):
    scan = 0 if up else _user32.MapVirtualKeyW(vk, 0)
    inp = INPUT(type=1)
    inp.ki = KEYBDINPUT(vk, scan, KEYEVENTF_KEYUP if up else 0, 0, None)
    return _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) == 1


def _is_down(vk):
    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)


def find_window(keyword):
    found = []
    kw = keyword.lower()

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        if not _user32.IsWindowVisible(hwnd):
            return True
        n = _user32.GetWindowTextLengthW(hwnd)
        if n:
            buf = ctypes.create_unicode_buffer(n + 1)
            _user32.GetWindowTextW(hwnd, buf, n + 1)
            if kw in buf.value.lower():
                found.append((hwnd, buf.value))
        return True

    _user32.EnumWindows(cb, 0)
    return found


def activate(hwnd):
    """拉到前台。后台进程调 SetForegroundWindow 有限制，需要临时附加输入线程。"""
    fg = _user32.GetForegroundWindow()
    if fg == hwnd:
        return True
    tid = _user32.GetWindowThreadProcessId(fg, None)
    my = _kernel32.GetCurrentThreadId()
    attached = bool(tid and tid != my and
                    _user32.AttachThreadInput(my, tid, True))
    try:
        _user32.BringWindowToTop(hwnd)
        _user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            _user32.AttachThreadInput(my, tid, False)
    time.sleep(0.15)
    return _user32.GetForegroundWindow() == hwnd


def post_text(hwnd, text, interval=0.02):
    """逐字投 WM_CHAR。不走剪贴板、不走键盘、不过输入法。

    interval=0 时全部一次投完（消息自动排队），看起来是瞬发；
    设成 0.03~0.2 会逐字出，更像真人打字。
    """
    n = 0
    for ch in text:
        if _user32.PostMessageW(hwnd, WM_CHAR, ord(ch), 1):
            n += 1
        if interval > 0:
            time.sleep(interval)
    return n


def submit(hwnd):
    """回车提交。优先真按键，切不到前台就退而投消息。"""
    activate(hwnd)
    if _user32.GetForegroundWindow() == hwnd:
        time.sleep(0.1)
        _key(VK_RETURN)
        _key(VK_RETURN, up=True)
    else:
        _user32.PostMessageW(hwnd, WM_KEYDOWN, VK_RETURN, 0)
        _user32.PostMessageW(hwnd, WM_KEYUP, VK_RETURN, 0)


def deliver(text, target="最终幻想", send=True, interval=0.02):
    wins = find_window(target)
    if not wins:
        print(f"[错误] 找不到标题含「{target}」的窗口")
        return False
    hwnd, title = wins[0]
    n = post_text(hwnd, text, interval)
    print(f"已投递 {n}/{len(text)} 字 -> {title}")
    if send:
        time.sleep(0.15)
        submit(hwnd)
        print("已回车提交")
    return True

# ---------------------------------------------------------------- 热键

CANDIDATES = ["F12", "F11", "F10", "F9", "F8", "F7", "F6",
              "Ctrl+Alt+Q", "Ctrl+Alt+X", "Ctrl+Alt+Z", "Ctrl+Alt+W"]
_FKEYS = {f"F{i}": 0x6F + i for i in range(1, 25)}
_MODS = {"CTRL": (VK_CONTROL, MOD_CONTROL), "ALT": (VK_MENU, MOD_ALT),
         "SHIFT": (VK_SHIFT, MOD_SHIFT), "WIN": (0x5B, MOD_WIN)}


def parse_hotkey(spec):
    parts = [p.strip().upper() for p in spec.split("+") if p.strip()]
    if not parts:
        raise ValueError("空热键")
    *mods, key = parts
    if key in _FKEYS:
        vk = _FKEYS[key]
    elif len(key) == 1 and key.isalnum():
        vk = ord(key)
    else:
        raise ValueError(f"看不懂的键名: {key}")
    bits = MOD_NOREPEAT
    for m in mods:
        if m not in _MODS:
            raise ValueError(f"看不懂的修饰键: {m}")
        bits |= _MODS[m][1]
    return bits, vk


def probe(spec, hid=900):
    try:
        mods, vk = parse_hotkey(spec)
    except ValueError:
        return False
    if _user32.RegisterHotKey(None, hid, mods, vk):
        _user32.UnregisterHotKey(None, hid)
        return True
    return False


def keyscan():
    free = [s for s in CANDIDATES if probe(s)]
    print("热键占用情况：")
    for s in CANDIDATES:
        print(f"  {s:<16}{'可用' if s in free else '被占用'}")
    if free:
        print(f"\n推荐: {free[0]}")
    else:
        print("\n候选全被占用，用 -k Ctrl+Alt+某个字母 自己挑一个")


def resolve_key(spec):
    if spec.lower() != "auto":
        if not probe(spec):
            raise SystemExit(f"热键 {spec} 被占用，换一个或直接用默认 auto")
        return parse_hotkey(spec)

    if HOTKEY_FILE.exists():
        saved = HOTKEY_FILE.read_text(encoding="utf-8").strip()
        if saved and probe(saved):
            print(f"沿用上次的键位: {saved}")
            return parse_hotkey(saved)

    for s in CANDIDATES:
        if probe(s):
            HOTKEY_FILE.write_text(s + "\n", encoding="utf-8")
            print(f"已选用 {s}（记在 hotkey.txt）")
            return parse_hotkey(s)
    raise SystemExit("候选热键全被占用，用 -k 指定一个")

# ---------------------------------------------------------------- 主流程


def handle(args, vocab, cache, rec):
    """一次完整的 录音 -> 识别 -> 吸附 -> 投递。"""
    try:
        rec.start()
    except Exception as e:
        print(f"[错误] 打开麦克风失败: {type(e).__name__}: {e}")
        return
    print("● 录音中...")
    deadline = time.time() + args.max_sec
    while time.time() < deadline and _is_down(args.vk):
        time.sleep(0.01)
    audio = rec.stop()

    if audio.size < SR * 0.15:
        print("[跳过] 太短")
        return
    peak, rms = audio_level(audio)
    loud = "偏小" if peak < 0.05 else ("偏大" if peak > 0.95 else "正常")
    print(f"○ {audio.size / SR:.1f}s  峰值 {peak:.3f} RMS {rms:.3f}（{loud}）")

    t0 = time.time()
    try:
        text = transcribe_timeout(
            audio, args.timeout,
            size=args.model, beam=args.beam, threads=args.threads,
            prompt=args.prompt, no_latin=not args.latin,
            short_beam=args.short_beam, device=args.device)
    except Exception as e:
        print(f"[错误] 识别失败: {type(e).__name__}: {e}")
        return
    dt = time.time() - t0
    if text is None:
        print(f"  ⏱ {dt:.1f}s 没出结果，放弃这一条")
        return
    print(f"  识别 {dt:.2f}s")

    if not text:
        print("[跳过] 没识别出内容")
        return

    res = align(text, vocab, cache)
    if not res:
        print(f"[跳过] {text!r} 里没有可比较的字符")
        return
    score, word = res[0]
    print(f"  原文 {text!r} -> {word}"
          f"   [{'  '.join(f'{w}({s:.2f})' for s, w in res)}]")
    if score < 0.45:
        print(f"  ⚠ 置信度偏低 {score:.2f}，照发")

    deliver(word, args.target, not args.no_send, args.interval)


def serve(args):
    vocab, cache = load_vocab()
    print(f"词表 {len(vocab)} 条")
    if not args.model:
        args.model = "small"
    if args.prompt == "auto":
        args.prompt = build_prompt(vocab)
        print(f"提示词: {args.prompt[:40]}...")

    try:
        import sounddevice  # noqa: F401
        import faster_whisper  # noqa: F401
        import pypinyin  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"缺依赖: {e.name}\npip install sounddevice "
                         f"faster-whisper pypinyin")

    mods, vk = resolve_key(args.key)
    args.vk = vk
    label = args.key if args.key.lower() != "auto" else "热键"

    rec = Recorder()
    lock = threading.Lock()
    print(f"按住 {label} 说话，松开自动上屏。Ctrl+C 退出。")

    if not _user32.RegisterHotKey(None, 1, mods, vk):
        raise SystemExit(f"{label} 注册失败，换一个：-k F9 或 -k Ctrl+Alt+Q")

    m = get_model(args.model, args.threads, args.device)   # 启动就加载好
    if not args.latin:
        print("⚠ 已开启拉丁屏蔽：识别质量会明显下降，仅用于实验")
        latin_token_ids(m)
    print(f"识别超时 {args.timeout:.1f}s · 长句 beam {args.beam} · "
          f"短词 beam {args.short_beam} · 投递间隔 {args.interval:.2f}s")
    print("准备就绪。")

    msg = wt.MSG()
    try:
        while True:
            if _user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
                if msg.message == WM_HOTKEY:
                    if lock.locked():
                        continue
                    threading.Thread(
                        target=lambda: (lock.acquire(),
                                        handle(args, vocab, cache, rec),
                                        lock.release()),
                        daemon=True).start()
            else:
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n已退出。")
    finally:
        _user32.UnregisterHotKey(None, 1)


def main():
    ap = argparse.ArgumentParser(description="FF14 语音输入")
    ap.add_argument("cmd", nargs="?", default="serve",
                    choices=["serve", "keyscan"])
    ap.add_argument("-k", "--key", default="auto", help="热键，默认自动挑")
    ap.add_argument("--model", default="small",
                    help="tiny / base / small（默认）/ medium")
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--short-beam", type=int, default=3,
                    help="1 秒以内录音用的 beam，调低更快但更不准（默认 3）")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="auto=有显卡就用（默认）")
    ap.add_argument("-t", "--target", default="最终幻想", help="窗口标题关键字")
    ap.add_argument("--no-send", action="store_true", help="只填字不回车")
    ap.add_argument("--max-sec", type=float, default=15.0)
    ap.add_argument("--timeout", type=float, default=1.5,
                    help="识别超时秒数，超了直接放弃（默认 1.5）")
    ap.add_argument("--interval", type=float, default=0.02,
                    help="逐字投递间隔（秒）。默认 0.02，0=瞬发，越大越慢越拟人")
    ap.add_argument("--prompt", default="auto",
                    help="识别提示词，auto=用词表里的短词自动生成")
    ap.add_argument("--latin", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="允许模型输出英文（默认）。--no-latin 会强制中文，"
                         "但会严重破坏识别质量，仅供实验")
    args = ap.parse_args()

    if args.cmd == "keyscan":
        keyscan()
    else:
        serve(args)


if __name__ == "__main__":
    main()
