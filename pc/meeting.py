# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "soundcard>=0.4.3",
#   "numpy",
#   "sherpa-onnx>=1.10",
#   "opencc-python-reimplemented",
#   "python-docx",
#   "transformers>=4.40,<5",
#   "sentencepiece",
#   "torch",
#   "av",
#   "reportlab",
#   "pypac",
#   "truststore",
# ]
# ///
"""
会議字幕(お客様向け・Gemini 優先、ChatGPT にも貼り付け可)

start.bat をダブルクリックすると http://localhost:8765 が開きます。
- 「オンライン会議」または「対面」を選んで開始
- 日本語・中国語・英語を自動判別してリアルタイム字幕(中国語・英語は日本語訳つき)
- 終了後 Gemini(無料の API キー)が自動で議事録に。キーがなければ ChatGPT にコピー&貼り付け
- Word / PDF 出力
"""

import argparse
import ctypes
import datetime as dt
import io
import json
import os
import queue
import re
import sys
import shutil
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
REC_DIR = BASE / "recordings"
BUILD = "2026-11-26e"  # 用來確認電腦上跑的是哪一版(一鍵更新也用這個比對)
# 一鍵更新:新版放在這裡(Josh 合併 PR 才會發布)
UPDATE_BASE = "https://raw.githubusercontent.com/lee09930239-dotcom/kaigi-mobile/main/pc/"
WRITE_LOG = BASE / "meta_writes.log"
GH = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SV_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
SV_URL = f"{GH}/asr-models/{SV_NAME}.tar.bz2"
VAD_URL = f"{GH}/asr-models/silero_vad.onnx"
SPK_NAME = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
SPK_URL = f"{GH}/speaker-recongition-models/{SPK_NAME}"
JA_NAME = "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8"
JA_URL = f"{GH}/asr-models/{JA_NAME}.tar.bz2"
MT_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_LANG = {"en": "eng_Latn", "zh": "zho_Hant", "yue": "zho_Hant"}  # 中国語・英語 → 日本語(日本語は訳さない)
LANG_LABEL = {"ja": "日", "zh": "中", "yue": "広", "en": "英", "ko": "韓"}

QWEN_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
QWEN_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
LLAMA_PORT = 8767  # 社内用ツール(8766)とぶつからないように
LLAMA_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
LLAMA_PINNED = ("b6500", "b6000", "b5500")
QWEN_IDLE_SLEEP = 1200  # 会議が終わって 20 分使わなければ止める(メモリと CPU を空ける)

CAPTURE_RATE = 48000
ASR_RATE = 16000
CHUNK = 4800  # 0.1 秒 @ 48kHz
SPK_THRESHOLD = 0.5  # 聲紋相似度高於此值視為同一人
SPK_NEW = 0.35  # 低於此值且夠長才當成新的說話者
MAX_SEG = 8  # 每段最長秒數,日文模型切短一點比較準


# ---------- 模型下載 ----------

def download(url, dest):
    print(f"ダウンロード中 {url.rsplit('/', 1)[-1]} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done * 100 // total}%", end="", flush=True)
    print()
    if total and done != total:  # 途中で切れた
        tmp.unlink(missing_ok=True)
        raise RuntimeError("ダウンロードが途中で切れました。もう一度 start_*.bat を起動してください")
    tmp.replace(dest)


def ensure_tar(name, url):
    d = MODEL_DIR / name
    if not (d / "model.int8.onnx").exists():
        if d.exists():  # 前回の展開が途中で止まった
            shutil.rmtree(d, ignore_errors=True)
        tar = MODEL_DIR / f"{name}.tar.bz2"
        if not tar.exists():
            download(url, tar)
        tmpdir = MODEL_DIR / f"_extract_{name}"
        shutil.rmtree(tmpdir, ignore_errors=True)
        try:
            with tarfile.open(tar) as t:
                try:
                    t.extractall(tmpdir, filter="data")
                except TypeError:
                    t.extractall(tmpdir)
        except Exception:
            tar.unlink(missing_ok=True)  # 壊れた圧縮ファイルは消して次回ダウンロードし直す
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise
        (tmpdir / name).replace(d)  # 展開が終わってから本来の場所へ移す
        shutil.rmtree(tmpdir, ignore_errors=True)
        (d / ".complete").write_text("ok")
        tar.unlink()
    return d


def repair_models():
    """モデルが壊れているとき:展開済みの音声認識モデルを消して取り直せるようにする"""
    for name in (SV_NAME, JA_NAME):
        d = MODEL_DIR / name
        if d.exists() and not (d / ".complete").exists():
            shutil.rmtree(d, ignore_errors=True)
    for f in ("silero_vad.onnx", SPK_NAME):
        p = MODEL_DIR / f
        if p.exists() and p.stat().st_size < 100_000:
            p.unlink()


def ensure_models():
    MODEL_DIR.mkdir(exist_ok=True)
    sv = ensure_tar(SV_NAME, SV_URL)
    ja = ensure_tar(JA_NAME, JA_URL)
    vad = MODEL_DIR / "silero_vad.onnx"
    if not vad.exists():
        download(VAD_URL, vad)
    spk = MODEL_DIR / SPK_NAME
    if not spk.exists():
        download(SPK_URL, spk)
    return sv, ja, vad, spk


# ---------- 收音 ----------

def com_init():
    # soundcard 在子執行緒使用 WASAPI 前需要初始化 COM
    if sys.platform == "win32":
        try:
            ctypes.windll.ole32.CoInitializeEx(None, 0)
        except Exception:
            pass


def list_devices(sc):
    return {
        "mics": [m.name for m in sc.all_microphones()],
        "speakers": [m.name for m in sc.all_microphones(include_loopback=True) if m.isloopback],
        "default_mic": sc.default_microphone().name,
        "default_speaker": sc.default_speaker().name,
    }


def get_device(sc, name, loopback):
    for m in sc.all_microphones(include_loopback=loopback):
        if m.name == name and (m.isloopback == loopback):
            return m
    if loopback:
        return sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
    return sc.default_microphone()


def screenshot_loop(folder, stop, t_start, every=2.0, paused=None):
    """開會時每 2 秒看一次畫面,畫面變了而且停下來(不是動畫中)才存一張 JPEG。
    shots/index.jsonl 記錄每張圖是會議開始後第幾秒,整理時可以跟字幕對起來"""
    import mss
    from PIL import Image

    folder.mkdir(exist_ok=True)
    index = folder / "index.jsonl"
    last_saved = prev = None
    n = 0
    with mss.mss() as sct:
        mon = sct.monitors[0]  # 全部螢幕(Teams 在副螢幕也拍得到)
        while not stop.wait(every):
            if paused is not None and paused.is_set():
                continue
            try:
                raw = sct.grab(mon)
                img = Image.frombytes("RGB", raw.size, raw.rgb)
                thumb = np.asarray(img.convert("L").resize((96, 54)), np.float32)
            except Exception:
                continue
            stable = prev is not None and np.abs(thumb - prev).mean() < 2
            changed = last_saved is None or np.abs(thumb - last_saved).mean() > 4
            prev = thumb
            if stable and changed:
                n += 1
                t = time.time() - t_start
                name = f"{n:04d}_{int(t // 60):02d}m{int(t % 60):02d}s.jpg"
                if img.width > 1920:
                    img = img.resize((1920, round(img.height * 1920 / img.width)))
                img.save(folder / name, quality=70)
                with index.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": round(t, 1), "file": name}) + "\n")
                last_saved = thumb


def find_playing(devices, skip_name):
    """其他輸出裝置各錄 0.4 秒,回傳正在出聲的那一個(Teams 用的是別的耳機/喇叭時)"""
    for d in devices:
        if d.name == skip_name:
            continue
        try:
            with d.recorder(samplerate=CAPTURE_RATE, channels=1, blocksize=CHUNK) as r:
                y = r.record(numframes=int(CAPTURE_RATE * 0.4))
            if len(y) and float(np.abs(y).max()) > 1e-3:
                return d
        except Exception:
            continue
    return None


def capture(resolve, track, out_q, wav_path, stop, alternatives=None, paused=None):
    """resolve(fallback) 回傳要錄的裝置。拔耳機等裝置變動時改抓 Windows 目前的預設裝置繼續錄。
    alternatives:對方聲音(loopback)用。一直完全無聲時,檢查其他耳機/喇叭,有在出聲的就自動換過去"""
    com_init()
    override = None
    wf = wave.open(str(wav_path), "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(ASR_RATE)
    t0 = time.time()
    written = 0
    failed = False
    try:
        while not stop.is_set():
            try:
                device = override or resolve(failed)
                with device.recorder(samplerate=CAPTURE_RATE, channels=1, blocksize=CHUNK) as rec:
                    if failed:
                        out_q.put((track, f"音訊裝置有變動,已改用「{device.name}」繼續錄音"))
                        failed = False
                    silent_since = time.time()
                    while not stop.is_set():
                        data = rec.record(numframes=CHUNK)
                        x = data[:, 0] if data.ndim > 1 else data
                        if alternatives:
                            if len(x) and float(np.abs(x).max()) > 1e-4:
                                silent_since = time.time()
                            elif time.time() - silent_since > 15:
                                silent_since = time.time()
                                found = find_playing(alternatives(), device.name)
                                if found:
                                    override = found
                                    out_q.put((track, f"「{device.name}」沒有聲音,對方聲音改從「{found.name}」收"))
                                    break
                        n = len(x) // 3 * 3
                        x = x[:n].reshape(-1, 3).mean(axis=1).astype(np.float32)  # 48k → 16k
                        # 沒有聲音在播放時 loopback 不會送資料(或裝置切換中),補上靜音讓音軌時間對齊
                        gap = int((time.time() - t0) * ASR_RATE) - written - len(x)
                        if gap > ASR_RATE // 2:
                            x = np.concatenate([np.zeros(gap, np.float32), x])
                        if paused is not None and paused.is_set():
                            x = np.zeros_like(x)  # 暫停中:錄成無聲(時間軸不斷),也不會辨識
                        written += len(x)
                        wf.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
                        out_q.put((track, x))
            except Exception as e:
                if not failed:
                    out_q.put((track, e))
                    failed = True
                time.sleep(1)
    finally:
        wf.close()


# ---------- 辨識、說話者、翻譯 ----------

class Engine:
    def __init__(self, sv_dir, ja_dir, vad_path, spk_path, threads):
        import opencc
        import sherpa_onnx

        self.so = sherpa_onnx
        self.rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(sv_dir / "model.int8.onnx"),
            tokens=str(sv_dir / "tokens.txt"),
            num_threads=threads,
            use_itn=True,
            language="auto",
        )
        # 日文用 ReazonSpeech 訓練的日文專用模型,口語和遠距收音都準很多
        self.ja = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(ja_dir / "model.int8.onnx"),
            tokens=str(ja_dir / "tokens.txt"),
            num_threads=threads,
        )
        self.spk = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(spk_path), num_threads=1)
        )
        self.vad_cfg = sherpa_onnx.VadModelConfig()
        self.vad_cfg.silero_vad.model = str(vad_path)
        self.vad_cfg.silero_vad.min_silence_duration = 0.5  # 太短會把一句話切碎
        self.vad_cfg.silero_vad.min_speech_duration = 0.25
        self.vad_cfg.silero_vad.max_speech_duration = MAX_SEG
        self.vad_cfg.silero_vad.threshold = 0.4
        self.vad_cfg.sample_rate = ASR_RATE
        self.window = self.vad_cfg.silero_vad.window_size
        self.s2t = opencc.OpenCC("s2twp")

    def new_vad(self):
        return self.so.VoiceActivityDetector(self.vad_cfg, buffer_size_in_seconds=60)

    def transcribe(self, samples):
        # SenseVoice 判斷語言;日文再交給日文專用模型
        st = self.rec.create_stream()
        st.accept_waveform(ASR_RATE, samples)
        self.rec.decode_stream(st)
        text = st.result.text.strip()
        lang = st.result.lang.strip("<|>") or "?"
        if lang == "ja":
            st = self.ja.create_stream()
            st.accept_waveform(ASR_RATE, samples)
            self.ja.decode_stream(st)
            text = st.result.text.strip() or text
        if lang in ("zh", "yue"):
            text = self.s2t.convert(text)
        if lang == "ja":
            # 日文模型把縮寫輸出成小寫(pou、rfi、nda),改回大寫
            text = re.sub(r"(?<![A-Za-z])[a-z]{1,3}(?![A-Za-z])", lambda m: m.group().upper(), text)
        return lang, apply_glossary(text)

    def embed(self, samples):
        st = self.spk.create_stream()
        st.accept_waveform(ASR_RATE, samples)
        st.input_finished()
        e = np.array(self.spk.compute(st), dtype=np.float32)
        return e / (np.linalg.norm(e) + 1e-9)


class SpeakerTracker:
    """同一條音軌內依聲紋分出不同說話者"""

    def __init__(self, prefix, max_speakers=None):
        self.prefix = prefix
        self.max = max_speakers or 8
        self.centroids = []
        self.last = None

    def assign(self, emb, seconds):
        if self.centroids:
            sims = [float(c @ emb) for c in self.centroids]
            best = int(np.argmax(sims))
            # 短片段聲紋不可靠:只有夠長、又跟每個人都不像,才新增說話者
            new = seconds >= 2.0 and sims[best] < SPK_NEW and len(self.centroids) < self.max
            if not new:
                c = self.centroids[best] * 0.9 + emb * 0.1
                self.centroids[best] = c / np.linalg.norm(c)
                self.last = best
                return f"{self.prefix}{best + 1}"
        self.centroids.append(emb)
        self.last = len(self.centroids) - 1
        return f"{self.prefix}{self.last + 1}"


def recluster(embs, secs, n=None, merge_sim=0.7):
    """整場一起分說話者:先細分成很多小群,再把最像的群合併到 n 人
    (不知道人數時合到彼此不像為止),最後把每一段重新分給最像的人。回傳 0 起算的編號"""
    E = np.asarray(embs, np.float32)
    w = np.asarray(secs, np.float32)
    if len(E) < 2:
        return [0] * len(E)
    long_ = w >= 1.5  # 短片段聲紋不可靠,只用來分配、不用來決定中心
    if long_.sum() < max(2, n or 2):
        long_ = np.ones(len(E), bool)
    idx = np.flatnonzero(long_)
    cents, members = [], []
    for i in idx:
        if cents:
            sims = np.array([c @ E[i] for c in cents])
            b = int(sims.argmax())
            if sims[b] >= 0.6:
                members[b].append(i)
                c = (E[members[b]] * w[members[b], None]).sum(0)
                cents[b] = c / np.linalg.norm(c)
                continue
        cents.append(E[i].copy())
        members.append([i])
    total = float(w[idx].sum())
    while len(cents) > 1:
        C = np.array(cents)
        S = C @ C.T
        np.fill_diagonal(S, -9)
        wt = np.array([w[m].sum() for m in members])
        small = wt < total * 0.01  # 不到全場 1% 的小群(雜音、咳嗽)先併掉
        if n and len(cents) <= n and not small.any():
            break
        a, b = np.unravel_index(S.argmax(), S.shape)
        if small.any():
            k = int(np.flatnonzero(small)[np.argmin(wt[small])])
            if n and len(cents) <= n and S[k].max() < 0.3:
                break  # 人數已經夠少,這個小群跟誰都不像,當成講話很少的人
            a, b = k, int(S[k].argmax())
        elif not n and S[a, b] < merge_sim:
            break
        a, b = min(a, b), max(a, b)
        members[a] += members[b]
        c = (E[members[a]] * w[members[a], None]).sum(0)
        cents[a] = c / np.linalg.norm(c)
        del cents[b], members[b]
    C = np.array(cents)
    for _ in range(3):
        lab = (E @ C.T).argmax(1)
        for k in range(len(C)):
            sel = (lab == k) & long_
            if sel.any():
                c = (E[sel] * w[sel, None]).sum(0)
                C[k] = c / np.linalg.norm(c)
    lab = (E @ C.T).argmax(1)
    order = {}
    for x in lab:
        order.setdefault(int(x), len(order))
    return [order[int(x)] for x in lab]


class Translator:
    def __init__(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        torch.set_num_threads(2)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MT_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL).eval()
        # 先翻成簡體再轉繁體:NLLB 的簡體中文品質比繁體好很多
        self.zh = self.tok.convert_tokens_to_ids("zho_Hans")
        self.ja = self.tok.convert_tokens_to_ids("jpn_Jpan")
        import opencc

        self.s2t = opencc.OpenCC("s2twp")

    def __call__(self, text, lang):
        to_ja = True
        core = re.sub(r"[\s,，。.!?！？]|對|是|好|嗯|喔|那個|就是", "", text)
        if len(core) < 4:
            return ""  # 附和、太短的句子不翻,避免亂翻
        self.tok.src_lang = NLLB_LANG[lang]
        inp = self.tok(text, return_tensors="pt")
        with self.torch.inference_mode():
            out = self.model.generate(**inp, forced_bos_token_id=self.ja if to_ja else self.zh, max_new_tokens=160,
                                      num_beams=1, no_repeat_ngram_size=3)
        zh = self.tok.decode(out[0], skip_special_tokens=True)
        if not to_ja:
            zh = self.s2t.convert(zh)
        zh = re.sub(r"([,，、.。])[,，、.。\s]*", r"\1", zh).strip(" ,，")
        parts = re.split(r"(?<=[,，。])", zh)
        dedup = [x for i, x in enumerate(parts)
                 if x.strip(" ,，。") and (i == 0 or x.strip(" ,，。") != parts[i - 1].strip(" ,，。"))]
        return "".join(dedup)


# ---------- 會議 ----------

class Hub:
    def __init__(self):
        self.lock = threading.Lock()
        self.clients = []

    def publish(self, event):
        with self.lock:
            for q in self.clients:
                q.put(event)

    def subscribe(self):
        q = queue.Queue()
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.clients.remove(q)


def safe_name(s):
    s = re.sub(r'[\\/:*?"<>|\r\n]+', "_", s).strip(" .")
    return s[:60] or "會議"


class Meeting:
    def __init__(self, folder):
        self.dir = folder
        self.meta_path = folder / "meta.json"
        self.lines_path = folder / "transcript.jsonl"
        self.lock = threading.Lock()
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8")) if self.meta_path.exists() else {}
        self._mt = self._mtime()
        self._base = json.loads(json.dumps(self.meta))  # 上次讀取/儲存時的內容,用來判斷自己改了哪些欄位
        self.lines = {}
        if self.lines_path.exists():
            for raw in self.lines_path.read_text(encoding="utf-8").splitlines():
                if raw.strip():
                    r = json.loads(raw)
                    self.lines.setdefault(r["id"], {}).update(r)

    @property
    def id(self):
        return self.dir.name

    def _mtime(self):
        try:
            return self.meta_path.stat().st_mtime_ns
        except OSError:
            return None

    def refresh(self):
        """meta.json 被外部改過(例如 Claude 直接寫入議事錄)就重新讀取"""
        with self.lock:
            if self._mtime() != self._mt:
                try:
                    self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                    self._base = json.loads(json.dumps(self.meta))
                    self._mt = self._mtime()
                except (OSError, ValueError):
                    pass
        return self

    def save_meta(self):
        with self.lock:
            # meta.json 可能被 Claude 從外面改過:以檔案為準,只覆蓋這邊真的改過的欄位
            if self._mtime() not in (None, self._mt):
                try:
                    disk = json.loads(self.meta_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    disk = None
                if isinstance(disk, dict):
                    for k in set(self._base) | set(self.meta):
                        if k not in self.meta:
                            disk.pop(k, None)
                        elif self.meta[k] != self._base.get(k):
                            disk[k] = self.meta[k]
                    self.meta = disk
            changed = sorted(k for k in set(self._base) | set(self.meta) if self.meta.get(k) != self._base.get(k))
            tmp = self.meta_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.meta, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.meta_path)
            self._mt = self._mtime()
            self._base = json.loads(json.dumps(self.meta))
            try:  # 診斷用:誰在什麼時候改了 meta.json
                with WRITE_LOG.open("a", encoding="utf-8") as f:
                    f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} pid={os.getpid()} build={BUILD} {self.id} changed={','.join(changed)}\n")
            except OSError:
                pass

    def update_line(self, r):
        with self.lock:
            self.lines.setdefault(r["id"], {}).update(r)
            with open(self.lines_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def sorted_lines(self):
        return sorted(self.lines.values(), key=lambda r: (r.get("offset", 0), r["id"]))

    def speaker_name(self, sid):
        return self.meta.get("speakers", {}).get(sid) or default_speaker_name(sid)

    def summary(self):
        return {
            "id": self.id,
            "name": self.meta.get("name", ""),
            "date": self.meta.get("start", "")[:16].replace("T", " "),
            "minutes": round(self.meta.get("duration", 0) / 60),
            "lines": len(self.lines),
            "shots": len(list((self.dir / "shots").glob("*.jpg"))) if (self.dir / "shots").exists() else 0,
        }


def default_speaker_name(sid):
    if sid == "me":
        return "自分"
    m = re.match(r"(them|p)(\d+)$", sid)
    if m:
        return ("相手" if m[1] == "them" else "話者") + m[2]
    return sid


class QwenTranslator:
    """このパソコンだけで要点をまとめる AI(Qwen)。llama-server を低優先度で動かす"""

    def __init__(self, threads=4):
        import opencc

        self.s2t = opencc.OpenCC("s2twp")
        self.url = f"http://127.0.0.1:{LLAMA_PORT}"
        self.proc = None
        self.lat = 0.0  # 最近的翻譯秒數(平滑)
        if not self.healthy():
            exe = self.ensure_server()
            model = self.ensure_model()
            log = open(MODEL_DIR / "llama.log", "ab")
            # 低優先權:電腦忙的時候 Windows 會先讓字幕(辨識)跑,Qwen 只用剩下的 CPU
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
            self.proc = subprocess.Popen([str(exe), "-m", str(model), "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
                                          "-t", str(threads), "-c", "8192"],
                                         stdout=log, stderr=log, creationflags=flags)
            atexit.register(self.close)
            for _ in range(240):
                if self.proc.poll() is not None:
                    raise RuntimeError("llama-server 啟動失敗,詳情見 models/llama.log")
                if self.healthy():
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError("llama-server 2 分鐘內沒有啟動")

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def healthy(self):
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    @staticmethod
    def ensure_server():
        d = MODEL_DIR / "llama"
        found = list(d.rglob("llama-server.exe")) if d.exists() else []
        if found:
            return found[0]
        # 找最近幾個版本裡「Windows、CPU、x64」的 zip(新舊版命名不同:cpu-x64 / avx2-x64)
        seen, cands = [], []
        try:
            req = urllib.request.Request(LLAMA_API.replace("/latest", "?per_page=15"),
                                         headers={"User-Agent": "kaigi-jimaku"})
            with urllib.request.urlopen(req, timeout=30) as r:
                rels = json.loads(r.read().decode("utf-8"))
            for rel in rels:
                names = [a.get("name", "") for a in rel.get("assets", [])]
                seen.append(f"{rel.get('tag_name')}: {', '.join(n for n in names if 'win' in n.lower())}")
                for pat in (r"bin-win-cpu-x64\.zip$", r"bin-win-avx2-x64\.zip$", r"win.*cpu.*x64.*\.zip$"):
                    hit = [a for a in rel.get("assets", []) if re.search(pat, a.get("name", ""), re.I)]
                    if hit:
                        cands.append((hit[0]["name"], hit[0]["browser_download_url"]))
                        break
                if cands:
                    break
        except Exception as e:
            seen.append(f"GitHub API 失敗:{e}")
            print(f"GitHub API 無法使用({e}),改用固定版本", flush=True)
        for tag in LLAMA_PINNED:  # 找不到時用確定存在過的版本
            n = f"llama-{tag}-bin-win-cpu-x64.zip"
            cands.append((n, f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{n}"))
        errs = []
        for name, url in cands:
            z = MODEL_DIR / name
            try:
                download(url, z)
                break
            except Exception as e:
                errs.append(f"{url} → {e}")
        else:
            raise RuntimeError("llama.cpp 下載失敗\n" + "\n".join(errs) + "\n看到的版本:\n" + "\n".join(seen[:15]))
        import zipfile

        d.mkdir(exist_ok=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(d)
        z.unlink()
        found = list(d.rglob("llama-server.exe"))
        if not found:
            raise RuntimeError("llama.cpp 壓縮檔裡沒有 llama-server.exe")
        return found[0]

    @staticmethod
    def ensure_model():
        f = MODEL_DIR / QWEN_FILE
        if f.exists():
            return f
        print("下載 Qwen 翻譯模型(約 2GB,只有第一次)...", flush=True)
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(QWEN_REPO, QWEN_FILE, local_dir=str(MODEL_DIR)))

    def chat(self, sys_msg, user, max_tokens=400, schema=None, timeout=240):
        body = {"messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
                "temperature": 0.2, "max_tokens": max_tokens, "cache_prompt": True}
        if schema:
            body["json_schema"] = schema  # llama-server 會限制輸出一定符合這個格式
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"].strip()


def llama_schema(x):
    """Gemini 形式(型が大文字)の schema → llama-server 形式"""
    if isinstance(x, dict):
        return {k: (v.lower() if k == "type" else llama_schema(v)) for k, v in x.items()}
    if isinstance(x, list):
        return [llama_schema(v) for v in x]
    return x


class App:
    def __init__(self, threads, translate):
        self.hub = Hub()
        self.threads = threads
        self.want_translate = translate
        self.status = "loading"  # loading → idle → recording → idle
        self.message = "載入模型中…"
        self.engine = None
        self.translator = None
        self.translator_status = "off" if not translate else "loading"
        self.meeting = None
        self.started_at = None
        self.stop_evt = None
        self.pause_evt = threading.Event()
        self.workers = []
        self.summarizing = set()
        self.checking = set()
        self.clip = ClipWatch(self)
        self.qwen = None
        self.qwen_status = "loading"  # loading / ready / failed / sleep
        self.qwen_lock = threading.Lock()
        self.audio_q = None
        self.n_tracks = 1
        self.idle_since = 0.0
        self.lock = threading.Lock()
        self.notion_lock = threading.Lock()
        import soundcard as sc

        self.sc = sc

    # 背景載入模型,網頁可以先打開
    def load(self):
        try:
            try:
                sv, ja, vad, spk = ensure_models()
                self.engine = Engine(sv, ja, vad, spk, self.threads)
            except Exception as e:
                if "Protobuf" not in str(e) and "parsing" not in str(e) and "Load model" not in str(e):
                    raise
                print("モデルファイルが壊れています。ダウンロードし直します...", flush=True)
                self.set_status("loading", "モデルファイルが壊れていたため、ダウンロードし直しています(数分かかります)")
                repair_models()
                sv, ja, vad, spk = ensure_models()
                self.engine = Engine(sv, ja, vad, spk, self.threads)
            self.set_status("idle", "")
        except Exception as e:
            self.set_status("error", f"音声認識モデルを読み込めませんでした:{e}\n→ 黒い画面を閉じて、models フォルダを削除してから start_*.bat をもう一度起動してください")
            return
        threading.Thread(target=self.load_qwen, daemon=True).start()
        if self.want_translate:
            try:
                print("翻訳モデルを読み込み中(1分ほど。初回のみ約 2.5GB ダウンロード)...", flush=True)
                self.translator = Translator()
                self.translator_status = "ready"
            except Exception as e:
                print(f"翻訳モデルを読み込めませんでした。原文のみ表示します:{e}")
                self.translator_status = "failed"
            self.publish_state()

    def load_qwen(self):
        if not self.qwen_lock.acquire(blocking=False):
            return
        try:
            if self.qwen:
                return
            self.qwen_status = "loading"
            print("ローカル AI(要点まとめ用)を起動中…(初回のみ約 2GB ダウンロード)", flush=True)
            self.qwen = QwenTranslator()
            self.qwen_status = "ready"
            print("ローカル AI を起動しました", flush=True)
        except Exception as e:
            self.qwen_status = "failed"
            print(f"ローカル AI を起動できませんでした(Gemini は使えます):{e}", flush=True)
        finally:
            self.qwen_lock.release()
            self.publish_state()

    def qwen_sleep_later(self, since):
        time.sleep(QWEN_IDLE_SLEEP)
        if self.status == "idle" and self.idle_since == since and self.qwen:
            q, self.qwen = self.qwen, None
            self.qwen_status = "sleep"
            q.close()
            print("会議がないのでローカル AI を休ませます(録音開始で自動で起動)", flush=True)
            self.publish_state()

    def caption_lag(self):
        q = self.audio_q
        return q.qsize() * CHUNK / CAPTURE_RATE / self.n_tracks if q else 0.0

    def llm(self, d):
        """ブラウザからのローカル AI 呼び出し。字幕が遅れているときは断る(字幕が最優先)"""
        q = self.qwen
        if not q or not q.healthy():
            if self.qwen_status in ("sleep", "failed") or (q and not q.healthy()):
                self.qwen = None
                threading.Thread(target=self.load_qwen, daemon=True).start()
            raise RuntimeError("ローカル AI を準備中です")
        if self.caption_lag() > 3:
            raise RuntimeError("字幕の処理を優先しています。少しあとでまとめます")
        sch = d.get("schema")
        out = q.chat(str(d.get("system") or "JSON だけを出力してください。"), str(d.get("user", ""))[:9000],
                     max_tokens=int(d.get("max_tokens") or 450), schema=llama_schema(sch) if sch else None,
                     timeout=120 + int(d.get("max_tokens") or 450) // 4)
        return out

    def set_status(self, status, message):
        self.status = status
        self.message = message
        self.publish_state()

    def state(self):
        s = {
            "status": self.status,
            "message": self.message,
            "translator": self.translator_status,
            "local_ai": self.qwen_status,
        }
        if self.meeting:
            s["meeting"] = {
                "id": self.meeting.id,
                **self.meeting.meta,
                "lines": self.meeting.sorted_lines() if self.status == "recording" else [],
            }
            s["elapsed"] = time.time() - self.started_at if self.status == "recording" else 0
            s["paused"] = self.pause_evt.is_set()
        return s

    def publish_state(self):
        self.hub.publish({"type": "state", "status": self.status, "message": self.message,
                          "translator": self.translator_status})

    def devices(self):
        com_init()
        return list_devices(self.sc)

    def start(self, name, mode, mic_name, speaker_name, my_mic_name=None, people=None, screenshots=False):
        with self.lock:
            if self.status != "idle":
                raise RuntimeError("今は開始できません")
            com_init()
            now = dt.datetime.now()
            name = name.strip() or f"{now:%m/%d} 會議"
            folder = REC_DIR / f"{now:%Y%m%d-%H%M%S}"
            folder.mkdir(parents=True)
            m = Meeting(folder)
            m.meta = {"name": name, "mode": mode, "start": now.isoformat(timespec="seconds"),
                      "notes": "", "speakers": {}, "duration": 0, "people": people}
            m.save_meta()
            self.meeting = m
            self.started_at = time.time()
            self.stop_evt = threading.Event()
            self.pause_evt.clear()
            audio_q = queue.Queue()
            tracks = []
            sc = self.sc

            def dev(name, loopback):
                return lambda fallback: get_device(sc, None if fallback else name, loopback)

            mic = dev(mic_name, False)
            if mode == "online":
                tracks = [(dev(speaker_name, True), "them"), (mic, "me")]
            else:
                tracks = [(mic, "room")]
                if my_mic_name:
                    # 自己戴領夾/耳機麥克風:這條音軌就是我方,現場麥克風收其他人
                    tracks.append((dev(my_mic_name, False), "me"))
            m.meta["tracks"] = [t for _, t in tracks]
            m.save_meta()
            self.workers = []
            for dev, track in tracks:
                alts = (lambda: [d for d in sc.all_microphones(include_loopback=True) if d.isloopback]) if track == "them" else None
                t = threading.Thread(target=capture, args=(dev, track, audio_q, folder / f"{track}.wav", self.stop_evt, alts, self.pause_evt), daemon=True)
                t.start()
                self.workers.append(t)
            if screenshots:
                m.meta["screenshots"] = True
                m.save_meta()
                threading.Thread(target=self.run_screenshots, args=(m.dir / "shots", self.stop_evt, self.started_at), daemon=True).start()
            self.audio_q, self.n_tracks = audio_q, max(1, len(tracks))
            if self.qwen is None and self.qwen_status in ("sleep", "failed"):
                threading.Thread(target=self.load_qwen, daemon=True).start()
            self.proc = threading.Thread(target=self.process, args=(m, audio_q, now), daemon=True)
            self.proc.start()
            self.set_status("recording", "")
            print(f"録音開始:{name}({'オンライン' if mode == 'online' else '対面'})")

    def pause(self, on):
        """暫停/繼續:同一場會議、同一個資料夾,暫停中的聲音不錄"""
        with self.lock:
            if self.status != "recording" or on == self.pause_evt.is_set():
                return
            t = round(time.time() - self.started_at, 1)
            pauses = self.meeting.meta.setdefault("pauses", [])
            if on:
                self.pause_evt.set()
                pauses.append([t, None])
            else:
                self.pause_evt.clear()
                if pauses and pauses[-1][1] is None:
                    pauses[-1][1] = t
            self.meeting.save_meta()
            self.hub.publish({"type": "paused", "paused": on})

    def run_screenshots(self, folder, stop, t_start):
        try:
            screenshot_loop(folder, stop, t_start, paused=self.pause_evt)
        except Exception as e:
            self.hub.publish({"type": "error", "message": f"畫面截圖停止了:{e}(錄音不受影響)"})

    def stop(self):
        with self.lock:
            if self.status != "recording":
                return
            self.set_status("stopping", "終了処理中…")
            if self.pause_evt.is_set():
                pauses = self.meeting.meta.get("pauses") or []
                if pauses and pauses[-1][1] is None:
                    pauses[-1][1] = round(time.time() - self.started_at, 1)
                self.pause_evt.clear()
            self.stop_evt.set()
            for t in self.workers:
                t.join(timeout=3)
            self.proc.join(timeout=30)
            self.meeting.meta["duration"] = round(time.time() - self.started_at)
            self.meeting.save_meta()
            print(f"停止しました。録音と文字起こしの場所:{self.meeting.dir}")
            self.audio_q = None
            self.idle_since = time.time()
            threading.Thread(target=self.qwen_sleep_later, args=(self.idle_since,), daemon=True).start()
            self.set_status("idle", "")
        if claude_configured():
            threading.Thread(target=self.summarize, args=(self.meeting.id,), daemon=True).start()

    def process(self, m, audio_q, start):
        pipe = Pipeline(self, m, start)
        while True:
            try:
                track, x = audio_q.get(timeout=0.3)
            except queue.Empty:
                if self.stop_evt.is_set() and all(not w.is_alive() for w in self.workers):
                    break
                continue
            if isinstance(x, Exception):
                self.hub.publish({"type": "error", "message": f"録音が途切れました({track})。デバイスに再接続中…:{x}"})
                continue
            if isinstance(x, str):
                self.hub.publish({"type": "error", "message": x})
                continue
            pipe.feed(track, x)
        pipe.finish(wait_translation=False)

    def import_audio(self, data, filename, name, notes, people, lastmod_ms):
        """手機等其他裝置錄的音檔:轉成 16kHz 單聲道 room.wav,建立會議後整段辨識"""
        import av  # PyAV:讀 m4a/mp3/wav 等(內含 ffmpeg,不用另外安裝)

        if self.status != "idle":
            raise RuntimeError("録音中、または別の会議を処理中です。しばらくしてからお試しください")
        now = dt.datetime.now()
        start = dt.datetime.fromtimestamp(lastmod_ms / 1000) if lastmod_ms else now
        folder = REC_DIR / f"{now:%Y%m%d-%H%M%S}"
        folder.mkdir(parents=True)
        ext = Path(filename or "audio").suffix.lower() or ".m4a"
        src = folder / f"source{ext}"
        src.write_bytes(data)
        try:
            chunks = []
            with av.open(str(src)) as c:
                rs = av.AudioResampler(format="s16", layout="mono", rate=ASR_RATE)
                for frame in c.decode(audio=0):
                    for f in rs.resample(frame):
                        chunks.append(f.to_ndarray().reshape(-1))
                for f in rs.resample(None):
                    chunks.append(f.to_ndarray().reshape(-1))
            pcm = np.concatenate(chunks).astype(np.int16) if chunks else np.zeros(0, np.int16)
        except Exception as e:
            raise RuntimeError(f"この音声ファイルを読み込めません({filename}):{e}")
        if len(pcm) < ASR_RATE:
            raise RuntimeError("音声が短すぎるか、音が入っていません")
        with wave.open(str(folder / "room.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(ASR_RATE)
            w.writeframes(pcm.tobytes())
        duration = round(len(pcm) / ASR_RATE)
        if lastmod_ms:  # 檔案時間通常是錄音結束的時間
            start = start - dt.timedelta(seconds=duration)
        name = (name or "").strip() or f"{start:%m/%d} 會議(手機錄音)"
        m = Meeting(folder)
        m.meta = {"name": name, "mode": "room", "start": start.isoformat(timespec="seconds"),
                  "notes": notes or "", "speakers": {}, "duration": duration, "people": people,
                  "tracks": ["room"], "imported": filename or ""}
        m.save_meta()
        threading.Thread(target=self.reprocess, args=(m.id,), daemon=True).start()
        return m.id

    def reprocess(self, mid):
        """用目前的模型把錄音檔整段重新辨識"""
        with self.lock:
            if self.status != "idle":
                raise RuntimeError("録音中、または別の会議を処理中です。しばらくしてからお試しください")
            self.set_status("processing", "重新辨識中…")
        try:
            m = Meeting(REC_DIR / mid)
            tracks = [t for t in ("them", "me", "room") if (m.dir / f"{t}.wav").exists()]
            m.meta["tracks"] = tracks
            audio = {}
            for t in tracks:
                with wave.open(str(m.dir / f"{t}.wav")) as w:
                    audio[t] = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
            if m.lines_path.exists():
                m.lines_path.replace(m.dir / "transcript_prev.jsonl")
            m.lines = {}
            pipe = Pipeline(self, m, dt.datetime.fromisoformat(m.meta["start"]), live=False)
            step = ASR_RATE * 2
            total = max(len(a) for a in audio.values())
            for i in range(0, total, step):
                for t, a in audio.items():
                    if i < len(a):
                        pipe.feed(t, a[i:i + step])
                self.hub.publish({"type": "progress", "id": mid, "pct": round(i * 90 / total),
                                  "text": "認識中"})
            pipe.finish(wait_translation=True, progress_id=mid)
            m.save_meta()
            if not m.lines:
                self.hub.publish({"type": "progress", "id": mid, "pct": 0, "done": True, "empty": True,
                                  "text": "音声から発言を認識できませんでした。無音か音が小さすぎる録音の可能性があります(別の音声ファイルでお試しください)"})
                return
            self.hub.publish({"type": "progress", "id": mid, "pct": 100, "text": "完了", "done": True})
            if claude_configured():
                threading.Thread(target=self.summarize, args=(mid,), daemon=True).start()
        except Exception as e:
            self.hub.publish({"type": "progress", "id": mid, "pct": 0, "text": f"失敗:{e}", "done": True})
        finally:
            self.set_status("idle", "")

    def respeak(self, mid, people=None):
        """話者だけ分け直す:文字・翻訳はそのまま、会議全体の声の特徴でまとめて分ける"""
        with self.lock:
            if self.status != "idle":
                raise RuntimeError("録音中、または別の会議を処理中です。しばらくしてからお試しください")
            self.set_status("processing", "話者を分け直しています…")
        try:
            m = Meeting(REC_DIR / mid)
            if people:
                m.meta["people"] = people
            people = m.meta.get("people")
            eng = self.engine
            groups = {"them": [r for r in m.lines.values() if str(r.get("sid", "")).startswith("them")],
                      "room": [r for r in m.lines.values() if re.fullmatch(r"p\d+", str(r.get("sid", "")))]}
            has_me = any(r.get("sid") == "me" for r in m.lines.values()) or (m.dir / "me.wav").exists()
            done_tracks = [t for t in groups if groups[t] and (m.dir / f"{t}.wav").exists()]
            if not done_tracks:
                raise RuntimeError("録音ファイル、または分けられる話者が見つかりません")
            for ti, t in enumerate(done_tracks):
                with wave.open(str(m.dir / f"{t}.wav")) as w:
                    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
                vad = eng.new_vad()
                segs = []

                def drain():
                    while not vad.empty():
                        sg = vad.front
                        x = np.array(sg.samples)
                        vad.pop()
                        segs.append((sg.start / ASR_RATE, len(x) / ASR_RATE, eng.embed(x)))
                win = eng.window
                for i in range(0, len(a) - win, win):
                    vad.accept_waveform(a[i:i + win])
                    if i % (win * 500) == 0:
                        drain()
                        self.hub.publish({"type": "progress", "id": mid, "text": "声の特徴を分析中",
                                          "pct": round((ti + i / len(a)) * 90 / len(done_tracks))})
                vad.flush()
                drain()
                if not segs:
                    continue
                n = (max(1, int(people) - 1) if has_me else int(people)) if people else None
                lab = recluster([x[2] for x in segs], [x[1] for x in segs], n)
                starts = np.array([x[0] for x in segs])
                prefix = "them" if t == "them" else "p"
                for r in groups[t]:
                    j = int(np.argmin(np.abs(starts - r["offset"])))
                    if abs(starts[j] - r["offset"]) < 1.5:
                        m.update_line({"id": r["id"], "sid": f"{prefix}{lab[j] + 1}"})
            m.meta["speakers"] = {}  # 番号が変わるので、前に付けた名前はリセット
            m.save_meta()
            self.hub.publish({"type": "progress", "id": mid, "pct": 100, "text": "完了", "done": True, "respeak": True})
        except Exception as e:
            self.hub.publish({"type": "progress", "id": mid, "pct": 0, "text": f"失敗:{e}", "done": True})
        finally:
            self.set_status("idle", "")

    def revise(self, mid, instruction):
        """今の議事録を、指示どおりに Gemini で直す"""
        with self.lock:
            if mid in self.summarizing:
                return
            self.summarizing.add(mid)
        self.hub.publish({"type": "summary", "id": mid, "status": "running"})
        try:
            m = self.meeting if self.meeting and self.meeting.id == mid else Meeting(REC_DIR / mid)
            text = gemini_revise(m, instruction)
            m = self.meeting if self.meeting and self.meeting.id == mid else Meeting(REC_DIR / mid)
            m.meta["summary"] = text
            m.meta.setdefault("revisions", []).append(instruction)
            m.save_meta()
            self.hub.publish({"type": "summary", "id": mid, "status": "done", "summary": text})
        except Exception as e:
            self.hub.publish({"type": "summary", "id": mid, "status": "error", "message": str(e)})
        finally:
            self.summarizing.discard(mid)

    def summarize(self, mid, lang=None):
        """バックグラウンドで Gemini にまとめてもらい、meta に保存して画面に通知"""
        with self.lock:
            if mid in self.summarizing:
                return
            self.summarizing.add(mid)
        m = self.meeting if self.meeting and self.meeting.id == mid else Meeting(REC_DIR / mid)
        lang = lang or m.meta.get("out_lang") or "ja"
        self.hub.publish({"type": "summary", "id": mid, "status": "running"})
        try:
            if not m.lines:
                raise RuntimeError("文字起こしがありません")
            text = claude_summarize(m, lang)
            # 整理期間名稱、筆記可能被改過,重新取最新的
            m = self.meeting if self.meeting and self.meeting.id == mid else Meeting(REC_DIR / mid)
            m.meta["summary"] = text
            m.meta["out_lang"] = lang
            m.meta.pop("zh", None)
            m.save_meta()
            self.hub.publish({"type": "summary", "id": mid, "status": "done", "summary": text})
        except Exception as e:
            self.hub.publish({"type": "summary", "id": mid, "status": "error", "message": str(e)})
            self.summarizing.discard(mid)
            return
        self.summarizing.discard(mid)

    def translate_check(self, mid):
        """議事錄整理好之後,接著做中文確認版"""
        with self.lock:
            if mid in self.checking:
                return
            self.checking.add(mid)
        self.hub.publish({"type": "zh", "id": mid, "status": "running"})
        try:
            m = self.meeting if self.meeting and self.meeting.id == mid else Meeting(REC_DIR / mid)
            zh = claude_translate_zh(m)
            m = self.meeting if self.meeting and self.meeting.id == mid else Meeting(REC_DIR / mid)
            m.meta["zh"] = zh
            m.save_meta()
            self.hub.publish({"type": "zh", "id": mid, "status": "done"})
        except Exception as e:
            self.hub.publish({"type": "zh", "id": mid, "status": "error", "message": str(e)})
        finally:
            self.checking.discard(mid)

    def update_meta(self, mid, data):
        m = self.meeting.refresh() if self.meeting and self.meeting.id == mid else Meeting(REC_DIR / mid)
        for k in ("name", "notes", "notes_after", "speakers", "summary", "out_lang", "length", "style_note", "digests", "turns", "summary_zh", "summary_zh_src"):
            if k in data:
                if k == "summary" and data[k] != m.meta.get("summary"):
                    m.meta.pop("zh", None)  # 手動改了議事錄,中文確認版要重做
                m.meta[k] = data[k]
        m.save_meta()
        if "length" in data or "style_note" in data:  # 次の会議もこの設定から始める
            cfg = load_config()
            for k in ("length", "style_note"):
                if k in data:
                    cfg[k] = data[k]
            save_config(cfg)
        if "speakers" in data:
            self.hub.publish({"type": "speakers", "speakers": m.meta["speakers"]})


GLOSSARY_PATH = BASE / "用語.txt"
GLOSSARY_DEFAULT = """# よく聞き間違える言葉:1行に「間違った書き方=正しい書き方」。保存すると次の発言から反映
# 例:田中産業=タナカ産業
"""
_glossary = {"mtime": None, "pairs": []}
TERMS_PATH = BASE / "對照表.txt"  # 「台灣說法=日文說法」,翻譯後面標出對方的說法
_terms = {"mtime": None, "rows": []}


def load_terms():
    """讀 對照表.txt(改過才重讀)。回傳 [(台灣說法list, 日文說法list)]"""
    try:
        mt = TERMS_PATH.stat().st_mtime
        if mt != _terms["mtime"]:
            rows = []
            for line in TERMS_PATH.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    tw, ja = line.split("=", 1)
                    tw = [x.strip() for x in tw.split("、") if x.strip()]
                    ja = [x.strip() for x in ja.split("、") if x.strip()]
                    if tw and ja:
                        rows.append((tw, ja))
            _terms.update(mtime=mt, rows=rows)
    except OSError:
        return []
    return _terms["rows"]


def terms_in(text, limit=80):
    """整理/翻譯用:對照表裡在這段文字出現過的詞,回傳「台灣說法=日文說法」清單"""
    out = []
    for tw, ja in load_terms():
        if any(len(w) >= 2 and w in text for w in tw + ja):
            line = f"{'、'.join(tw)}={'、'.join(ja)}"
            if line not in out:
                out.append(line)
    return out[:limit]


def term_hints(text, lang, limit=4):
    """日文句子 → 標出台灣說法;中文句子 → 標出日文說法。例:〔ビード=焊道〕"""
    if not load_terms():
        return ""
    src_i = 0 if lang in ("zh", "yue") else 1
    found = []
    for row in _terms["rows"]:
        src, dst = row[src_i], row[1 - src_i]
        for w in sorted(src, key=len, reverse=True):
            i = text.find(w)
            if i >= 0 and dst[0] not in text:
                found.append((i, i + len(w), w, dst[0]))
                break
    found.sort(key=lambda f: (-(f[1] - f[0]), f[0]))
    used, hints = [], []
    for a, b, w, d in found:  # 長的詞優先,重疊的不重複標
        if all(b <= x or a >= y for x, y in used) and d != w:
            used.append((a, b))
            hints.append((a, f"{w}={d}"))
    hints = [h for _, h in sorted(hints)[:limit]]
    return "〔" + "・".join(hints) + "〕" if hints else ""


def apply_glossary(text):
    try:
        if not GLOSSARY_PATH.exists():
            GLOSSARY_PATH.write_text(GLOSSARY_DEFAULT, encoding="utf-8")
        mt = GLOSSARY_PATH.stat().st_mtime
        if mt != _glossary["mtime"]:
            pairs = []
            for line in GLOSSARY_PATH.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    a, b = line.split("=", 1)
                    if a.strip():
                        pairs.append((a.strip(), b.strip()))
            pairs.sort(key=lambda p: -len(p[0]))
            _glossary.update(mtime=mt, pairs=pairs)
    except OSError:
        pass
    for a, b in _glossary["pairs"]:
        text = text.replace(a, b)
    return text


FILLERS_EN = {"okay", "ok", "yeah", "yes", "no", "uh", "um", "hmm", "oh", "so", "different", "right", "i", "o"}


def is_noise(lang, text):
    """雜音、咳嗽、鍵盤聲常被辨識成單字或別的語言,這些不顯示"""
    core = re.sub(r"[\s、。,.!?！？…・「」()()\-]", "", text)
    if not core:
        return True
    if lang not in ("ja", "zh", "en", "yue"):
        return True
    if re.fullmatch(r"[嗯啊呃哦喔欸唔恩额噢]+", core):
        return True
    if lang == "en" and len(text.split()) <= 2 and core.lower().strip(".") in FILLERS_EN | {w + "." for w in FILLERS_EN}:
        return True
    if lang in ("zh", "yue") and len(core) <= 1:
        return True
    return False


class Pipeline:
    """音軌 → 斷句 → 辨識 → 說話者 → 翻譯。錄音中和重新辨識共用"""

    def __init__(self, app, m, start, live=True):
        self.app, self.m, self.start, self.live = app, m, start, live
        self.eng = app.engine
        self.tracks = m.meta.get("tracks") or ["room"]
        self.vads = {t: self.eng.new_vad() for t in self.tracks}
        self.bufs = {t: np.zeros(0, np.float32) for t in self.tracks}
        self.fed = {t: 0 for t in self.tracks}
        people = m.meta.get("people")
        others = max(1, int(people) - 1) if people and "me" in self.tracks else (int(people) if people else None)
        self.trackers = {"them": SpeakerTracker("them", others), "room": SpeakerTracker("p", others)}
        # 面對面+個人麥克風:現場麥克風也會收到我的聲音,要等我方音軌處理到同一時間點再判斷
        self.dedupe = "me" in self.tracks and "room" in self.tracks
        self.me_spans = []
        self.pending = []
        self.rid = max(m.lines) if m.lines else 0
        self.embs = []  # (行id, 音軌, 聲紋, 秒數):重新辨識結束後整場一起分群
        self.tr_q = queue.Queue()
        self.tw = threading.Thread(target=self.translate_worker, daemon=True)
        self.tw.start()

    def feed(self, track, x):
        eng = self.eng
        b = np.concatenate([self.bufs[track], x])
        v = self.vads[track]
        while len(b) >= eng.window:
            v.accept_waveform(b[: eng.window])
            b = b[eng.window:]
        self.bufs[track] = b
        self.fed[track] += len(x)
        self.drain(track)

    def drain(self, track, final=False):
        v = self.vads[track]
        while not v.empty():
            seg = v.front
            item = (track, seg.start, np.array(seg.samples))
            v.pop()
            if self.dedupe and track == "room":
                self.pending.append(item)
            else:
                if track == "me":
                    self.me_spans.append((seg.start, seg.start + len(item[2])))
                self.handle(*item)
        if self.dedupe:
            ready = self.fed.get("me", 0) - 3 * ASR_RATE
            keep = []
            for it in self.pending:
                if final or it[1] + len(it[2]) < ready:
                    self.handle(*it)
                else:
                    keep.append(it)
            self.pending = keep

    def overlap_me(self, a, n):
        b = a + n
        ov = sum(max(0, min(b, y) - max(a, x)) for x, y in self.me_spans if y > a and x < b)
        return ov / max(1, n)

    def handle(self, track, start, samples):
        eng, m = self.eng, self.m
        if self.dedupe and track == "room" and self.overlap_me(start, len(samples)) > 0.5:
            return  # 我在講話時現場麥克風收到的聲音,以個人麥克風為準
        lang, text = eng.transcribe(samples)
        if is_noise(lang, text):
            return
        secs = len(samples) / ASR_RATE
        if track == "me":
            sid = "me"
        else:
            emb = eng.embed(samples)
            sid = self.trackers[track].assign(emb, secs)
        self.rid += 1
        if track != "me" and not self.live:
            self.embs.append((self.rid, track, emb, secs))
        t = start / ASR_RATE
        r = {"id": self.rid, "sid": sid, "offset": round(t, 2),
             "time": (self.start + dt.timedelta(seconds=t)).strftime("%H:%M:%S"),
             "lang": lang, "text": text}
        m.update_line(r)
        if self.live:
            self.app.hub.publish({"type": "line", **r})
            print(f"[{r['time']}] {m.speaker_name(sid)} ({lang}) {text}")
        if self.app.translator and lang in NLLB_LANG:
            self.tr_q.put(r)

    def translate_worker(self):
        while True:
            r = self.tr_q.get()
            if r is None:
                return
            # 翻譯跟不上時,把同一人連續的片段合成一句一起翻(較快,也較通順),翻譯放在最後一段下面
            while self.live and len(r["text"]) < 120:
                try:
                    nxt = self.tr_q.get_nowait()
                except queue.Empty:
                    break
                if nxt is None or nxt.get("sid") != r.get("sid") or nxt["lang"] != r["lang"]:
                    self.tr_q.queue.appendleft(nxt)
                    break
                r = dict(nxt, text=r["text"] + nxt["text"])
            try:
                zh = apply_glossary(self.app.translator(r["text"], r["lang"]))
            except Exception as e:
                zh = f"(翻譯失敗:{e})"
            hint = term_hints(r["text"], r["lang"])
            zh = f"{zh} {hint}".strip() if hint else zh
            if zh:
                self.m.update_line({"id": r["id"], "zh": zh})
                if self.live:
                    self.app.hub.publish({"type": "line", "id": r["id"], "zh": zh})

    def finish(self, wait_translation=False, progress_id=None):
        for t in self.tracks:
            self.vads[t].flush()
            self.drain(t)
        if self.dedupe:
            self.drain("room", final=True)
        total = self.tr_q.qsize()
        self.tr_q.put(None)
        if wait_translation:
            while self.tw.is_alive():
                self.tw.join(timeout=1)
                if progress_id and total:
                    left = self.tr_q.qsize()
                    self.app.hub.publish({"type": "progress", "id": progress_id,
                                          "pct": 90 + round((total - left) * 10 / total), "text": "翻訳中"})
        else:
            self.tw.join(timeout=20)

        if not self.live and self.embs:
            self.relabel()

    def relabel(self):
        """重新辨識完:用整場的聲紋重新分說話者(比錄音中邊聽邊分準)"""
        people = self.m.meta.get("people")
        has_me = "me" in self.tracks
        n = (max(1, int(people) - 1) if has_me else int(people)) if people else None
        for t in ("them", "room"):
            rows = [x for x in self.embs if x[1] == t]
            if len(rows) < 2:
                continue
            lab = recluster([x[2] for x in rows], [x[3] for x in rows], n)
            prefix = "them" if t == "them" else "p"
            for (rid, _, _, _), k in zip(rows, lab):
                if rid in self.m.lines:
                    self.m.update_line({"id": rid, "sid": f"{prefix}{k + 1}"})
        self.m.meta["speakers"] = {}

def list_meetings():
    if not REC_DIR.exists():
        return []
    out = []
    for d in sorted(REC_DIR.iterdir(), reverse=True):
        if (d / "meta.json").exists():
            try:
                out.append(Meeting(d).summary())
            except Exception:
                pass
    return out


# ---------- 匯出 ----------

PROMPT_JA = """【議事録作成のお願い】
あなたは会議の議事録を作るアシスタントです。このメッセージには「作り方の指示」と、「----」の行より後に「会議の情報・メモ・文字起こし」がすべて入っています。ほかのファイルを探したり質問を返したりせず、このメッセージだけを読んで、すぐに議事録を作ってください。

以下は会議の文字起こし(音声自動認識のため誤字あり)と私のメモです。日本語で、下記の2つを、指定の区切り行も含めてそのまま出力してください。区切り行以外の前置きや説明は不要です。

=====摘要=====
- 会議全体の要点を短く要約(行数は下の「長さ」の指定に従う)
- 「長さ」が「標準」「詳細」のときだけ、続けて「**各者の主張**」として話者(当方/先方、または話者名)ごとに主張・論点を箇条書き

=====議事録=====
次の構成で書くこと:

**背景**
- 会議に至った経緯・前提を2〜3点

## 1. 結論
- 合意した事項、決まった次のアクション(誰が何をするか)

## 2. 情報共有
- 時系列ではなくテーマごとに「**案件概要**」「**スケジュール(見込み)**」「**仕様・技術**」「**進め方**」「**検討中**」などの太字小見出しで分けて箇条書き(該当しない見出しは省略、必要なら追加)

## 3. ToDo
- 内容(期限) の形式。担当が当方以外なら先頭に担当者名

書き方のルール:
- です・ます調にしない。体言止めや「〜する」で簡潔に
- 数値・規模・日付・型番は省略せず具体的に残す
- 見込みや未確定の情報は「見込み」「未確定」「要確認」と明記。先方の説明だけで裏付けがない情報は「先方説明に基づく」と付記
- 認識誤りと思われる箇所は文脈から補正し、判断できない部分は「要確認」と書く
- 中国語・英語の発言も日本語でまとめる。社名・人名・型番は原語のまま残してよい
- 話者ラベルは音声から自動推定したもので誤りが多い。誰の発言かは内容(自社/御社などの言い方、質問と回答の流れ)から判断すること
- メモは私が書いたもので正確。メモの内容は必ず議事録に反映し、文字起こしと食い違う場合はメモを優先する

重複を書かないルール(最重要):
- 同じ事実・同じ決定事項は議事録の中で1回だけ書く。背景・結論・情報共有・ToDo の間で言い換えて繰り返さない
- 結論に書いたことは情報共有に書かない。ToDo は「誰が・何を・いつまでに」だけにし、結論の言い換えにしない
- 会議中に何度も出た話は1つにまとめる。あいさつ・雑談・自己紹介・確認のための繰り返しは書かない
- 1項目は1行。前置き・感想・「〜について話し合った」のような中身のない文は書かない
"""

PROMPT_ZH = """以下是一場會議的逐字稿(語音自動辨識,可能有錯字)和我的筆記。
請輸出下面兩個部分,分隔行也要原樣保留。分隔行以外不需要任何開場白或說明。

=====摘要=====
- 用 3~5 行摘要整場會議
- 接著以「**各方主張**」為標題,依說話者(我方/對方,或說話者名稱)條列各自的主張與論點

=====議事録=====
依照以下結構撰寫:

**背景**
- 這場會議的來由與前提,2~3 點

## 1. 結論
- 達成共識的事項、決定的下一步(誰做什麼)

## 2. 情報共有
- 依主題用「**案件概要**」「**時程(預估)**」「**規格・技術**」「**進行方式**」「**檢討中**」等粗體小標分組條列(不適用的小標省略,需要時可增加)

## 3. ToDo
- 內容(期限)的格式。負責人不是我方時,開頭寫上負責人

寫法規則:
- 簡潔條列,不要寫成長句
- 數字、規模、日期、型號不要省略,具體保留
- 預估或未確定的資訊註明「預估」「未確定」「待確認」。只有對方口頭說明、沒有佐證的資訊註明「對方說明」
- 看起來是辨識錯誤的地方依上下文修正,無法判斷的加上(?)
- 說話者標籤是依聲音自動判斷,常有錯誤。是誰說的請依內容判斷(例如「我們公司/貴公司」的說法、提問與回答的順序)
- 筆記是我自己寫的,內容正確。筆記的內容一定要寫進議事錄,和逐字稿有出入時以筆記為準
"""

SEP_SUMMARY = "=====摘要====="
SEP_MINUTES = "=====議事録====="


def fmt_duration(sec):
    sec = int(sec)
    return f"{sec // 3600}:{sec // 60 % 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


def speaker_names(m):
    names = []
    for r in m.sorted_lines():
        n = m.speaker_name(r["sid"])
        if n not in names:
            names.append(n)
    return names


LENGTHS = {
    "short": ("簡潔", "【長さ:簡潔】摘要は3行以内(「各者の主張」は書かない)。議事録は A4 で1枚以内。背景は1〜2点、各見出しは要点だけ3〜5項目まで"),
    "normal": ("標準", "【長さ:標準】摘要は3〜5行、「各者の主張」は話者ごとに2〜3点まで。議事録は A4 で1〜2枚"),
    "long": ("詳細", "【長さ:詳細】摘要は5行程度で「各者の主張」も書く。議事録は数値・経緯・理由を省略せず詳しく(ただし重複は書かない)"),
}


def style_lines(meta):
    """議事録の長さと、お客様が書いた「書き方の希望」"""
    cfg = load_config()
    length = meta.get("length") or cfg.get("length") or "short"
    note = meta.get("style_note")
    if note is None:
        note = cfg.get("style_note", "")
    out = ["■ 長さ", LENGTHS.get(length, LENGTHS["short"])[1]]
    if note.strip():
        out += ["", "■ 書き方の希望(上の構成やルールと違っても、こちらを最優先で守ること)", note.strip()]
    return out


def claude_prompt(m, lang="ja", audio=False):
    meta = m.meta
    ja = lang == "ja"
    out = [
        PROMPT_JA if ja else PROMPT_ZH,
        *style_lines(meta),
        *(["", "■ 録音について", "会議の録音(音声)を添付している。下の文字起こしは自動認識で聞き間違いが多いので、内容・数字・固有名詞は音声を聞いて正しく直すこと。話者の区別も声で判断すること。"] if audio else []),
        "----",
        f"{'会議名' if ja else '會議名稱'}:{meta.get('name', '')}",
        f"{'日時' if ja else '日期'}:{meta.get('start', '')[:16].replace('T', ' ')}({fmt_duration(meta.get('duration', 0))})",
        f"{'形式' if ja else '形式'}:{('オンライン' if ja else '線上會議') if meta.get('mode') == 'online' else ('対面' if ja else '面對面')}",
        f"{'話者' if ja else '說話者'}:{'、'.join(speaker_names(m))}",
        "",
        "■ " + ("メモ(会議中)" if ja else "我的筆記(會議中)"),
        meta.get("notes", "").strip() or "(なし)",
        "",
        "■ " + ("メモ(会議後に追記)" if ja else "我的筆記(會後追加)"),
        meta.get("notes_after", "").strip() or "(なし)",
        "",
        "■ " + ("文字起こし" if ja else "逐字稿"),
    ]
    apply_glossary("")
    terms = sorted({b for _, b in _glossary["pairs"] if b})
    if terms:
        out[-1:-1] = ["■ " + ("用語(正しい表記)" if ja else "專有名詞(正確寫法)"), "、".join(terms), ""]
    hits = terms_in("\n".join([meta.get("notes", ""), meta.get("notes_after", "")] + [r.get("text", "") for r in m.sorted_lines()]))
    if hits:
        out[-1:-1] = ["■ " + ("用語集(台湾での言い方=日本語。この会議に出た語。日本語の議事録は右側、中国語の場合は左側の表記に統一すること)" if ja else "用語對照(台灣說法=日文說法。本場會議出現的詞。中文議事錄用左邊、日文用右邊的寫法統一)"), *hits, ""]
    # 機器翻譯品質不穩,只給原文,翻譯交給 Claude
    for r in m.sorted_lines():
        out.append(f"[{r.get('time', '')}] {m.speaker_name(r['sid'])}:{r.get('text', '')}")
    return "\n".join(out)


def split_summary(text):
    """把 Claude 的回覆分成 (摘要, 議事録)"""
    text = (text or "").strip()
    if SEP_MINUTES in text:
        a, b = text.split(SEP_MINUTES, 1)
        return a.replace(SEP_SUMMARY, "").strip(), b.strip()
    if SEP_SUMMARY in text:
        return text.replace(SEP_SUMMARY, "").strip(), ""
    return "", text


# ---------- ChatGPT の回答を自動で取り込む(クリップボード監視) ----------

def read_clipboard():
    """Windows のクリップボードの文字を読む(読めなければ None)"""
    if os.name != "nt":
        return None
    u, k = ctypes.windll.user32, ctypes.windll.kernel32
    u.OpenClipboard.argtypes = [ctypes.c_void_p]
    u.GetClipboardData.restype = ctypes.c_void_p
    u.GetClipboardData.argtypes = [ctypes.c_uint]
    k.GlobalLock.restype = ctypes.c_void_p
    k.GlobalLock.argtypes = [ctypes.c_void_p]
    k.GlobalUnlock.argtypes = [ctypes.c_void_p]
    for _ in range(5):  # ほかのアプリが使用中なら少し待つ
        if u.OpenClipboard(None):
            break
        time.sleep(0.05)
    else:
        return None
    try:
        h = u.GetClipboardData(13)  # CF_UNICODETEXT
        if not h:
            return None
        ptr = k.GlobalLock(h)
        if not ptr:
            return None
        try:
            return ctypes.wstring_at(ptr)
        finally:
            k.GlobalUnlock(h)
    finally:
        u.CloseClipboard()


def clipboard_seq():
    if os.name != "nt":
        return 0
    return ctypes.windll.user32.GetClipboardSequenceNumber()


def looks_like_minutes(text):
    """ChatGPT の回答(議事録)らしいか。こちらが作った依頼文は取り込まない"""
    t = (text or "").strip()
    if len(t) < 30 or "【議事録作成のお願い】" in t or "【議事録の修正のお願い】" in t or "■ 文字起こし" in t:
        return False
    return SEP_SUMMARY in t or SEP_MINUTES in t or ("## 1." in t and "## 3." in t)


class ClipWatch:
    """「ChatGPT 用にコピー」を押してから 30 分間、クリップボードに議事録が入ったら自動で取り込む"""

    def __init__(self, app):
        self.app = app
        self.mid = None
        self.gen = 0

    def start(self, mid):
        self.gen += 1
        self.mid = mid
        threading.Thread(target=self.run, args=(self.gen, mid), daemon=True).start()

    def stop(self):
        self.gen += 1
        self.mid = None

    def run(self, gen, mid):
        seq = clipboard_seq()
        end = time.time() + 1800
        while gen == self.gen and time.time() < end:
            time.sleep(0.7)
            n = clipboard_seq()
            if n == seq:
                continue
            seq = n
            try:
                text = read_clipboard()
            except Exception:
                continue
            if not looks_like_minutes(text):
                continue
            text = text.strip()
            m = Meeting(REC_DIR / mid)
            if text == (m.meta.get("summary") or "").strip():
                continue
            self.app.update_meta(mid, {"summary": text})
            self.app.hub.publish({"type": "summary", "id": mid, "status": "done", "summary": text, "via": "clip"})
            print("ChatGPT の回答を議事録に取り込みました", flush=True)


# ---------- Gemini に聞かせる録音(小さい音声ファイルに変換) ----------
AUDIO_MAX_MIN = 90  # これより長い会議は録音を付けない(Gemini に送れるサイズの上限)


def meeting_audio(m):
    """会議の音声トラックを1本に混ぜて ogg(opus 24kbps)にする。返り値 (bytes, mime)"""
    import av

    tracks = [t for t in ("them", "me", "room") if (m.dir / f"{t}.wav").exists()]
    if not tracks:
        raise FileNotFoundError("録音ファイルがありません")
    mix = None
    for t in tracks:
        with wave.open(str(m.dir / f"{t}.wav")) as w:
            a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        if mix is None:
            mix = a
        else:
            if len(a) > len(mix):
                a, mix = mix, a
            mix = mix.copy()
            mix[:len(a)] += a
    if len(mix) / ASR_RATE > AUDIO_MAX_MIN * 60:
        raise ValueError("long")
    peak = float(np.abs(mix).max() or 1)
    mix = (np.clip(mix / max(peak, 0.05) * 0.9, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    for codec, fmt, mime, rate in (("libopus", "ogg", "audio/ogg", 16000), ("libmp3lame", "mp3", "audio/mp3", 16000),
                                   ("aac", "adts", "audio/aac", 16000)):
        try:
            buf = io.BytesIO()
            with av.open(buf, "w", format=fmt) as c:
                st = c.add_stream(codec, rate=rate)
                st.bit_rate = 16000
                try:
                    st.layout = "mono"
                except Exception:
                    pass
                for i in range(0, len(mix), ASR_RATE * 10):
                    fr = av.AudioFrame.from_ndarray(mix[i:i + ASR_RATE * 10][None, :], format="s16", layout="mono")
                    fr.sample_rate = ASR_RATE
                    for pk in st.encode(fr):
                        c.mux(pk)
                for pk in st.encode(None):
                    c.mux(pk)
            return buf.getvalue(), mime
        except Exception as e:
            print(f"音声の変換({codec})に失敗:{e}", flush=True)
    raise RuntimeError("録音を変換できませんでした")


# ---------- 一鍵更新 ----------

def install_update(code, want):
    """ブラウザが GitHub から取ってきた新しい meeting.py を検査して置き換える"""
    if f'BUILD = "{want}"' not in code or "def main():" not in code:
        raise ValueError("ダウンロードしたファイルが正しくありません。もう一度お試しください")
    compile(code, "meeting.py", "exec")  # 文法エラーのある版は入れない
    me = Path(__file__).resolve()
    bk = BASE / "backup"
    bk.mkdir(exist_ok=True)
    shutil.copy2(me, bk / f"meeting_{BUILD}.py")  # 前の版はここに残す(戻すときはこれを meeting.py に)
    tmp = me.with_suffix(".new")
    tmp.write_text(code, encoding="utf-8", newline="\n")
    os.replace(tmp, me)
    (BASE / "updated.flag").write_text(want, encoding="utf-8")
    print(f"新しい版({want})に更新しました。再起動します…", flush=True)


CONFIG_PATH = BASE / "config.json"  # Gemini の API キー(このパソコンだけに保存)
GEMINI_MODELS = ["gemini-flash-latest", "gemini-2.5-flash"]  # 上から順に試す
GEMINI_KEY_URL = "https://aistudio.google.com/apikey"


def load_config():
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")


def gemini_configured():
    return bool(load_config().get("gemini_key"))


def claude_configured():
    """サーバー側の自動まとめは使わない(会社のネットワークでも動くよう、Gemini はブラウザから呼ぶ)"""
    return False


_pac = {"session": None}


def http_post(url, body, headers, timeout):
    """ブラウザと同じネットワーク設定(会社のプロキシ・証明書)で POST する。(status, text) を返す"""
    try:
        import truststore  # Windows の証明書ストアを使う(会社の SSL 検査対策)

        truststore.inject_into_ssl()
    except Exception:
        pass
    try:
        from pypac import PACSession  # Windows のプロキシ自動設定(PAC)を読む
    except Exception:
        PACSession = None
    if PACSession:
        import requests

        if _pac["session"] is None:
            _pac["session"] = PACSession()
            try:
                pac = _pac["session"].get_pac()
                print("ネットワーク:" + ("会社のプロキシ自動設定(PAC)を使用" if pac else "PAC なし(Windows のプロキシ設定または直接接続)"), flush=True)
            except Exception as e:
                print(f"ネットワーク:PAC の読み込みに失敗({e})", flush=True)
        try:
            r = _pac["session"].post(url, data=body, headers=headers, timeout=timeout)
            return r.status_code, r.text
        except requests.exceptions.Timeout:
            raise TimeoutError()
        except requests.exceptions.SSLError as e:
            raise RuntimeError(f"安全な接続(SSL)を確立できません:{e}")
        except requests.exceptions.ProxyError as e:
            raise RuntimeError(f"会社のプロキシ経由で接続できません:{e}")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Google に接続できません:{e}")
    req = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError) or "timed out" in str(e.reason):
            raise TimeoutError()
        raise RuntimeError(f"Google に接続できません:{e.reason}")


def gemini_call(key, text, timeout=300):
    body = json.dumps({"contents": [{"role": "user", "parts": [{"text": text}]}],
                       "generationConfig": {"temperature": 0.3}}).encode("utf-8")
    headers = {"Content-Type": "application/json", "x-goog-api-key": key}
    last = None
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            code, raw = http_post(url, body, headers, timeout)
        except TimeoutError:
            raise RuntimeError(f"Google から {timeout} 秒応答がありません。ネットワークの設定(プロキシ等)で Google AI への接続ができない可能性があります")
        if code != 200:
            msg = raw
            try:
                msg = json.loads(raw)["error"]["message"]
            except Exception:
                pass
            if code == 404:  # モデル名が古い → 次を試す
                last = msg
                continue
            if code in (400, 401, 403) and "key" in msg.lower():
                raise RuntimeError("API キーが正しくありません。設定し直してください")
            if code == 429:
                raise RuntimeError("今日の無料枠を使い切ったか、混み合っています。しばらくしてから、または ChatGPT にコピーしてください")
            raise RuntimeError(f"Gemini エラー({code}):{msg}")
        d = json.loads(raw)
        parts = ((d.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        out = "".join(p.get("text", "") for p in parts).strip()
        if not out:
            raise RuntimeError("Gemini から回答がありませんでした。ChatGPT にコピーしてお試しください")
        return out
    raise RuntimeError(f"使えるモデルが見つかりません:{last}")


def gemini_revise(m, instruction):
    key = load_config().get("gemini_key")
    if not key:
        raise RuntimeError("Gemini の API キーが設定されていません")
    return gemini_call(key, revise_text(m, instruction))


def revise_text(m, instruction, with_source=True):
    """with_source=False は ChatGPT 用:文字起こしを付けず短くする(同じチャットに貼れば前の内容も見てくれる)"""
    cur = (m.meta.get("summary") or "").strip()
    if not cur:
        raise RuntimeError("まだ議事録がありません。先に議事録を作ってください")
    rows = [
        "【議事録の修正のお願い】",
        "下の「今の議事録」を「修正の指示」どおりに直し、全文を出し直してください。",
        f"{SEP_SUMMARY} と {SEP_MINUTES} の区切り行は必ず残し、前置きや説明は書かないこと。指示されていない部分は変えないこと。",
        "同じ内容を2回書かないこと(背景・結論・情報共有・ToDo の間で繰り返さない)。",
    ]
    note = (m.meta.get("style_note") if m.meta.get("style_note") is not None else load_config().get("style_note", "")).strip()
    if note:
        rows += [f"いつもの書き方の希望も守ること:{note}"]
    if with_source:
        rows += ["事実の確認が必要なときは、最後の「会議の資料」(文字起こし)を参照すること。"]
    else:
        rows += ["事実の確認が必要なときは、このチャットの前のメッセージ(会議の文字起こし)を参照すること。"]
    rows += ["", "■ 修正の指示", instruction.strip(), "", "■ 今の議事録", cur]
    if with_source:
        rows += ["", "■ 会議の資料", claude_prompt(m, "ja")]
    return "\n".join(rows)


def claude_summarize(m, lang="ja"):
    """名前は元のまま。中身は Gemini で議事録を作る"""
    key = load_config().get("gemini_key")
    if not key:
        raise RuntimeError("Gemini の API キーが設定されていません")
    return gemini_call(key, claude_prompt(m, "ja"))


def parse_check(text):
    summary, minutes = split_summary(text)
    return {"summary": summary, "minutes": minutes}


def new_doc():
    from docx import Document
    from docx.oxml.ns import qn
    from docx.shared import Pt

    doc = Document()
    for name in ("Normal", "Heading 1", "Heading 2", "Heading 3", "Title"):
        st = doc.styles[name]
        st.font.name = "Yu Gothic"
        rpr = st.element.get_or_add_rPr()
        rpr.get_or_add_rFonts().set(qn("w:eastAsia"), "Yu Gothic")
    doc.styles["Normal"].font.size = Pt(10.5)
    return doc


def add_inline(p, text):
    # **粗體** 轉成粗體字
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if part:
            p.add_run(part).bold = i % 2 == 1


def add_markdown(doc, text):
    """把 Claude 回覆的簡易 Markdown 轉成 Word 段落"""
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.strip() in ("---", "***"):
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        s = line.strip()
        if m := re.match(r"^(#{1,4})\s+(.*)", s):
            level = min(len(m[1]), 3)
            doc.add_heading(m[2].replace("**", ""), 1 if level <= 2 else 2)
        elif m := re.match(r"^[-*・•]\s+(.*)", s):
            p = doc.add_paragraph(style="List Bullet 2" if indent >= 2 else "List Bullet")
            add_inline(p, m[1])
        elif s.strip("*::") in ("背景", "背景說明"):
            doc.add_heading(s.strip("*::"), 1)
        elif re.fullmatch(r"\*\*[^*]+\*\*[::]?", s):
            p = doc.add_paragraph()
            add_inline(p, s)
            p.paragraph_format.space_before = docx_pt(8)
            p.paragraph_format.space_after = docx_pt(2)
            p.paragraph_format.keep_with_next = True
        else:
            add_inline(doc.add_paragraph(), s)


def docx_pt(v):
    from docx.shared import Pt

    return Pt(v)


def add_info(doc, m, ja=True):
    meta = m.meta
    t = doc.add_table(rows=0, cols=2)
    t.style = "Light Grid Accent 1"
    rows = [
        ("日時" if ja else "日期", f"{meta.get('start', '')[:16].replace('T', ' ')}({fmt_duration(meta.get('duration', 0))})"),
        ("形式", ("オンライン" if ja else "線上會議") if meta.get("mode") == "online" else ("対面" if ja else "面對面")),
        ("出席者" if ja else "說話者", "、".join(speaker_names(m))),
    ]
    for k, v in rows:
        c = t.add_row().cells
        c[0].text, c[1].text = k, v
    set_widths(t, [3, 13])
    doc.add_paragraph()


def add_notes(doc, m, ja=True):
    notes = m.meta.get("notes", "").strip()
    after = m.meta.get("notes_after", "").strip()
    if notes:
        doc.add_heading("メモ" if ja else "筆記", 1)
        add_markdown(doc, notes)
    if after:
        doc.add_heading("メモ(会議後に追記)" if ja else "筆記(會後追加)", 1)
        add_markdown(doc, after)


def add_transcript(doc, m, ja=True):
    lines = m.sorted_lines()
    has_zh = any(r.get("zh") for r in lines)
    t = doc.add_table(rows=1, cols=4 if has_zh else 3)
    t.style = "Light List Accent 1"
    hdr = ["時刻", "話者", "原文"] + (["日本語訳"] if has_zh else [])
    for c, txt in zip(t.rows[0].cells, hdr):
        c.text = txt
    for r in lines:
        c = t.add_row().cells
        c[0].text = r.get("time", "")
        c[1].text = m.speaker_name(r["sid"])
        c[2].text = r.get("text", "")
        if has_zh:
            c[3].text = r.get("zh", "")
    set_widths(t, [2.3, 2.4, 5.65, 5.65] if has_zh else [2.3, 2.4, 11.3])


def set_widths(table, cms):
    from docx.shared import Cm

    table.autofit = False
    for col, w in zip(table._tbl.tblGrid.gridCol_lst, cms):
        col.w = Cm(w)
    for row in table.rows:
        for cell, w in zip(row.cells, cms):
            cell.width = Cm(w)


EXPORT_KINDS = {"transcript": "文字起こし", "summary": "要約", "minutes": "議事録"}


def export_docx(m, kind, lang="ja"):
    meta = m.meta
    ja = lang != "zh"
    summary, minutes = split_summary(meta.get("summary" if ja else "summary_zh", ""))
    if not ja and kind != "transcript" and not minutes:
        raise ValueError("中文版がまだできていません")
    yymmdd = meta.get("start", "")[2:10].replace("-", "")
    name = meta.get("name") or "会議"
    doc = new_doc()
    if kind == "minutes":
        if not minutes:
            raise ValueError("まだ ChatGPT の議事録が貼り付けられていません")
        doc.add_heading(f"{yymmdd} {name}" + ("" if ja else "(中文)"), 0)
        add_info(doc, m, ja)
        add_markdown(doc, minutes)
    elif kind == "summary":
        if not summary:
            raise ValueError("まだ ChatGPT の要約が貼り付けられていません")
        doc.add_heading(f"{yymmdd} {name}({'要約' if ja else '摘要'})", 0)
        add_info(doc, m, ja)
        add_markdown(doc, summary)
        add_notes(doc, m, ja)
    else:
        doc.add_heading(f"{yymmdd} {name}({'文字起こし' if ja else '逐字稿'})", 0)
        add_info(doc, m, ja)
        add_notes(doc, m, ja)
        doc.add_heading("文字起こし" if ja else "逐字稿", 1)
        add_transcript(doc, m, ja)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ---------- PDF ----------

PDF_FONTS = [  # Windows 標準の日本語フォント(最初に見つかったものを使う)
    (r"C:\Windows\Fonts\meiryo.ttc", r"C:\Windows\Fonts\meiryob.ttc"),
    (r"C:\Windows\Fonts\YuGothM.ttc", r"C:\Windows\Fonts\YuGothB.ttc"),
    (r"C:\Windows\Fonts\msgothic.ttc", None),
]


def pdf_fonts():
    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if "JP" in pdfmetrics.getRegisteredFontNames():
        return
    env = os.environ.get("MEETING_PDF_FONT")
    for reg, bold in ([(env, None)] if env else []) + PDF_FONTS:
        if reg and Path(reg).exists():
            pdfmetrics.registerFont(TTFont("JP", reg, subfontIndex=0))
            b = bold if bold and Path(bold).exists() else reg
            pdfmetrics.registerFont(TTFont("JP-B", b, subfontIndex=0))
            reg_jp, bold_jp = reg, b
            for it in (0, 1):
                addMapping("JP", 0, it, "JP")
                addMapping("JP", 1, it, "JP-B")
            break
    else:
        raise ValueError("日本語フォント(メイリオ等)が見つからないため PDF を作れません")
    # 中国語版用(繁体字):Microsoft JhengHei があれば使う。なければ日本語フォントで代用
    zh = [(r"C:\Windows\Fonts\msjh.ttc", r"C:\Windows\Fonts\msjhbd.ttc")]
    for reg, bold in zh:
        if Path(reg).exists():
            pdfmetrics.registerFont(TTFont("ZH", reg, subfontIndex=0))
            pdfmetrics.registerFont(TTFont("ZH-B", bold if Path(bold).exists() else reg, subfontIndex=0))
            break
    else:
        pdfmetrics.registerFont(TTFont("ZH", reg_jp, subfontIndex=0))
        pdfmetrics.registerFont(TTFont("ZH-B", bold_jp, subfontIndex=0))
    for it in (0, 1):
        addMapping("ZH", 0, it, "ZH")
        addMapping("ZH", 1, it, "ZH-B")


def export_pdf(m, lang="ja"):
    """要約+議事録の PDF(lang="zh" は中国語版)"""
    from html import escape

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    zh = lang == "zh"
    summary, minutes = split_summary(m.meta.get("summary_zh" if zh else "summary", ""))
    if not minutes:
        raise ValueError("中文版がまだできていません" if zh else "まだ議事録がありません")
    pdf_fonts()
    F, FB = ("ZH", "ZH-B") if zh else ("JP", "JP-B")
    base = dict(fontName=F, wordWrap="CJK", leading=16, fontSize=10.5)
    st = {
        "title": ParagraphStyle("t", **{**base, "fontName": FB, "fontSize": 17, "leading": 24, "spaceAfter": 8}),
        "h1": ParagraphStyle("h1", **{**base, "fontName": FB, "fontSize": 13.5, "leading": 20, "spaceBefore": 12,
                                      "spaceAfter": 4, "textColor": colors.HexColor("#1f3864")}),
        "h2": ParagraphStyle("h2", **{**base, "fontName": FB, "fontSize": 11.5, "leading": 18, "spaceBefore": 8, "spaceAfter": 2}),
        "p": ParagraphStyle("p", **{**base, "spaceAfter": 3}),
        "li": ParagraphStyle("li", **{**base, "leftIndent": 14, "bulletIndent": 4, "spaceAfter": 2}),
        "li2": ParagraphStyle("li2", **{**base, "leftIndent": 28, "bulletIndent": 18, "spaceAfter": 2}),
    }

    def inline(t):
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(t))

    def markdown(text):
        out = []
        for raw in text.splitlines():
            line = raw.rstrip()
            if not line.strip() or line.strip() in ("---", "***"):
                continue
            indent = len(line) - len(line.lstrip(" \t"))
            s = line.strip()
            if mm := re.match(r"^(#{1,4})\s+(.*)", s):
                out.append(Paragraph(inline(mm[2].replace("**", "")), st["h1" if len(mm[1]) <= 2 else "h2"]))
            elif mm := re.match(r"^[-*・•]\s+(.*)", s):
                out.append(Paragraph(inline(mm[1]), st["li2" if indent >= 2 else "li"], bulletText="•"))
            elif s.strip("*:：") == "背景":
                out.append(Paragraph("背景", st["h1"]))
            elif re.fullmatch(r"\*\*[^*]+\*\*[:：]?", s):
                out.append(Paragraph(inline(s), st["h2"]))
            else:
                out.append(Paragraph(inline(s), st["p"]))
        return out

    meta = m.meta
    yymmdd = meta.get("start", "")[2:10].replace("-", "")
    info = Table([["日時", f"{meta.get('start', '')[:16].replace('T', ' ')}({fmt_duration(meta.get('duration', 0))})"],
                  ["形式", "オンライン" if meta.get("mode") == "online" else "対面"],
                  ["出席者", "、".join(speaker_names(m))]],
                 colWidths=[2.6 * cm, 14 * cm])
    info.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), F, 10), ("FONT", (0, 0), (0, -1), FB, 10),
                              ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#dbe5f1")),
                              ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#8eaadb")),
                              ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story = [Paragraph(escape(f"{yymmdd} {meta.get('name') or '会議'}"), st["title"]), info, Spacer(1, 10)]
    if summary.strip():
        story += [Paragraph("要約", st["h1"])] + markdown(summary)
    story += [Paragraph("議事録", st["h1"])] + markdown(minutes)
    kept, i = [], 0
    while i < len(story):  # 見出しをページ末尾に残さない
        a = story[i]
        if isinstance(a, Paragraph) and a.style.name in ("h1", "h2") and i + 1 < len(story):
            kept.append(KeepTogether([a, story[i + 1]]))
            i += 2
        else:
            kept.append(a)
            i += 1

    def footer(c, d):
        c.setFont(F, 8)
        c.setFillColor(colors.grey)
        c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{d.page}")

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title=f"{yymmdd} {meta.get('name') or '会議'}").build(kept, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


# ---------- 網頁 ----------

def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False)
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def meeting(self, mid):
            if not re.fullmatch(r"\d{8}-\d{6}", mid) or not (REC_DIR / mid / "meta.json").exists():
                raise FileNotFoundError(mid)
            if app.meeting and app.meeting.id == mid:
                return app.meeting.refresh()
            return Meeting(REC_DIR / mid)

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            try:
                if path == "/":
                    return self.send(200, PAGE, "text/html; charset=utf-8")
                if path == "/api/state":
                    return self.send(200, app.state())
                if path == "/api/devices":
                    return self.send(200, app.devices())
                if path == "/api/meetings":
                    return self.send(200, list_meetings())
                if m := re.fullmatch(r"/api/meetings/([\d-]+)", path):
                    mt = self.meeting(m[1])
                    return self.send(200, {"id": mt.id, **mt.meta, "lines": mt.sorted_lines(),
                                           "summarizing": mt.id in app.summarizing, "checking": mt.id in app.checking,
                                           "shots": mt.summary()["shots"]})
                if m := re.fullmatch(r"/api/meetings/([\d-]+)/prompt", path):
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    return self.send(200, claude_prompt(self.meeting(m[1]), q.get("lang", ["ja"])[0], q.get("audio") == ["1"]),
                                     "text/plain; charset=utf-8")
                if m := re.fullmatch(r"/api/meetings/([\d-]+)/audio", path):
                    try:
                        data, mime = meeting_audio(self.meeting(m[1]))
                    except ValueError:
                        return self.send(413, {"error": f"{AUDIO_MAX_MIN} 分より長い会議は録音を付けずにまとめます"})
                    return self.send(200, data, mime)
                if path == "/api/version":
                    return self.send(200, {"build": BUILD, "url": UPDATE_BASE})
                if m := re.fullmatch(r"/api/meetings/([\d-]+)/docx/(\w+)", path):
                    mt = self.meeting(m[1])
                    kind = m[2] if m[2] in EXPORT_KINDS else "transcript"
                    lang = "zh" if "lang=zh" in self.path else "ja"
                    try:
                        data = export_docx(mt, kind, lang)
                    except ValueError as e:
                        return self.send(400, {"error": str(e)})
                    fname = f"{EXPORT_KINDS[kind]}{'_中文' if lang == 'zh' else ''}_{safe_name(mt.meta.get('name', ''))}_{mt.id[:8]}.docx"
                    (mt.dir / fname).write_bytes(data)
                    disp = "attachment; filename*=UTF-8''" + urllib.parse.quote(fname)
                    return self.send(200, data, "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                     {"Content-Disposition": disp})
                if m := re.fullmatch(r"/api/meetings/([\d-]+)/pdf", path):
                    mt = self.meeting(m[1])
                    lang = "zh" if "lang=zh" in self.path else "ja"
                    try:
                        data = export_pdf(mt, lang)
                    except ValueError as e:
                        return self.send(400, {"error": str(e)})
                    fname = f"議事録{'_中文' if lang == 'zh' else ''}_{safe_name(mt.meta.get('name', ''))}_{mt.id[:8]}.pdf"
                    (mt.dir / fname).write_bytes(data)
                    disp = "attachment; filename*=UTF-8''" + urllib.parse.quote(fname)
                    return self.send(200, data, "application/pdf", {"Content-Disposition": disp})
                if path == "/api/gemini":
                    key = load_config().get("gemini_key", "")
                    cfg = load_config()
                    return self.send(200, {"configured": bool(key), "url": GEMINI_KEY_URL, "key": key,
                                           "style": {"length": cfg.get("length", "short"), "style_note": cfg.get("style_note", "")}})
                if path == "/events":
                    return self.events()
                self.send(404, {"error": "not found"})
            except FileNotFoundError:
                self.send(404, {"error": "会議が見つかりません"})
            except Exception as e:
                self.send(500, {"error": str(e)})

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            try:
                if path == "/api/start":
                    d = self.body()
                    people = d.get("people")
                    app.start(d.get("name", ""), d.get("mode", "online"), d.get("mic"), d.get("speaker"),
                              d.get("my_mic") or None, int(people) if str(people or "").isdigit() else None,
                              False)
                    return self.send(200, {"ok": True})
                if path == "/api/pause":
                    app.pause(bool(self.body().get("paused")))
                    return self.send(200, {"ok": True, "paused": app.pause_evt.is_set()})
                if path == "/api/stop":
                    app.stop()
                    return self.send(200, {"ok": True, "id": app.meeting.id if app.meeting else None})
                if m := re.fullmatch(r"/api/meetings/([\d-]+)", path):
                    self.meeting(m[1])
                    app.update_meta(m[1], self.body())
                    return self.send(200, {"ok": True})
                if path == "/api/import":
                    meta = json.loads(urllib.parse.unquote(self.headers.get("X-Meta") or "%7B%7D"))
                    n = int(self.headers.get("Content-Length") or 0)
                    data = self.rfile.read(n)
                    people = meta.get("people")
                    mid = app.import_audio(data, meta.get("filename", ""), meta.get("name", ""), meta.get("notes", ""),
                                           int(people) if str(people or "").isdigit() else None,
                                           int(meta.get("lastmod") or 0))
                    return self.send(200, {"ok": True, "id": mid})
                if m := re.fullmatch(r"/api/meetings/([\d-]+)/summarize", path):
                    self.meeting(m[1])
                    if not gemini_configured():
                        return self.send(400, {"error": "先に Gemini の API キーを設定してください"})
                    threading.Thread(target=app.summarize, args=(m[1], "ja"), daemon=True).start()
                    return self.send(200, {"ok": True})
                if m := re.fullmatch(r"/api/meetings/([\d-]+)/revise", path):
                    self.meeting(m[1])
                    ins = str(self.body().get("instruction", "")).strip()
                    if not ins:
                        return self.send(400, {"error": "直してほしいことを書いてください"})
                    if not gemini_configured():
                        return self.send(400, {"error": "先に Gemini の API キーを設定してください"})
                    threading.Thread(target=app.revise, args=(m[1], ins), daemon=True).start()
                    return self.send(200, {"ok": True})
                if m := re.fullmatch(r"/api/meetings/([\d-]+)/clipwatch", path):
                    self.meeting(m[1])
                    if self.body().get("on"):
                        app.clip.start(m[1])
                    else:
                        app.clip.stop()
                    return self.send(200, {"ok": True, "supported": os.name == "nt"})
                if path == "/api/update":
                    d = self.body()
                    if app.status == "recording":
                        return self.send(400, {"error": "録音中は更新できません。会議が終わってから押してください"})
                    try:
                        install_update(str(d.get("code", "")), str(d.get("build", "")))
                    except (ValueError, SyntaxError) as e:
                        return self.send(400, {"error": str(e)})
                    threading.Thread(target=lambda: (time.sleep(1.0), os._exit(3)), daemon=True).start()
                    return self.send(200, {"ok": True})
                if m := re.fullmatch(r"/api/meetings/([\d-]+)/revise_prompt", path):
                    mt = self.meeting(m[1])
                    d = self.body()
                    try:
                        return self.send(200, revise_text(mt, str(d.get("instruction", "")), not d.get("short")),
                                         "text/plain; charset=utf-8")
                    except RuntimeError as e:
                        return self.send(400, {"error": str(e)})
                if path == "/api/llm":
                    try:
                        return self.send(200, {"text": app.llm(self.body())})
                    except Exception as e:
                        return self.send(400, {"error": str(e)})
                if path == "/api/gemini":
                    d = self.body()
                    key = str(d.get("key", "")).strip()
                    cfg = load_config()
                    if not key:
                        cfg.pop("gemini_key", None)
                        save_config(cfg)
                        return self.send(200, {"ok": True})
                    if d.get("verified"):  # ブラウザで接続テスト済み
                        cfg["gemini_key"] = key
                        save_config(cfg)
                        print("Gemini の API キーを保存しました", flush=True)
                        return self.send(200, {"ok": True})
                    print("Gemini 接続テスト中…", flush=True)
                    try:
                        gemini_call(key, "「OK」とだけ返してください。", timeout=20)
                    except RuntimeError as e:
                        print(f"Gemini 接続テスト失敗:{e}", flush=True)
                        return self.send(400, {"error": str(e)})
                    print("Gemini 接続テスト成功", flush=True)
                    cfg["gemini_key"] = key
                    save_config(cfg)
                    return self.send(200, {"ok": True})
                if m := re.fullmatch(r"/api/meetings/([\d-]+)/respeak", path):
                    self.meeting(m[1])
                    if app.status != "idle":
                        return self.send(400, {"error": "録音中、または別の会議を処理中です。しばらくしてからお試しください"})
                    p = str(self.body().get("people") or "")
                    threading.Thread(target=app.respeak, args=(m[1], int(p) if p.isdigit() else None),
                                     daemon=True).start()
                    return self.send(200, {"ok": True})
                if m := re.fullmatch(r"/api/meetings/([\d-]+)/reprocess", path):
                    self.meeting(m[1])
                    if app.status != "idle":
                        return self.send(400, {"error": "録音中、または別の会議を処理中です。しばらくしてからお試しください"})
                    threading.Thread(target=app.reprocess, args=(m[1],), daemon=True).start()
                    return self.send(200, {"ok": True})
                self.send(404, {"error": "not found"})
            except FileNotFoundError:
                self.send(404, {"error": "会議が見つかりません"})
            except Exception as e:
                self.send(500, {"error": str(e)})

        def events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q = app.hub.subscribe()
            try:
                while True:
                    try:
                        e = q.get(timeout=15)
                        self.wfile.write(f"data: {json.dumps(e, ensure_ascii=False)}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                app.hub.unsubscribe(q)

    return H


PAGE = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>会議字幕</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--text:#1d2329;--muted:#6b7480;--line:#e3e6ea;--accent:#2563eb;--rec:#dc2626;--go:#16a34a;--tr:#9a6700}
*{box-sizing:border-box}
body{margin:0;font-family:"Yu Gothic UI","Meiryo UI","Segoe UI",sans-serif;background:var(--bg);color:var(--text);font-size:15px}
button{font:inherit;cursor:pointer;border:1px solid var(--line);background:var(--card);border-radius:8px;padding:8px 14px}
button:disabled{opacity:.5;cursor:default}
input,select,textarea{font:inherit;border:1px solid var(--line);border-radius:8px;padding:9px 11px;background:#fff;width:100%}
.hidden{display:none!important}
.wrap{max-width:760px;margin:0 auto;padding:32px 16px}
h1{font-size:22px;margin:0 0 20px}
h2{font-size:16px;margin:28px 0 10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px}
.field{margin-bottom:16px}
.field label{display:block;font-size:13px;color:var(--muted);margin-bottom:6px}
.modes{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.mode{border:2px solid var(--line);border-radius:10px;padding:14px;cursor:pointer;background:#fff;text-align:left}
.mode b{display:block;font-size:16px;margin-bottom:4px}
.mode span{font-size:13px;color:var(--muted)}
.mode.on{border-color:var(--accent);background:#eff5ff}
details{margin-bottom:16px;font-size:14px}
summary{cursor:pointer;color:var(--muted)}
.big{width:100%;padding:16px;font-size:18px;font-weight:bold;border:none;border-radius:10px;color:#fff;background:var(--go)}
.big.stop{background:var(--rec)}
.note{font-size:13px;color:var(--muted);margin-top:8px}
.list .item{display:flex;align-items:center;gap:10px;padding:10px 0;border-bottom:1px solid var(--line)}
.list .item:last-child{border-bottom:none}
.list .info{flex:1;min-width:0}
.list .info b{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.list .info span{font-size:13px;color:var(--muted)}
.banner{padding:10px 14px;border-radius:8px;background:#fff7e6;border:1px solid #f5d38a;margin-bottom:16px;font-size:14px}
#rec{display:flex;flex-direction:column;height:100vh}
.bar{display:flex;align-items:center;gap:12px;padding:10px 16px;background:var(--card);border-bottom:1px solid var(--line)}
.dot{width:12px;height:12px;border-radius:50%;background:var(--rec);animation:blink 1.2s infinite}
.dot.off{background:#aaa;animation:none}
@keyframes blink{50%{opacity:.25}}
#timer{font-variant-numeric:tabular-nums;font-weight:bold;min-width:64px}
#recName{flex:1;border-color:transparent;font-weight:bold;font-size:16px}
#recName:hover,#recName:focus{border-color:var(--line)}
.bar .big{width:auto;padding:10px 22px;font-size:16px}
.main{flex:1;display:grid;grid-template-columns:1fr 340px;min-height:0}
#log{overflow-y:auto;padding:16px}
.row{margin:0 0 10px;padding:10px 14px;border-radius:10px;background:var(--card);border-left:5px solid #999}
.meta{font-size:12px;color:var(--muted);margin-bottom:2px}
.who{font-weight:bold;cursor:pointer;border-bottom:1px dashed currentColor}
.src{font-size:19px;line-height:1.5}
.zh{font-size:17px;line-height:1.5;color:var(--tr)}
.side{border-left:1px solid var(--line);background:var(--card);display:flex;flex-direction:column;padding:12px;gap:8px;min-height:0}
.side textarea{flex:1;resize:none;font-size:15px;line-height:1.6}
.side .small{font-size:12px;color:var(--muted)}
#spkList{display:flex;flex-wrap:wrap;gap:6px}
.chip{font-size:13px;padding:4px 10px;border-radius:999px;color:#fff;cursor:pointer;border:none}
.empty{color:var(--muted);text-align:center;margin-top:20vh}
dialog{border:none;border-radius:12px;padding:20px;width:min(600px,94vw)}
dialog::backdrop{background:rgba(0,0,0,.35)}
.step{margin:0 0 18px;font-size:14px;padding-left:38px;position:relative}
.step .num{position:absolute;left:0;top:-2px;width:26px;height:26px;border-radius:50%;background:var(--accent);color:#fff;font-weight:bold;display:flex;align-items:center;justify-content:center}
.step>b{font-size:15px}
.row2{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.preview{border:1px solid var(--line);border-radius:8px;padding:4px 14px;max-height:40vh;overflow:auto;font-size:14px;line-height:1.6;margin-top:8px}
.preview h4{margin:12px 0 4px;font-size:15px}.preview ul{margin:2px 0;padding-left:20px}.preview p{margin:6px 0}
.spin{display:inline-block;width:12px;height:12px;border:2px solid #ccc;border-top-color:#555;border-radius:50%;animation:sp 1s linear infinite;vertical-align:-2px;margin-right:6px}
@keyframes sp{to{transform:rotate(360deg)}}
.btns{display:flex;gap:8px;justify-content:flex-end;margin-top:14px;flex-wrap:wrap}
.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#1d2329;color:#fff;padding:10px 18px;border-radius:8px;font-size:14px;z-index:9}
/* 現在の要点(大) + 字幕(小) */
.main{grid-template-columns:var(--c1,minmax(0,1.35fr)) 7px minmax(160px,1fr) 7px var(--c3,280px)}
.gut{cursor:col-resize;background:var(--line);opacity:.5;touch-action:none}.gut:hover,.gut.on{opacity:1;background:var(--accent)}
#digest{overflow-y:auto;padding:16px 18px;border-right:1px solid var(--line);background:#fbfcfe}
.dg-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.dg-head h3{margin:0;font-size:18px;flex:1;white-space:nowrap}
.dg-head .small{font-size:12px;color:var(--muted)}
.dg-head select{width:auto;padding:5px 8px;font-size:13px}
.dg-head button{padding:5px 10px;font-size:13px}
.dg-box{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px;margin-bottom:12px}
.dg-box h4{margin:0 0 6px;font-size:15px;color:var(--muted)}
.dg-box ul{margin:0;padding-left:22px}
.dg-box li{font-size:19px;line-height:1.55;margin:2px 0}
.dg-box .none{font-size:14px;color:var(--muted)}
.dg-box.direction{border-left:6px solid var(--accent)}
.dg-box.agreed{border-left:6px solid var(--go)}
.dg-box.open{border-left:6px solid #d97706}
.dg-box.now li{font-size:16px}
.dg-hist summary{font-size:13px}
.dg-hist .item{font-size:13px;color:var(--muted);border-top:1px dashed var(--line);padding:6px 0}
#dgTurn{margin:0 0 12px;padding:10px 12px;border:2px solid #2563eb;border-radius:10px;background:#eff6ff}#dgTurn .tw{font-size:12px;color:#475569;margin-bottom:4px}#dgTurn .tc{font-size:19px;font-weight:bold;line-height:1.45}#dgTurn ul{margin:6px 0 0 18px;padding:0}#dgTurn li{margin:2px 0}
.outs{display:grid;grid-template-columns:repeat(2,minmax(0,200px));gap:6px 16px;margin:8px 0}.outs label{display:flex;gap:6px;align-items:center;font-size:15px;cursor:pointer}.outs input{width:auto}
.dg-wait{color:var(--muted);text-align:center;margin-top:12vh;line-height:1.8}
#log{padding:10px}
#log .row{padding:6px 10px;margin-bottom:6px;border-left-width:4px}
#log .src{font-size:14px;line-height:1.45}
#log .zh{font-size:13px;line-height:1.4}
#log .meta{font-size:11px}
#log .empty{margin-top:10vh;font-size:13px}

@media (max-width:800px){.main{grid-template-columns:1fr!important}.gut{display:none}.side{display:none}#digest{border-right:none;max-height:55vh}}
button:disabled{opacity:.4;cursor:not-allowed;filter:grayscale(1)}
button.copied{background:#1e8e3e!important;color:#fff!important;border-color:#1e8e3e!important}
.txnote{color:#b26a00;font-weight:600;margin:6px 0}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0}
.chips button{font-size:13px;padding:4px 10px;border-radius:999px;background:#eef2ff;border-color:#c7d2fe;color:#1e3a8a}
.opt{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:8px}
.opt select{width:auto}
.opt label{font-size:13px;color:var(--muted)}
#upBanner{background:#e8f7ee;border-color:#9fd8b3}
</style></head><body>

<!-- スタート画面 -->
<div id="home" class="wrap">
  <h1>会議字幕 <span id="buildTag" style="font-size:12px;color:var(--muted);font-weight:normal"></span></h1>
  <div id="upBanner" class="banner hidden"><b>新しいバージョンがあります</b> <span id="upNotes"></span>
    <div class="row2"><button id="upBtn" class="primary">⟳ 今すぐ更新</button></div>
    <div class="note">録音・議事録・設定はそのまま残ります。更新は 30 秒ほどで終わり、画面は自動で読み直されます。</div></div>
  <div id="loading" class="banner">モデルを読み込み中です。少しお待ちください(初回のみダウンロードに時間がかかります)…</div>
  <div class="card">
    <div class="field"><label>会議名</label><input id="name" placeholder="例:〇〇社 定例会"></div>
    <div class="field"><label>会議の形式</label>
      <div class="modes">
        <button class="mode on" data-mode="online"><b>オンライン会議</b><span>Teams・Zoom など。イヤホン推奨。自分と相手の声を分けて録音</span></button>
        <button class="mode" data-mode="room"><b>対面</b><span>マイクでその場の声を録音。話者を自動で区別</span></button>
      </div>
    </div>
    <div class="field hidden" id="peopleField"><label>出席人数(自分を含む・任意)</label><input id="people" type="number" min="2" max="20" placeholder="例:4" style="width:120px"></div>
    <details><summary>マイク・スピーカーの設定(通常は変更不要)</summary>
      <div class="field" style="margin-top:10px"><label>マイク</label><select id="mic"></select></div>
      <div class="field" id="spkField"><label>相手の声(イヤホン/スピーカー)</label><select id="speaker"></select></div>
      <div class="field hidden" id="myMicField"><label>自分専用マイク(任意:ピンマイク等。あると自分と他の人をきれいに分けられます)</label><select id="myMic"></select></div>
    </details>
    <button id="startBtn" class="big" disabled>● 録音開始</button>
    <div class="note" id="trNote"></div>
    <button id="importOpen" style="margin-top:10px">⇪ 録音ファイルを取り込む(スマホで録音した会議)</button>
  </div>
  <h2 style="display:flex;justify-content:space-between;align-items:baseline">過去の会議 </h2>
  <div class="card list" id="meetings"><div class="note">まだ記録がありません</div></div>
  <details class="card" id="gemBox" style="margin-top:18px">
    <summary style="cursor:pointer;font-weight:bold">☁ オプション:Gemini(Google の AI)で要点・議事録を自動作成 <span id="gemState" class="note" style="font-weight:normal"></span></summary>
    <div class="banner" style="margin:12px 0;background:#fff7ed;border-color:#fdba74">
      <b>⚠ 使う前にお読みください</b><br>
      ・会議の内容(文字・音声)が Google に送られます。<br>
      ・無料版では、送った内容が Google のサービス改善に使われることがあります。<br>
      ・社外秘の会議では使わないなど、会社のルールを確認してからお使いください。</div>
    <div class="note">使わない場合も、字幕・翻訳・会議中の要点まとめは、このパソコンの中だけで動きます(議事録は ChatGPT などにコピーして作成できます)。Gemini を設定すると、要点がより速く正確になり、うまくいかないときは自動でこのパソコンの AI に切り替わります。</div>
    <button class="openGem" style="margin-top:10px">内容を確認しました。Gemini を設定する</button>
  </details>
</div>

<!-- 録音画面 -->
<div id="rec" class="hidden">
  <div class="bar">
    <div class="dot"></div><div id="timer">0:00</div>
    <input id="recName" title="クリックして会議名を変更">
    <span id="pausedTag" class="hidden" style="font-weight:bold;color:#b45309">一時停止中(録音していません)</span>
    <button id="pauseBtn" class="big">⏸ 一時停止</button>
    <button id="stopBtn" class="big stop">■ 終了</button>
  </div>
  <div class="main">
    <div id="digest">
      <div class="dg-head"><h3 id="dgTitle">📌 現在の要点</h3>
        <span class="small" id="dgStatus"></span>
        <select id="dgLang" title="要点の言語"><option value="ja">日本語</option><option value="zh">中文</option><option value="en">English</option></select>
        <select id="dgInt" title="自動でまとめる間隔"><option value="120">2 分ごと</option><option value="180">3 分ごと</option><option value="300">5 分ごと</option><option value="600">10 分ごと</option><option value="0">自動なし</option></select>
        <button id="dgNow">今すぐまとめる</button></div>
      <div id="dgBody"></div>
    </div>
    <div class="gut" data-g="1" title="ドラッグで幅を変更(ダブルクリックで元に戻す)"></div>
    <div id="log"><div class="empty" id="emptyMsg">聞き取り中… 話し始めるとここに字幕が出ます</div></div>
    <div class="gut" data-g="2" title="ドラッグで幅を変更(ダブルクリックで元に戻す)"></div>
    <div class="side">
      <b>話者</b>
      <div id="spkList"></div>
      <div class="small">名前をクリックすると変更できます。同じ人が別人と判定されたら、同じ名前にすればまとまります。</div>
      <b style="margin-top:8px">メモ</b>
      <textarea id="notes" placeholder="会議中のメモはここへ(自動保存)"></textarea>
      <div class="small" id="saved"></div>
    </div>
  </div>
</div>

<!-- 終了後 -->
<dialog id="done">
  <h3 id="doneTitle" style="margin-top:0">会議が終了しました</h3>
  <div class="field"><label>会議名</label><input id="doneName"></div>
  <div id="aiStatus" class="note" style="margin:-6px 0 12px"></div>
  <div class="step"><span class="num">1</span><b>メモを確認</b> <span class="note">(任意・メモの内容が優先されます)</span>
    <textarea id="doneNotes" style="min-height:60px;margin-top:6px" placeholder="会議中のメモ"></textarea>
    <textarea id="doneAfter" style="min-height:50px;margin-top:6px" placeholder="会議後に思い出したこと・紙のメモなど"></textarea></div>
  <div class="step"><span class="num">2</span><b>AI にまとめてもらう</b>
    <div class="opt"><label>長さ</label><select id="lenSel"><option value="short">簡潔(おすすめ)</option><option value="normal">標準</option><option value="long">詳細</option></select>
      <label id="audWrap"><input type="checkbox" id="audChk" style="width:auto" checked> 録音も Gemini に聞かせる(字幕の聞き間違いを直す・少し時間がかかる)</label></div>
    <details id="styleBox" style="margin:8px 0 0"><summary>議事録の書き方の希望(任意・次の会議にも使われます)</summary>
      <textarea id="styleNote" style="min-height:56px;margin-top:6px" placeholder="例:ToDo は表にする / 先方の発言を中心に / 社外に出せる書き方で / 敬体(です・ます)で"></textarea></details>
    <div id="gemOn"><div class="row2"><button id="gemBtn" class="primary">✨ 自動でまとめる(Gemini)</button></div>
      <div class="note" id="gemNote">会議が終わると自動でまとめます。1〜2分で下に議事録が出ます。</div></div>
    <div id="gemOff" class="note">Gemini を設定すると、ボタン1つで自動でまとめられ、この画面の中で直せます(無料)。<a href="#" class="openGem">Gemini を設定する</a></div>
    <details id="gptBox" style="margin:8px 0 0"><summary>ChatGPT を使う(コピー&貼り付け)</summary>
      <div class="row2"><button id="copyBtn">📋 ChatGPT 用にコピー</button>
        <button id="openGpt">ChatGPT を開く</button></div>
      <div class="note">ChatGPT の入力欄で <b>Ctrl+V</b> → Enter。回答の下の「<b>コピー</b>」を押すと、<b>自動でこの画面に入ります</b>(入らないときは下の欄に Ctrl+V)。<br>
        「長すぎる」と言われたら <a href="#" id="saveTxt">テキストファイルで保存</a> して、ChatGPT に添付してください。</div></details></div>
  <div class="step"><span class="num">3</span><b>議事録</b>
    <div class="note">ここで直接直せます(自動保存)。</div>
    <textarea id="doneSummary" style="min-height:110px;margin-top:6px" placeholder="=====摘要===== から始まる回答がここに入ります"></textarea>
    <div id="aiPreview" class="preview hidden"></div>
    <div id="revBox" class="hidden" style="margin-top:10px"><label class="note" style="display:block;margin:0 0 4px">直したいところを書くか、ボタンを押してください(いくつでも組み合わせ可)</label>
      <div class="chips" id="revChips">
        <button data-t="もっと短く、要点だけに">もっと短く</button>
        <button data-t="同じ内容の繰り返しを削って、1回だけにする">重複を削る</button>
        <button data-t="ToDo を「担当者・内容・期限」がわかる形に直す">ToDo を明確に</button>
        <button data-t="社外の人に送っても問題ない書き方にする(社内向けの話・感想は削る)">社外向けに</button>
        <button data-t="です・ます調(敬体)にする">です・ます調</button>
        <button data-t="もう少し詳しく(数字・理由を補う)">もう少し詳しく</button>
      </div>
      <textarea id="revText" style="min-height:56px" placeholder="例:2. の「価格」の部分を削除 / 「王さん」は「汪さん」が正しい / 結論を一番上に"></textarea>
      <div class="row2"><button id="revBtn" class="primary">✏️ Gemini で直す</button>
        <button id="revGpt">📋 ChatGPT 用の修正依頼をコピー</button></div>
      <div class="note" id="revNote"></div></div></div>
  <div class="step"><span class="num">4</span><b>議事録を保存</b> <span class="note">(内容が OK になったら)</span>
    <div class="outs" id="outs">
      <label><input type="checkbox" data-o="pdf-ja"> PDF 日本語</label>
      <label><input type="checkbox" data-o="pdf-zh"> PDF 中文</label>
      <label><input type="checkbox" data-o="word-ja"> Word 日本語</label>
      <label><input type="checkbox" data-o="word-zh"> Word 中文</label></div>
    <div class="row2"><button id="outBtn" class="needs primary">⬇ 保存する</button> <span class="note" id="outNote"></span></div>
    <div class="note">中文は保存するときに AI が日本語の議事録を訳します。ファイルは会議フォルダ(recordings)にも保存されます。</div>
    <div class="note" style="margin-top:8px">文字起こし(全文・自動認識のまま) <button data-kind="transcript" style="font-size:12px;padding:3px 10px">Word で保存</button></div>
    <details style="margin-top:6px"><summary class="note">その他のファイル</summary><div class="row2">
      <button data-kind="summary" class="needs">Word 要約</button></div></details></div>
  <details><summary>話者の分け方・字幕がおかしいとき</summary>
    <div style="margin-top:8px" class="opt"><button id="spkBtn">話者だけ分け直す</button> <label>出席人数(自分を含む)</label><input id="spkN" type="number" min="1" max="20" style="width:80px" placeholder="自動">
      <span class="note" style="margin:0">字幕はそのまま、声の特徴で会議全体をまとめて分け直します(数分)</span></div>
    <div style="margin-top:8px"><button id="reBtn">録音をもう一度認識し直す</button> <span class="note" id="reNote">今の文字起こしは上書きされます(古いものは transcript_prev.jsonl に保存)</span></div>
  </details>
  <div class="btns"><button id="closeBtn">閉じる</button></div>
</dialog>

<dialog id="gemDlg">
  <h3 style="margin-top:0">Gemini の設定(無料・最初の1回だけ)</h3>
  <ol class="note" style="padding-left:18px;line-height:1.9">
    <li><a href="#" id="gemOpen">ここをクリック</a>して Google アカウント(Gmail)でログイン</li>
    <li>「API キーを作成」(Create API key)を押す</li>
    <li>表示されたキー(英数字の長い文字列)をコピーして、下に貼り付け</li>
  </ol>
  <div class="field"><label>API キー</label><input id="gemKey" type="password" placeholder="ここにキーを貼り付け"></div>
  <div class="note">キーはこのパソコンの config.json にだけ保存されます。他の人に教えないでください。<br>※ 無料版では、送った内容が Google のサービス改善に使われることがあります。社外秘の会議では使わないなど、社内ルールに合わせてお使いください。</div>
  <div id="gemMsg" class="hidden" style="margin-top:12px;padding:10px 12px;border-radius:8px;font-size:14px"></div>
  <div class="btns"><button id="gemDel" class="hidden">キーを削除</button><button id="gemClose">閉じる</button><button id="gemSave" class="primary">保存してテスト</button></div>
</dialog>

<dialog id="importDlg">
  <h3 style="margin-top:0">録音ファイルを取り込む</h3>
  <div class="note">スマホのボイスメモなど(m4a・mp3・wav)。取り込むと字幕に変換され、あとは通常の会議と同じ手順です。</div>
  <div class="field" style="margin-top:10px"><label>音声ファイル</label><input id="impFile" type="file" accept="audio/*,.m4a,.mp3,.wav,.aac,.mp4"></div>
  <div class="field"><label>会議名</label><input id="impName" placeholder="例:〇〇社 打ち合わせ"></div>
  <div class="field"><label>メモ(任意)</label><textarea id="impNotes" style="min-height:80px"></textarea></div>
  <div class="field"><label>出席人数(自分を含む・任意)</label><input id="impPeople" type="number" min="2" max="20" style="width:120px"></div>
  <div class="note" id="impNote"></div>
  <div class="btns"><button id="impClose">キャンセル</button><button id="impGo" class="primary">取り込む</button></div>
</dialog>

<script>
const $=s=>document.querySelector(s);
const COLORS=['#2563eb','#db2777','#7c3aed','#ea580c','#0891b2','#65a30d','#b45309','#475569'];
let mode='online', state={}, cur=null, lines={}, speakers={}, t0=0, timerId=null, colorMap={};
function toast(msg){const d=document.createElement('div');d.className='toast';d.textContent=msg;document.body.appendChild(d);setTimeout(()=>d.remove(),2800)}
async function api(path,body){const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const t=await r.text();let j;try{j=JSON.parse(t)}catch{j=t}if(!r.ok)throw new Error(j.error||t);return j}
function fmt(s){s=Math.floor(s);const h=Math.floor(s/3600),m=Math.floor(s/60)%60,x=String(s%60).padStart(2,'0');return h?`${h}:${String(m).padStart(2,'0')}:${x}`:`${m}:${x}`}
function defName(sid){if(sid==='me')return'自分';let m=sid.match(/^(them|p)(\d+)$/);return m?(m[1]==='them'?'相手':'話者')+m[2]:sid}
function spkName(sid){return speakers[sid]||defName(sid)}
function color(sid){if(sid==='me')return'#16a34a';if(!(sid in colorMap))colorMap[sid]=COLORS[Object.keys(colorMap).length%COLORS.length];return colorMap[sid]}
const LANG={ja:'日',zh:'中',yue:'広',en:'英',ko:'韓'};
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}

// ---- スタート画面 ----
document.querySelectorAll('.mode').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.mode').forEach(x=>x.classList.toggle('on',x===b));mode=b.dataset.mode;
  $('#spkField').classList.toggle('hidden',mode!=='online');
  $('#myMicField').classList.toggle('hidden',mode!=='room');$('#peopleField').classList.toggle('hidden',mode!=='room');
  if(mode==='room'){const o=[...$('#mic').options].find(o=>/配列|array|陣列/i.test(o.value));if(o)$('#mic').value=o.value}
  else if(devs)$('#mic').value='';
});
let devs=null;
async function loadDevices(){try{devs=await api('/api/devices');
  $('#mic').innerHTML=`<option value="">自動(現在:${esc(devs.default_mic)})</option>`+devs.mics.map(n=>`<option>${esc(n)}</option>`).join('');$('#mic').value='';
  $('#speaker').innerHTML=`<option value="">自動(現在:${esc(devs.default_speaker)})</option>`+devs.speakers.map(n=>`<option>${esc(n)}</option>`).join('');$('#speaker').value='';
  $('#myMic').innerHTML='<option value="">使わない</option>'+devs.mics.map(n=>`<option>${esc(n)}</option>`).join('');
}catch(e){toast('マイク・スピーカーを読み込めません:'+e.message)}}
async function loadMeetings(){const list=await api('/api/meetings');
  $('#meetings').innerHTML=list.length?list.map(m=>`<div class="item"><div class="info"><b>${esc(m.name||'(名前なし)')}</b><span>${esc(m.date)} ・ ${m.minutes} 分 ・ ${m.lines} 発言</span></div><button data-open="${m.id}">開く / 出力</button></div>`).join(''):'<div class="note">まだ記録がありません</div>';
  document.querySelectorAll('[data-open]').forEach(b=>b.onclick=()=>openDone(b.dataset.open));}
$('#importOpen').onclick=()=>{$('#impFile').value='';$('#impName').value='';$('#impNotes').value='';$('#impPeople').value='';$('#impNote').textContent='';$('#impGo').disabled=false;$('#importDlg').showModal()};
$('#impClose').onclick=()=>$('#importDlg').close();
$('#impFile').onchange=()=>{const f=$('#impFile').files[0];if(f&&!$('#impName').value)$('#impName').value=f.name.replace(/\.[^.]+$/,'')};
let importing=null;
$('#impGo').onclick=async()=>{const f=$('#impFile').files[0];if(!f){toast('音声ファイルを選んでください');return}
  $('#impGo').disabled=true;$('#impNote').innerHTML='<span class="spin"></span>アップロード・変換中…(1時間の録音で約1分)';
  const meta={filename:f.name,name:$('#impName').value,notes:$('#impNotes').value,people:$('#impPeople').value,lastmod:f.lastModified||0};
  try{const r=await fetch('/api/import',{method:'POST',headers:{'X-Meta':encodeURIComponent(JSON.stringify(meta))},body:f});
    const t=await r.text();let j;try{j=JSON.parse(t)}catch{j={error:t}}if(!r.ok)throw new Error(j.error||t);
    $('#importDlg').close();loadMeetings();importing=j.id;await openDone(j.id,true);setTx(true);$('#aiStatus').innerHTML='<span class="spin"></span>字幕に変換中…';toast('取り込み完了。字幕に変換しています')}
  catch(e){$('#impNote').textContent='取り込みに失敗しました:'+e.message;$('#impGo').disabled=false}};
$('#startBtn').onclick=async()=>{$('#startBtn').disabled=true;
  if(mode==='room'&&$('#myMic').value&&$('#myMic').value===$('#mic').value){toast('自分専用マイクと会場マイクは別のものを選んでください');$('#startBtn').disabled=false;return}
  try{await api('/api/start',{name:$('#name').value,mode,mic:$('#mic').value,speaker:$('#speaker').value,
    my_mic:mode==='room'?$('#myMic').value:'',people:mode==='room'?$('#people').value:''});await refresh()}
  catch(e){toast('開始できません:'+e.message);$('#startBtn').disabled=false}};

// ---- 録音画面 ----
function showRec(m){
  cur=m.id;speakers=m.speakers||{};lines={};colorMap={};$('#log').innerHTML='<div class="empty" id="emptyMsg">聞き取り中… 話し始めるとここに字幕が出ます</div>';
  $('#recName').value=m.name;$('#notes').value=m.notes||'';(m.lines||[]).forEach(addLine);
  t0=Date.now()-(state.elapsed||0)*1000;clearInterval(timerId);timerId=setInterval(()=>$('#timer').textContent=fmt((Date.now()-t0)/1000),500);
  $('#home').classList.add('hidden');$('#rec').classList.remove('hidden');renderSpk();setPaused(!!state.paused);
  digests=m.digests||[];turns=m.turns||[];turnSt=null;turnRun='';lastLineAt=0;dgLastId=digests.length?(digests[digests.length-1].upto||0):0;dgLast=Date.now();dgState='';
  api('/api/gemini').then(g=>{gem=g;renderDigest()}).catch(()=>{});renderDigest();
}
// ---- 現在の要点:数分ごとに Gemini で「会議の流れ・合意事項・未決事項・今の話題」をまとめる ----
const LS={get(k,d){try{const v=localStorage.getItem(k);return v===null?d:v}catch(e){return d}},set(k,v){try{localStorage.setItem(k,v)}catch(e){}}};
let digests=[],dgState='',dgMsg='',dgLastId=0,dgLast=Date.now(),dgPaused=false,dgLang=LS.get('dgLang','ja'),dgInt=+LS.get('dgInt','180');
const DG={zh:{title:'📌 目前重點',direction:'🧭 會議走向',agreed:'✅ 目前共識',open:'❓ 還沒結論・要確認',now:'💬 現在在談',none:'(還沒有)',hist:'之前的整理',
    wait:'開始說話後,每隔幾分鐘會在這裡整理「會議走向」和「目前共識」。',upd:'更新',next:'下次',run:'整理中…',tr:'翻譯中…',name:'台灣繁體中文'},
  ja:{title:'📌 現在の要点',direction:'🧭 会議の流れ',agreed:'✅ 合意事項',open:'❓ 未決・要確認',now:'💬 今の話題',none:'(まだなし)',hist:'過去の要点',
    wait:'話し始めると、数分ごとにここに「会議の流れ」と「合意事項」をまとめます。<br>すぐ見たいときは「今すぐまとめる」を押してください。',upd:'更新',next:'次回',run:'まとめ中…',tr:'翻訳中…',name:'日本語'},
  en:{title:'📌 Key points so far',direction:'🧭 Direction',agreed:'✅ Agreed',open:'❓ Open / to confirm',now:'💬 Now discussing',none:'(none yet)',hist:'Earlier summaries',
    wait:'Once people start talking, the direction and agreements will be summarized here every few minutes.',upd:'updated',next:'next',run:'Summarizing…',tr:'Translating…',name:'English'}};
const DG_KEYS=['direction','agreed','open','now'];
const DG_SCHEMA={type:'OBJECT',properties:Object.fromEntries(DG_KEYS.map(k=>[k,{type:'ARRAY',items:{type:'STRING'}}])),required:DG_KEYS};
function dgData(it,lang){return it.lang===lang?it.data:(it.tr&&it.tr[lang])||null}
function hm(d){return d.toTimeString().slice(0,5)}
// ---- 3 つの欄の幅:境目の線をドラッグ(幅はこのパソコンに保存) ----
(function(){const mn=document.querySelector('#rec .main');let c1=0,c3=0;
  try{c1=+localStorage.getItem('colW1')||0;c3=+localStorage.getItem('colW3')||0}catch(e){}
  const apply=()=>{mn.style.setProperty('--c1',c1?c1+'px':'');mn.style.setProperty('--c3',c3?c3+'px':'');if(!c1)mn.style.removeProperty('--c1');if(!c3)mn.style.removeProperty('--c3')};
  const save=()=>{try{localStorage.setItem('colW1',c1||'');localStorage.setItem('colW3',c3||'')}catch(e){}};apply();
  document.querySelectorAll('#rec .gut').forEach(g=>{
    g.onpointerdown=e=>{e.preventDefault();g.setPointerCapture(e.pointerId);g.classList.add('on');
      const r=mn.getBoundingClientRect(),w1=document.querySelector('#digest').getBoundingClientRect().width,w3=document.querySelector('#rec .side').getBoundingClientRect().width;
      g.onpointermove=ev=>{if(g.dataset.g==='1')c1=Math.round(Math.max(200,Math.min(r.width-w3-14-160,ev.clientX-r.left)));
        else{c3=Math.round(Math.max(140,Math.min(r.width-w1-14-160,r.right-ev.clientX)));if(!c1)c1=Math.round(w1)}apply()}};
    g.onpointerup=g.onpointercancel=e=>{g.onpointermove=null;g.classList.remove('on');save()};
    g.ondblclick=()=>{c1=0;c3=0;apply();save()}})})();
function renderDigest(){const L=DG[dgLang]||DG.ja;$('#dgLang').value=dgLang;$('#dgInt').value=String(dgInt);$('#dgTitle').textContent=L.title;
  const last=digests[digests.length-1];
  let st=dgState==='running'?'<span class="spin"></span>'+L.run:dgState==='translating'?'<span class="spin"></span>'+L.tr:dgState==='error'?esc(dgMsg):'';
  if(!st&&last)st=`${last.time} ${L.upd}${last.via==='cloud'?' ☁':last.via==='local'?' 💻':''}`+(dgInt?`・${L.next} ${hm(new Date(dgLast+dgInt*1000))}`:'');
  $('#dgStatus').innerHTML=st;$('#dgNow').disabled=dgState==='running'||(!gem.key&&!localReady());
  if(!gem.key&&!localReady()){$('#dgBody').innerHTML=turnHtml()+'<div class="dg-wait">'+(state.local_ai==='failed'?'ローカル AI を起動できませんでした。':'<span class="spin"></span>このパソコンの AI を準備中です(初回のみ約 2GB ダウンロード)。')+'<br>準備ができると、数分ごとにここへ「会議の流れ」と「合意事項」をまとめます。<br>Gemini(任意)を設定すると、より速く正確になります:<a href="#" class="openGem">Gemini を設定する</a></div>';
    $('#dgBody').querySelectorAll('.openGem').forEach(a=>a.onclick=e=>{e.preventDefault();document.querySelector('.openGem:not(#dgBody .openGem)')?.click()});return}
  if(!last){$('#dgBody').innerHTML=turnHtml()+`<div class="dg-wait">${L.wait}</div>`;return}
  const d=dgData(last,dgLang)||last.data;
  const box=k=>`<div class="dg-box ${k}"><h4>${L[k]}</h4>${(d[k]||[]).length?'<ul>'+d[k].map(x=>'<li>'+esc(x)+'</li>').join('')+'</ul>':'<div class="none">'+L.none+'</div>'}</div>`;
  const hist=digests.slice(0,-1).reverse().map(it=>{const x=dgData(it,dgLang)||it.data;return `<div class="item"><b>${it.time}</b> ${esc((x.direction||[]).join(' / '))}${(x.agreed||[]).length?'<br>✅ '+esc(x.agreed.join(' / ')):''}</div>`}).join('');
  $('#dgBody').innerHTML=turnHtml()+DG_KEYS.map(box).join('')+(hist?`<details class="dg-hist"><summary>${L.hist}(${digests.length-1})</summary>${hist}</details>`:'');
}
// ---- AI の振り分け:Gemini が設定済みならまず Gemini、失敗・未設定ならこのパソコンのローカル AI ----
let gemDown=0,lastLineAt=0,turns=[],turnRun='',turnSt=null,turnBusy=false;
function localReady(){return state.local_ai==='ready'}
async function aiCall(prompt,schema,ms,localPrompt,maxTok){
  if(gem.key&&Date.now()>gemDown){try{return {text:await gemCall(gem.key,prompt,ms,null,schema),via:'cloud'}}
    catch(e){gemDown=Date.now()+(/無料枠|429/.test(e.message)?1800e3:120e3);console.log('Gemini 失敗 → ローカル',e.message)}}
  const r=await api('/api/llm',{user:localPrompt||prompt,schema,max_tokens:maxTok||450,system:schema?undefined:'指示どおりに出力してください。'});return {text:r.text,via:'local'}}
// ---- さっきの発言:誰かが長く話して 3 秒止まったら、その段落を「結論 → 背景」でまとめる(通訳用) ----
const TURN_MIN={zh:60,yue:60,ja:80,en:120};
const TURN_SCHEMA={type:'OBJECT',properties:{conclusion:{type:'STRING'},background:{type:'ARRAY',items:{type:'STRING'}}},required:['conclusion','background']};
function turnHtml(){const t=turns[turns.length-1];if(!t&&!turnRun)return'';
  const run=turnRun?`<div class="tw"><span class="spin"></span>${esc(turnRun)} さんの発言をまとめ中…</div>`:'';if(!t)return `<div id="dgTurn">${run}</div>`;
  const lab=(t.lang==='ja'?'→ 日本語の要点':'→ 中文重點')+(t.via==='cloud'?' ・☁':' ・💻');
  return `<div id="dgTurn">${run}<div class="tw">🗣 ${esc(t.time)} ${esc(t.who)} さっきの発言 ${lab}</div><div class="tc">${esc(t.data.conclusion||'')}</div>${(t.data.background||[]).length?'<ul>'+t.data.background.slice(0,3).map(x=>'<li>'+esc(x)+'</li>').join('')+'</ul>':''}</div>`}
async function turnTick(){if(!cur||dgPaused||turnBusy||!lastLineAt||Date.now()-lastLineAt<3000)return;if(!gem.key&&!localReady())return;
  const all=Object.values(lines).filter(l=>l.text).sort((a,b)=>a.id-b.id);const run=[];
  for(let i=all.length-1;i>=0;i--){const l=all[i];if(run.length&&(l.sid!==run[0].sid||l.lang!==run[0].lang))break;run.unshift(l);if(run.length>60)break}
  if(!run.length)return;const lang=run[0].lang,n=run.reduce((a,l)=>a+l.text.length,0);if(!TURN_MIN[lang]||n<TURN_MIN[lang])return;
  const k0=run[0].id,k1=run[run.length-1].id;if(turnSt&&turnSt[0]===k0&&(turnSt[1]===k1||n<turnSt[2]+30))return;
  turnSt=[k0,k1,n];const id=cur,who=spkName(run[0].sid),tl=(lang==='zh'||lang==='yue')?'ja':'zh';
  const text=run.map(l=>l.text).join('').slice(-1500),tn=tl==='ja'?'日本語':'台灣繁體中文';
  const p=`あなたは会議通訳を手伝います。次は ${who} さんが今話し終えた発言です(音声認識の結果で誤字あり)。通訳がこれから相手側に伝えます。\n`+
    `${tn}で JSON を出してください。conclusion = この発言の結論・一番伝えたいこと(1 文)。background = 結論を支える理由・背景・数字・条件(最大 3 つ、各 1 文、大事な順)。`+
    `結論が先、背景はあと。通訳がそのまま話せる自然な文に。数字・日付・社名・型番は残す。発言にないことは書かない。\n\n発言:\n`+text;
  turnBusy=true;turnRun=who;renderDigest();
  try{const r=await aiCall(p,TURN_SCHEMA,15000,null,220);const d=dgJson(r.text);if(cur!==id)return;
    const it={time:(run[0].time||'').slice(0,5),who,src:lang,lang:tl,via:r.via,from:k0,upto:k1,
      data:{conclusion:String(d.conclusion||'').trim(),background:(d.background||[]).map(x=>String(x).trim()).filter(Boolean).slice(0,3)}};
    if(turns.length&&turns[turns.length-1].from===k0){it.n=turns[turns.length-1].n;turns[turns.length-1]=it}else{it.n=turns.length+1;turns.push(it)}
    api('/api/meetings/'+id,{turns}).catch(()=>{})}
  catch(e){console.log('さっきの発言のまとめに失敗',e.message)}
  finally{turnBusy=false;turnRun='';renderDigest()}}
setInterval(turnTick,1000);
function dgClean(o){const r={};DG_KEYS.forEach(k=>{const v=o&&o[k];r[k]=(Array.isArray(v)?v:v?[v]:[]).map(x=>String(x).trim()).filter(Boolean)});return r}
function dgJson(t){return JSON.parse(t.replace(/^```(json)?|```$/g,'').trim())}
async function runDigest(force){if(!cur||(!gem.key&&!localReady())||dgState==='running')return;
  const id=cur,news=Object.values(lines).filter(l=>l.id>dgLastId&&l.text).sort((a,b)=>a.id-b.id);
  const prev=digests.length?digests[digests.length-1].data:null;dgLast=Date.now();
  if(!news.length&&!force)return;
  const tx0=news.map(l=>`[${(l.time||'').slice(0,5)}] ${spkName(l.sid)}:${l.text}`).join('\n');let tx=tx0.slice(-6000);
  const L=DG[dgLang]||DG.ja;dgState='running';renderDigest();
  const p='あなたは会議の記録係です。「前回の要点」と「最新の文字起こし」(音声認識の結果で誤字あり)から、更新した要点を JSON で出してください。\n'+
    '- direction: 会議の流れ。1〜2 文で、何の話から何の話に移り、今どこへ向かっているか\n'+
    '- agreed: ここまでに合意・決定したこと。前回の分は覆らない限り残し、新しいものを足す。数字・日付・担当を具体的に\n'+
    '- open: まだ結論が出ていない・確認や回答が必要なこと。解決したものは消す\n'+
    '- now: 今話している話題(1 つ)\n'+
    `各項目は短い 1 文。文字起こしにないことは書かない。なければ空の配列。すべて${L.name}で書く。\n\n`+
    `会議名:${$('#recName').value}\n`+($('#notes').value.trim()?`メモ:${$('#notes').value.trim().slice(-600)}\n`:'')+
    (prev?'前回の要点:'+JSON.stringify(prev)+'\n':'')+'最新の文字起こし:\n';
  try{const r=await aiCall(p+(tx||'(この間、新しい発言なし)'),DG_SCHEMA,60000,p+(tx0.slice(-2000)||'(この間、新しい発言なし)'));const data=dgClean(dgJson(r.text));if(cur!==id)return;
    if(news.length)dgLastId=news[news.length-1].id;
    const it={n:digests.length+1,time:hm(new Date()),t:Math.round((Date.now()-t0)/1000),lang:dgLang,upto:dgLastId,data,via:r.via};digests.push(it);dgState='';api('/api/meetings/'+id,{digests}).catch(()=>{})}
  catch(e){dgState='error';dgMsg='要点のまとめに失敗:'+e.message}
  dgLast=Date.now();renderDigest()}
async function dgTranslate(){const it=digests[digests.length-1];if(!it||(!gem.key&&!localReady())||dgData(it,dgLang)||dgState==='running')return renderDigest();
  const id=cur,lang=dgLang;dgState='translating';renderDigest();
  try{const out=dgClean(dgJson((await aiCall(`次の JSON の各文字列を自然な${DG[lang].name}に訳してください。構造と項目数はそのまま、JSON だけを出力。\n`+JSON.stringify(it.data),DG_SCHEMA,30000)).text));
    (it.tr=it.tr||{})[lang]=out;dgState='';if(cur===id)api('/api/meetings/'+id,{digests}).catch(()=>{})}
  catch(e){dgState='error';dgMsg='翻訳に失敗:'+e.message}renderDigest()}
$('#dgLang').onchange=()=>{dgLang=$('#dgLang').value;LS.set('dgLang',dgLang);dgTranslate()};
$('#dgInt').onchange=()=>{dgInt=+$('#dgInt').value;LS.set('dgInt',dgInt);dgLast=Date.now();renderDigest()};
$('#dgNow').onclick=()=>runDigest(true);
setInterval(()=>{if(cur&&!dgPaused&&dgInt>0&&Date.now()-dgLast>=dgInt*1000)runDigest(false)},5000);
function addLine(r){
  const old=lines[r.id]||{};const L=lines[r.id]={...old,...r};
  let el=document.getElementById('L'+r.id);
  if(!el){$('#emptyMsg')?.remove();el=document.createElement('div');el.className='row';el.id='L'+r.id;
    el.innerHTML='<div class="meta"><span class="who"></span> ・ <span class="tm"></span> ・ <span class="lg"></span></div><div class="src"></div><div class="zh"></div>';
    const near=$('#log').scrollTop+$('#log').clientHeight>$('#log').scrollHeight-200;$('#log').appendChild(el);if(near)$('#log').scrollTop=$('#log').scrollHeight;
    el.querySelector('.who').onclick=()=>rename(L.sid)}
  if(L.sid){el.style.borderLeftColor=color(L.sid);el.querySelector('.who').textContent=spkName(L.sid);el.querySelector('.who').style.color=color(L.sid)}
  el.querySelector('.tm').textContent=L.time||'';el.querySelector('.lg').textContent=LANG[L.lang]||L.lang||'';
  el.querySelector('.src').textContent=L.text||'';el.querySelector('.zh').textContent=L.zh||'';
  if(r.sid&&!old.sid)renderSpk();
}
function renderSpk(){const ids=[...new Set(Object.values(lines).map(l=>l.sid).filter(Boolean))];
  $('#spkList').innerHTML=ids.length?ids.map(s=>`<button class="chip" style="background:${color(s)}" data-sid="${s}">${esc(spkName(s))}</button>`).join(''):'<span class="small">まだ誰も話していません</span>';
  document.querySelectorAll('#spkList .chip').forEach(b=>b.onclick=()=>rename(b.dataset.sid));}
async function rename(sid){const n=prompt('「'+spkName(sid)+'」の新しい名前:',spkName(sid));if(n===null)return;
  speakers={...speakers,[sid]:n.trim()};await api('/api/meetings/'+cur,{speakers});refreshNames()}
function refreshNames(){Object.values(lines).forEach(l=>addLine({id:l.id}));renderSpk()}
let saveT=null;
function autosave(){clearTimeout(saveT);$('#saved').textContent='…';saveT=setTimeout(async()=>{
  try{await api('/api/meetings/'+cur,{name:$('#recName').value,notes:$('#notes').value});$('#saved').textContent='保存しました '+new Date().toLocaleTimeString()}
  catch(e){$('#saved').textContent='保存できません:'+e.message}},800)}
$('#notes').oninput=autosave;$('#recName').oninput=autosave;
function setPaused(p){dgPaused=p;$('#pauseBtn').textContent=p?'▶ 再開':'⏸ 一時停止';$('#pausedTag').classList.toggle('hidden',!p);document.querySelector('#rec .dot').classList.toggle('off',p)}
$('#pauseBtn').onclick=async()=>{const want=!$('#pauseBtn').textContent.includes('再開');$('#pauseBtn').disabled=true;
  try{const r=await api('/api/pause',{paused:want});setPaused(r.paused);toast(r.paused?'一時停止しました。「再開」で同じ会議の続きを録音します':'録音を再開しました')}
  catch(e){toast('失敗しました:'+e.message)}$('#pauseBtn').disabled=false};
$('#stopBtn').onclick=async()=>{if(!confirm('会議を終了しますか?'))return;
  $('#stopBtn').disabled=true;$('#stopBtn').textContent='終了処理中…';clearTimeout(saveT);
  try{await api('/api/meetings/'+cur,{name:$('#recName').value,notes:$('#notes').value});const r=await api('/api/stop',{});
    clearInterval(timerId);$('#rec').classList.add('hidden');$('#home').classList.remove('hidden');
    $('#stopBtn').disabled=false;$('#stopBtn').textContent='■ 終了';await refresh();openDone(r.id,true)}
  catch(e){toast('終了できません:'+e.message);$('#stopBtn').disabled=false;$('#stopBtn').textContent='■ 終了'}};

// ---- 終了後 ----
let doneId=null,sumDirty=false,notesT=null,sumT=null;
async function openDone(id,justEnded){doneId=id;const m=await api('/api/meetings/'+id);
  $('#doneTitle').textContent=m.imported&&justEnded?'録音を取り込みました':justEnded?'会議が終了しました':'会議の記録';
  $('#doneName').value=m.name||'';$('#doneSummary').value=m.summary||'';$('#doneNotes').value=m.notes||'';$('#doneAfter').value=m.notes_after||'';
  $('#aiStatus').textContent='';sumDirty=false;gem=await api('/api/gemini');const cfg=gem.style||{};
  $('#lenSel').value=m.length||cfg.length||'short';$('#styleNote').value=m.style_note!=null?m.style_note:(cfg.style_note||'');
  $('#styleBox').open=!!$('#styleNote').value.trim();$('#spkN').value=m.people||'';
  $('#audWrap').classList.toggle('hidden',!gem.configured);$('#revBtn').classList.toggle('hidden',!gem.configured);$('#revNote').textContent='';
  $('#gemOn').classList.toggle('hidden',!gem.configured);$('#gemOff').classList.toggle('hidden',gem.configured);$('#gptBox').open=!gem.configured;
  setGem(gemBusy?'running':null);syncNeeds();renderPreview();$('#done').showModal();
  setTx(importing===id||busyId===id||!(m.lines&&m.lines.length));if(!importing&&busyId!==id&&!(m.lines&&m.lines.length))$('#txNote').textContent='⚠ 文字起こしがありません(音声を認識できませんでした)';
  if(justEnded&&gem.configured&&!m.summary&&!m.imported)runGem('sum')}
let gem={configured:false,url:'',key:''},gemBusy=false;
const GEM_MODELS=['gemini-flash-latest','gemini-2.5-flash'];
// Gemini はブラウザから直接呼ぶ(ブラウザと同じネットワーク設定で接続できる)
async function gemCall(key,text,ms,audio,schema){let last='';const parts0=[{text}];if(audio)parts0.unshift({inline_data:{mime_type:audio.mime,data:audio.data}});
  for(const mdl of GEM_MODELS){const ac=new AbortController();const tm=setTimeout(()=>ac.abort(),ms);let r;
    try{r=await fetch('https://generativelanguage.googleapis.com/v1beta/models/'+mdl+':generateContent',{method:'POST',signal:ac.signal,
      headers:{'Content-Type':'application/json','x-goog-api-key':key},body:JSON.stringify({contents:[{role:'user',parts:parts0}],generationConfig:schema?{temperature:0.2,responseMimeType:'application/json',responseSchema:schema}:{temperature:0.3}})})}
    catch(e){throw new Error(e.name==='AbortError'?'Google から '+Math.round(ms/1000)+' 秒応答がありません。もう一度お試しください':'Google に接続できません('+e.message+')')}
    finally{clearTimeout(tm)}
    const j=await r.json().catch(()=>({}));const msg=(j.error&&j.error.message)||String(r.status);
    if(r.status===404){last=msg;continue}
    if(!r.ok){if((r.status===400||r.status===401||r.status===403)&&/key/i.test(msg))throw new Error('API キーが正しくありません。設定し直してください');
      if(r.status===429)throw new Error('今日の無料枠を使い切ったか、混み合っています。しばらくしてから、または ChatGPT を使ってください');
      throw new Error('Gemini エラー('+r.status+'):'+msg)}
    const parts=(((j.candidates||[{}])[0]||{}).content||{}).parts||[];const out=parts.map(p=>p.text||'').join('').trim();
    if(!out)throw new Error('Gemini から回答がありませんでした。もう一度お試しください');return out}
  throw new Error('使えるモデルが見つかりません:'+last)}
async function getAudio(id){const r=await fetch('/api/meetings/'+id+'/audio');if(!r.ok)return null;const b=await r.blob();
  const d=await new Promise((ok,ng)=>{const fr=new FileReader();fr.onload=()=>ok(fr.result);fr.onerror=ng;fr.readAsDataURL(b)});
  return {mime:(b.type||'audio/ogg').split(';')[0],data:String(d).split(',')[1]}}
async function runGem(kind,ins){if(gemBusy)return;gemBusy=true;const id=doneId;setGem('running');
  if(kind==='rev')$('#gemNote').innerHTML='<span class="spin"></span>指示どおりに直しています…',$('#revNote').innerHTML='<span class="spin"></span>指示どおりに直しています…';
  try{await saveDone();let out;
    if(kind==='rev')out=await gemCall(gem.key,await api('/api/meetings/'+id+'/revise_prompt',{instruction:ins}),300000);
    else{let audio=null;
      if($('#audChk').checked){$('#gemNote').innerHTML='<span class="spin"></span>録音を準備しています…';try{audio=await getAudio(id)}catch(e){audio=null}}
      $('#gemNote').innerHTML='<span class="spin"></span>Gemini がまとめています'+(audio?'(録音も聞いているので 2〜4 分)':'(1〜2分)')+'…';
      try{out=await gemCall(gem.key,await api('/api/meetings/'+id+'/prompt?lang=ja'+(audio?'&audio=1':'')),420000,audio)}
      catch(e){if(!audio||/キー|無料枠/.test(e.message))throw e;  // 録音付きで失敗したら、文字起こしだけでもう一度
        $('#gemNote').innerHTML='<span class="spin"></span>録音付きではまとめられなかったので、文字起こしだけでまとめています…';
        out=await gemCall(gem.key,await api('/api/meetings/'+id+'/prompt?lang=ja'),300000)}}
    await api('/api/meetings/'+id,{summary:out});gemBusy=false;
    if(doneId===id&&$('#done').open){$('#doneSummary').value=out;$('#revText').value='';$('#revNote').textContent=kind==='rev'?'✓ 直しました':'';sumDirty=false;syncNeeds();renderPreview();setGem(null)}
    toast(kind==='rev'?'議事録を直しました':'議事録ができました')}
  catch(e){gemBusy=false;if(doneId===id){setGem('error',e.message);if(kind==='rev')$('#revNote').innerHTML='<span style="color:#991b1b">直せませんでした:'+esc(e.message)+'</span>'}}}
function setGem(st,msg){$('#gemBtn').disabled=st==='running';
  if(st==='error'){$('#gptBox').open=true;setTimeout(()=>$('#gptBox').scrollIntoView({behavior:'smooth',block:'center'}),100)}$('#revBtn').disabled=st==='running';
  $('#revBox').classList.toggle('hidden',!$('#doneSummary').value.trim());
  $('#gemBtn').textContent=$('#doneSummary').value.trim()?'✨ もう一度まとめる(Gemini)':'✨ 自動でまとめる(Gemini)';
  $('#gemNote').innerHTML=st==='running'?'<span class="spin"></span>Gemini がまとめています(1〜2分)…':st==='error'?'<span style="color:#991b1b"><b>Gemini でまとめられませんでした:</b>'+esc(msg||'')+'</span><br><b>↓ 下の「ChatGPT を使う」で作れます</b>(もう一度試すときは上のボタン)':'会議が終わると自動でまとめます。直したいときは下の欄を直接編集できます。'}
$('#revBtn').onclick=async()=>{const t=$('#revText').value.trim();if(!t){toast('直してほしいことを書くか、上のボタンを押してください');return}
  runGem('rev',t)};
document.querySelectorAll('#revChips button').forEach(b=>b.onclick=()=>{const v=$('#revText').value.trim();
  $('#revText').value=(v?v+'\n':'')+b.dataset.t;$('#revText').focus()});
async function clipWatch(on){try{await api('/api/meetings/'+doneId+'/clipwatch',{on})}catch(e){}}
$('#revGpt').onclick=async()=>{const t=$('#revText').value.trim();if(!t){toast('直してほしいことを書くか、上のボタンを押してください');return}
  const b=$('#revGpt');try{await saveDone();await navigator.clipboard.writeText(await api('/api/meetings/'+doneId+'/revise_prompt',{instruction:t,short:true}));clipWatch(true);
    b.textContent='✓ コピーしました';b.classList.add('copied');clearTimeout(b._t);b._t=setTimeout(()=>{b.textContent='📋 ChatGPT 用の修正依頼をコピー';b.classList.remove('copied')},3000);
    $('#revNote').innerHTML='ChatGPT の<b>同じチャット</b>に Ctrl+V → Enter。回答の「コピー」を押すと自動でここに入ります。';
    toast('✓ コピーしました。ChatGPT に Ctrl+V で貼り付けてください')}catch(e){toast(e.message)}};
let styleT=null;
$('#lenSel').onchange=()=>api('/api/meetings/'+doneId,{length:$('#lenSel').value}).catch(()=>{});
$('#styleNote').oninput=()=>{clearTimeout(styleT);styleT=setTimeout(()=>api('/api/meetings/'+doneId,{style_note:$('#styleNote').value}).catch(()=>{}),800)};
$('#gemBtn').onclick=async()=>{if($('#doneSummary').value.trim()&&!confirm('今の議事録を作り直しますか?(上書きされます)'))return;
  runGem('sum')};
document.querySelectorAll('.openGem').forEach(a=>a.onclick=async e=>{e.preventDefault();gem=await api('/api/gemini');$('#gemKey').value='';
  gemMsg(null);$('#gemSave').textContent='保存してテスト';$('#gemKey').placeholder=gem.configured?'設定済み(変更するときだけ貼り付け)':'ここにキーを貼り付け';$('#gemDel').classList.toggle('hidden',!gem.configured);$('#gemDlg').showModal()});
$('#gemOpen').onclick=e=>{e.preventDefault();window.open(gem.url||'https://aistudio.google.com/apikey','_blank')};
$('#gemClose').onclick=()=>$('#gemDlg').close();
function gemBoxRender(){api('/api/gemini').then(g=>{$('#gemState').textContent=g.configured?'(✅ 設定済み)':'(未設定)'}).catch(()=>{})}
$('#gemDlg').addEventListener('close',gemBoxRender);gemBoxRender();
$('#gemDel').onclick=async()=>{await api('/api/gemini',{key:''});toast('キーを削除しました');$('#gemDlg').close();if($('#done').open)openDone(doneId)};
function gemMsg(kind,html){const d=$('#gemMsg');d.classList.toggle('hidden',!kind);if(!kind)return;
  d.style.background={run:'#eef2ff',ok:'#e8f7ee',ng:'#fdecec'}[kind];d.style.color={run:'#1e3a8a',ok:'#166534',ng:'#991b1b'}[kind];d.innerHTML=html}
$('#gemSave').onclick=async()=>{const key=$('#gemKey').value.trim();if(!key){gemMsg('ng','キーを貼り付けてください');return}
  $('#gemSave').disabled=true;$('#gemSave').textContent='テスト中…';gemMsg('run','<span class="spin"></span>Google に接続してテストしています(最大 30 秒)…');
  try{await gemCall(key,'「OK」とだけ返してください。',30000);await api('/api/gemini',{key,verified:true});gem.key=key;gemMsg('ok','<b>✓ 成功しました。</b>キーを保存しました。次の会議から自動で議事録をつくります。');gem.configured=true;$('#gemDel').classList.remove('hidden');
    $('#gemSave').textContent='保存しました';setTimeout(()=>{if($('#done').open)openDone(doneId)},300)}
  catch(e){gemMsg('ng','<b>✗ 失敗しました:</b>'+esc(e.message))}
  finally{$('#gemSave').disabled=false;if($('#gemSave').textContent==='テスト中…')$('#gemSave').textContent='保存してテスト'}};
async function saveDone(){const b={name:$('#doneName').value,notes:$('#doneNotes').value,notes_after:$('#doneAfter').value,length:$('#lenSel').value,style_note:$('#styleNote').value};
  if(sumDirty)b.summary=$('#doneSummary').value;await api('/api/meetings/'+doneId,b);sumDirty=false}
$('#doneNotes').oninput=$('#doneAfter').oninput=$('#doneName').oninput=()=>{clearTimeout(notesT);notesT=setTimeout(saveDone,800)};
$('#doneSummary').oninput=()=>{sumDirty=true;syncNeeds();renderPreview();clearTimeout(sumT);sumT=setTimeout(saveDone,800)};
function syncNeeds(){const has=$('#doneSummary').value.trim().length>0;document.querySelectorAll('#done .needs').forEach(b=>{b.disabled=!has;b.title=has?'':'先に ChatGPT の回答を貼り付けてください'})}
function renderPreview(){const t=$('#doneSummary').value.trim();$('#aiPreview').classList.toggle('hidden',!t);if(!t)return;
  const inl=x=>esc(x).replace(/\*\*(.+?)\*\*/g,'<b>$1</b>');let h='',ul=false;
  for(const raw of t.split('\n')){const l=raw.trim();const li=raw.match(/^\s*[-・*•]\s+(.*)/);
    if(li){if(!ul){h+='<ul>';ul=true}h+='<li>'+inl(li[1])+'</li>';continue}
    if(ul){h+='</ul>';ul=false}
    if(!l)continue;
    if(/^=====(.+)=====$/.test(l))h+='<h4 style="font-size:17px">'+esc(l.replace(/=/g,'').replace('摘要','要約'))+'</h4>';
    else if(/^#+\s/.test(l))h+='<h4>'+inl(l.replace(/^#+\s*/,''))+'</h4>';
    else h+='<p>'+inl(l)+'</p>'}
  if(ul)h+='</ul>';$('#aiPreview').innerHTML=h}
async function promptText(){await saveDone();clearTimeout(styleT);await api('/api/meetings/'+doneId,{length:$('#lenSel').value,style_note:$('#styleNote').value});
  const t=await api('/api/meetings/'+doneId+'/prompt?lang=ja');clipWatch(true);return t}
let busyId=null;
function setTx(on){['#copyBtn','#openGpt','#gemBtn','#revBtn'].forEach(q=>{const b=$(q);if(b)b.disabled=on});
  const a=$('#saveTxt');if(a){a.style.pointerEvents=on?'none':'';a.style.opacity=on?'.4':''}
  let n=$('#txNote');if(!n){n=document.createElement('div');n.id='txNote';n.className='txnote';$('#copyBtn').closest('.row2').before(n)}
  n.textContent=on?'⏳ 字幕に変換中です。終わるまでコピー・まとめはできません(ボタンは灰色)':'';n.classList.toggle('hidden',!on)}
$('#copyBtn').onclick=async()=>{const b=$('#copyBtn');try{await navigator.clipboard.writeText(await promptText());
  b.textContent='✓ コピーしました';b.classList.add('copied');clearTimeout(b._t);b._t=setTimeout(()=>{b.textContent='📋 ChatGPT 用にコピー';b.classList.remove('copied')},3000);
  toast('✓ コピーしました。ChatGPT に Ctrl+V で貼り付けてください')}catch(e){toast('コピーできません:'+e.message)}};
$('#openGpt').onclick=()=>window.open('https://chatgpt.com/','_blank');
$('#saveTxt').onclick=async e=>{e.preventDefault();try{const t=await promptText();const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([t],{type:'text/plain'}));a.download=($('#doneName').value||'会議')+'_ChatGPT用.txt';a.click();
  toast('保存しました。ChatGPT の「+」から添付し、「添付の指示どおりに議事録を作ってください」と送ってください')}catch(e){toast(e.message)}};
// ---- 保存:PDF/Word × 日本語/中文 を選んで 1 ボタン ----
(function(){const sel=(LS.get('outSel','pdf-ja')||'').split(',');document.querySelectorAll('#outs input').forEach(c=>{c.checked=sel.includes(c.dataset.o);
  c.onchange=()=>LS.set('outSel',[...document.querySelectorAll('#outs input:checked')].map(x=>x.dataset.o).join(','))})})();
function dlFile(url){const a=document.createElement('a');a.href=url;a.download='';document.body.appendChild(a);a.click();a.remove()}
async function ensureZh(){const ja=$('#doneSummary').value.trim();const m=await api('/api/meetings/'+doneId);
  if(m.summary_zh&&m.summary_zh_src===ja)return;
  if(!gem.key&&!localReady())throw new Error('中文版を作るには Gemini かこのパソコンの AI が必要です(AI を準備中ならしばらくお待ちください)');
  $('#outNote').innerHTML='<span class="spin"></span>中文版を作成中…'+(gem.key?'(1分ほど)':'(このパソコンの AI で数分かかります)');
  const p='次の会議の議事録を、台湾の人が読む自然な繁体字中国語(台灣繁體中文)に訳してください。\n'+
    '・「=====摘要=====」「=====議事録=====」などの区切り行と、見出し・箇条書き・表などの書式はそのまま残す\n・内容は足したり削ったりせず、そのまま訳す。説明や前置きは書かず、訳文だけを出力\n・社名・人名・数字・型番はそのまま\n\n'+ja;
  const r=await aiCall(p,null,180000,null,3500);let zh=r.text.replace(/^```[a-z]*\n?|```$/g,'').trim();
  await api('/api/meetings/'+doneId,{summary_zh:zh,summary_zh_src:ja})}
$('#outBtn').onclick=async()=>{const outs=[...document.querySelectorAll('#outs input:checked')].map(x=>x.dataset.o);
  if(!outs.length){toast('保存するものを選んでください');return}
  $('#outBtn').disabled=true;try{await saveDone();
    if(outs.some(o=>o.endsWith('zh')))await ensureZh();
    for(const o of outs){const [f,l]=o.split('-');const url='/api/meetings/'+doneId+(f==='pdf'?'/pdf':'/docx/minutes')+'?lang='+l;
      const r=await fetch(url);if(!r.ok){throw new Error((await r.json()).error)}dlFile(url);await new Promise(z=>setTimeout(z,700))}
    $('#outNote').textContent='✅ 保存しました('+outs.length+' 件)'}
  catch(e){$('#outNote').textContent='';toast(e.message)}finally{$('#outBtn').disabled=false;syncNeeds()}};
document.querySelectorAll('#done [data-kind]').forEach(b=>b.onclick=async()=>{await saveDone();
  const url='/api/meetings/'+doneId+'/docx/'+b.dataset.kind;
  const r=await fetch(url);if(!r.ok){toast((await r.json()).error);return}location.href=url});
$('#reBtn').onclick=async()=>{if(!confirm('録音全体を認識し直しますか?長い会議ほど時間がかかり、その間は新しい録音ができません。'))return;
  $('#reBtn').disabled=true;try{await saveDone();await api('/api/meetings/'+doneId+'/reprocess',{});$('#reNote').textContent='処理を開始しました…';busyId=doneId;setTx(true)}
  catch(e){toast(e.message);$('#reBtn').disabled=false}};
$('#closeBtn').onclick=async()=>{await saveDone();clipWatch(false);$('#done').close();loadMeetings()};
$('#spkBtn').onclick=async()=>{if(!confirm('話者を分け直しますか?(字幕の文字はそのまま。付けた話者名はリセットされます)'))return;
  $('#spkBtn').disabled=true;try{await saveDone();await api('/api/meetings/'+doneId+'/respeak',{people:$('#spkN').value});$('#reNote').textContent='声の特徴を分析しています…';busyId=doneId;setTx(true)}
  catch(e){toast(e.message);$('#spkBtn').disabled=false}};
// ---- 一鍵更新 ----
const UPD={};
async function checkUpdate(){try{const v=await api('/api/version');UPD.cur=v.build;UPD.url=v.url;$('#buildTag').textContent='ver '+v.build;
  const r=await fetch(v.url+'version.json?t='+Date.now(),{cache:'no-store'});if(!r.ok)return;const j=await r.json();
  if(j.build&&j.build>v.build){UPD.want=j.build;$('#upNotes').textContent='(ver '+j.build+')'+(j.notes?' '+j.notes:'');$('#upBanner').classList.remove('hidden')}}catch(e){}}
$('#upBtn').onclick=async()=>{if(state.status==='recording'){toast('録音中は更新できません');return}
  const b=$('#upBtn');b.disabled=true;b.textContent='ダウンロード中…';
  try{const r=await fetch(UPD.url+'meeting.py?t='+Date.now(),{cache:'no-store'});if(!r.ok)throw new Error('ダウンロードできません('+r.status+')');
    const code=await r.text();b.textContent='更新中…';await api('/api/update',{code,build:UPD.want});
    b.textContent='再起動中…(30 秒ほど)';const t0=Date.now();
    const wait=async()=>{try{const v=await (await fetch('/api/version',{cache:'no-store'})).json();if(v.build===UPD.want){location.reload();return}}catch(e){}
      if(Date.now()-t0>180000){b.textContent='黒い画面を閉じて、start.bat をもう一度ダブルクリックしてください';return}setTimeout(wait,2000)};setTimeout(wait,4000)}
  catch(e){toast('更新できませんでした:'+e.message);b.disabled=false;b.textContent='⟳ 今すぐ更新'}};
checkUpdate();

// ---- 状態 ----
async function refresh(){state=await api('/api/state');applyState()}
function applyState(){
  $('#loading').classList.toggle('hidden',state.status!=='loading'&&state.status!=='error');
  if(state.status==='error')$('#loading').textContent=state.message;
  $('#startBtn').disabled=state.status!=='idle';$('#importOpen').disabled=state.status!=='idle';
  if(state.status==='processing')$('#loading').classList.remove('hidden'),$('#loading').textContent='認識し直し中です。終わるまで新しい録音はできません';
  $('#trNote').textContent={loading:'翻訳モデルを読み込み中(1分ほど)。読み込み前は原文だけ表示されます',failed:'翻訳モデルを読み込めませんでした。原文だけ表示します',off:'翻訳はオフです',ready:'中国語・英語の発言は自動で日本語訳がつきます'}[state.translator]||'';
  if(state.status==='recording'&&state.meeting&&cur!==state.meeting.id)showRec(state.meeting);
  if(state.status!=='recording')cur=null;
  loadMeetings();
}
const es=new EventSource('/events');
es.onmessage=ev=>{const e=JSON.parse(ev.data);
  if(e.type==='state'){Object.assign(state,e);if(e.status==='recording'||e.status==='idle'||e.status==='error')refresh();else applyState();if(cur)renderDigest()}
  else if(e.type==='line'&&cur){if(e.text)lastLineAt=Date.now();addLine(e)}
  else if(e.type==='speakers'){speakers=e.speakers;if(cur)refreshNames()}
  else if(e.type==='error')toast(e.message);
  else if(e.type==='paused')setPaused(e.paused);
  else if(e.type==='summary'&&e.id===doneId&&$('#done').open){
    if(e.status==='done'){$('#doneSummary').value=e.summary;$('#revText').value='';sumDirty=false;syncNeeds();renderPreview();setGem(null);
      if(e.via==='clip')$('#revNote').textContent='✓ ChatGPT の回答を取り込みました';toast(e.via==='clip'?'✓ ChatGPT の回答を取り込みました':'議事録を更新しました')}
    else setGem(e.status,e.message)}
  else if(e.type==='progress'&&e.id===doneId){$('#reNote').textContent=`${e.text} ${e.pct}%`;
    if(e.id===importing){$('#aiStatus').innerHTML=e.done?(e.pct===100?(gem.configured?'字幕ができました。Gemini がまとめています…':'字幕ができました。下の手順 2 へ進んでください'):esc(e.text)):`<span class="spin"></span>字幕に変換中 ${e.pct}%`;if(e.done){importing=null;if(e.pct===100&&gem.configured)runGem('sum')}}
    if(e.done){$('#reBtn').disabled=false;$('#spkBtn').disabled=false;busyId=null;if(e.respeak){api('/api/meetings/'+doneId).then(m=>{speakers=m.speakers||{}}).catch(()=>{});toast('話者を分け直しました。議事録を作り直すときは「もう一度まとめる」を押してください')}setTx(!!e.empty);if(e.empty){$('#txNote').textContent='⚠ '+e.text;toast(e.text)}if(e.pct===100&&!e.respeak)toast('字幕ができました')}}};
es.onopen=refresh;
loadDevices();refresh();
</script></body></html>
"""


# ---------- 主程式 ----------

def main():
    ap = argparse.ArgumentParser(description="会議字幕")
    ap.add_argument("--no-translate", action="store_true", help="翻訳をオフにする")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--threads", type=int, default=2, help="辨識用 CPU 執行緒數")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    app = App(args.threads, not args.no_translate)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    except OSError:
        print(f"\n会議字幕がすでに起動しています(port {args.port} 使用中)。")
        print("ほかの黒いウィンドウを閉じるか、再起動してから start.bat をダブルクリックしてください。")
        input("Enter キーで閉じます…")
        return
    try:
        (BASE / "running.json").write_text(json.dumps({"build": BUILD, "pid": os.getpid(), "started": dt.datetime.now().isoformat(timespec="seconds")}), encoding="utf-8")
    except OSError:
        pass
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://localhost:{args.port}"
    print(f"会議字幕を起動しました:{url}\nこのウィンドウを閉じるか Ctrl+C で終了します\n", flush=True)
    flag = BASE / "updated.flag"
    if flag.exists() and time.time() - flag.stat().st_mtime < 120:  # 更新で再起動したときは、開いている画面がそのまま読み直すので新しいタブは開かない
        flag.unlink(missing_ok=True)
    elif not args.no_browser:
        webbrowser.open(url)
    threading.Thread(target=app.load, daemon=True).start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if app.status == "recording":
            app.stop()


if __name__ == "__main__":
    main()
