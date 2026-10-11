"""
AIREC 업로드 수신 서버 (Render 무료 웹서비스 / Docker)

흐름: AIREC 앱 POST(multipart: file, fileName, sn)
      → 즉시 200 응답 (앱이 재전송하지 않도록)
      → 백그라운드: ffmpeg로 10분 단위 Opus 압축 → Groq Whisper 한국어 전사
      → 녹음 시각 기준 타임스탬프 붙인 텍스트를 Apps Script 경유 구글 드라이브 'AIREC_inbox'에 저장
      → 음성 파일 삭제

환경변수(Render Environment):
  GROQ_API_KEY   Groq API 키
  UPLOAD_TOKEN   업로드 주소 비밀 토큰 (/api/upload/<UPLOAD_TOKEN>)
  GAS_URL        Apps Script 웹앱 URL
  GAS_TOKEN      Apps Script 공유 비밀값
실행: uvicorn app:app --host 0.0.0.0 --port 7860
"""
import base64
import datetime as dt
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
UPLOAD_TOKEN = os.environ.get("UPLOAD_TOKEN", "")
GAS_URL = os.environ.get("GAS_URL", "")
GAS_TOKEN = os.environ.get("GAS_TOKEN", "")
GROQ_URL = os.environ.get("GROQ_URL", "https://api.groq.com/openai/v1/audio/transcriptions")
MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3")
CHUNK_SEC = 600  # 10분 단위 분할 (Groq 무료 25MB 한도 대비 충분히 작게)

SPOOL = Path(os.environ.get("SPOOL_DIR", "/tmp/airec_spool"))
SPOOL.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("airec")

# Whisper가 무음 구간에서 자주 만들어내는 문구
HALLUCINATIONS = ["시청해주셔서 감사합니다", "구독과 좋아요", "MBC 뉴스", "자막 제공", "다음 영상에서 만나요"]

app = FastAPI()
jobs: "queue.Queue[Path]" = queue.Queue()
STATUS = {"processed": 0, "failed": 0, "last": None}


# ---------------- 업로드 수신 ----------------
@app.get("/")
@app.get("/health")
def health():
    return {"ok": True, "queue": jobs.qsize(), **STATUS}


@app.post("/api/upload/{token}")
async def upload(token: str, file: UploadFile = File(...),
                 fileName: str = Form(None), sn: str = Form("")):
    if not UPLOAD_TOKEN or token != UPLOAD_TOKEN:
        raise HTTPException(status_code=403, detail="forbidden")
    name = re.sub(r"[^0-9A-Za-z._-]", "_", fileName or file.filename or f"{int(time.time())}.wav")
    job_dir = SPOOL / f"{int(time.time()*1000)}_{Path(name).stem}"
    job_dir.mkdir(parents=True, exist_ok=True)
    with open(job_dir / name, "wb") as f:
        shutil.copyfileobj(file.file, f)
    (job_dir / "meta.json").write_text(json.dumps({"fileName": name, "sn": sn, "tries": 0}))
    jobs.put(job_dir)
    log.info("received %s (%s bytes)", name, (job_dir / name).stat().st_size)
    return {"ok": True}  # 200 → 앱에서 업로드 성공 처리


# ---------------- 처리 ----------------
def start_time_from_name(name: str) -> dt.datetime | None:
    m = re.match(r"(\d{14})", name)
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def split_audio(src: Path, out_dir: Path) -> list[Path]:
    """16kHz 모노 Opus 32kbps, 10분 단위로 분할 (30분 WAV ≈ 7MB)"""
    pattern = out_dir / "chunk_%03d.ogg"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "32k",
         "-f", "segment", "-segment_time", str(CHUNK_SEC), str(pattern)],
        check=True, timeout=600)
    return sorted(out_dir.glob("chunk_*.ogg"))


def transcribe(chunk: Path) -> list[dict]:
    """Groq Whisper 호출. 429(한도 초과)면 대기 후 재시도."""
    for attempt in range(6):
        with open(chunk, "rb") as f:
            r = httpx.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": (chunk.name, f, "audio/ogg")},
                data={"model": MODEL, "language": "ko", "response_format": "verbose_json",
                      "temperature": "0"},
                timeout=300)
        if r.status_code == 200:
            return r.json().get("segments", []) or []
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after", 60 * (attempt + 1)))
            log.warning("Groq rate limit, %ss 대기", wait)
            time.sleep(min(wait, 900))
            continue
        raise RuntimeError(f"Groq {r.status_code}: {r.text[:300]}")
    raise RuntimeError("Groq 재시도 초과")


def clean(segs: list[dict]) -> list[dict]:
    out, prev = [], None
    for s in segs:
        t = (s.get("text") or "").strip()
        if not t or s.get("no_speech_prob", 0) > 0.6 or s.get("avg_logprob", 0) < -1.0:
            continue
        if any(h in t for h in HALLUCINATIONS) or t == prev:
            continue
        out.append({**s, "text": t})
        prev = t
    return out


def gas_save(name: str, content: str, mime: str = "text/plain", b64: bool = False):
    payload = {"token": GAS_TOKEN, "name": name, "mimeType": mime,
               "content": content, "base64": b64}
    r = httpx.post(GAS_URL, json=payload, timeout=120, follow_redirects=True)
    if r.status_code != 200 or '"ok":true' not in r.text.replace(" ", ""):
        raise RuntimeError(f"GAS 저장 실패 {r.status_code}: {r.text[:200]}")


def process(job_dir: Path):
    meta = json.loads((job_dir / "meta.json").read_text())
    name = meta["fileName"]
    src = job_dir / name
    start = start_time_from_name(name)
    chunks_dir = job_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    chunks = split_audio(src, chunks_dir)

    lines = []
    for i, ch in enumerate(chunks):
        for s in clean(transcribe(ch)):
            sec = i * CHUNK_SEC + float(s.get("start", 0))
            stamp = (start + dt.timedelta(seconds=sec)).strftime("%H:%M:%S") if start else f"+{int(sec)}s"
            lines.append(f"[{stamp}] {s['text']}")

    stem = Path(name).stem
    date_str = start.strftime("%Y-%m-%d %H:%M:%S") if start else "unknown"
    if lines:
        header = (f"# AIREC 녹음 전사\nrecorded_at: {date_str}\nsource_file: {name}\n"
                  f"device_sn: {meta.get('sn','')}\nsegments: {len(lines)}\n---\n")
        gas_save(f"{stem}.txt", header + "\n".join(lines))
        log.info("saved transcript %s (%d lines)", stem, len(lines))
    else:
        log.info("no speech in %s — skip", name)


def worker():
    # 재시작 시 남아있던 작업 복구
    for d in sorted(SPOOL.iterdir()):
        if (d / "meta.json").exists():
            jobs.put(d)
    while True:
        job_dir = jobs.get()
        try:
            process(job_dir)
            STATUS["processed"] += 1
            STATUS["last"] = dt.datetime.now().isoformat(timespec="seconds")
            shutil.rmtree(job_dir, ignore_errors=True)  # 음성 파일 삭제
        except Exception as e:  # noqa: BLE001
            log.exception("처리 실패 %s: %s", job_dir.name, e)
            meta_p = job_dir / "meta.json"
            meta = json.loads(meta_p.read_text())
            meta["tries"] = meta.get("tries", 0) + 1
            meta_p.write_text(json.dumps(meta))
            if meta["tries"] < 3:
                threading.Timer(300, jobs.put, args=(job_dir,)).start()  # 5분 뒤 재시도
            else:
                STATUS["failed"] += 1
                # 최종 실패: 음성(압축본)을 드라이브에 보존해 유실 방지
                try:
                    ogg = job_dir / "fallback.ogg"
                    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i",
                                    str(job_dir / meta["fileName"]), "-ac", "1", "-ar", "16000",
                                    "-c:a", "libopus", "-b:a", "24k", str(ogg)], check=True, timeout=600)
                    gas_save(Path(meta["fileName"]).stem + "_FAILED.ogg",
                             base64.b64encode(ogg.read_bytes()).decode(), "audio/ogg", True)
                    shutil.rmtree(job_dir, ignore_errors=True)
                except Exception:  # noqa: BLE001
                    log.exception("fallback 저장도 실패 — /tmp에 보존")
        finally:
            jobs.task_done()


threading.Thread(target=worker, daemon=True).start()
