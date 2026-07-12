from __future__ import annotations

import asyncio
from faster_whisper import WhisperModel
import json
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent
WORD_BANK: dict[str, dict[str, Any]] = json.loads(
    (BASE_DIR / "word_bank.json").read_text(encoding="utf-8")
)

# Must be the FULL POST endpoint shown in the Whisper Swagger page.
WHISPER_URL = (
    "https://gelxxn--thai-whisper-api-thonburianwhisper-web.modal.run/"
    "transcribe-audio"
)

_local_whisper_model: WhisperModel | None = None


def _get_local_whisper_model() -> WhisperModel:
    global _local_whisper_model

    if _local_whisper_model is None:
        _local_whisper_model = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8",
        )

    return _local_whisper_model
WEIGHTS = {
    "text": 0.50,
    "pitch": 0.40,
    "mouthOpen": 0.05,
    "duration": 0.05,
}
PASS_THRESHOLD = 80.0

app = FastAPI(title="TASE001 Pronunciation API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_connections(request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    print(f"connected: {client_ip} -> {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"responded: {client_ip} -> {request.method} {request.url.path} [{response.status_code}]")
    return response


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def normalize_text(text: str) -> str:
    return "".join(str(text or "").strip().split())


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_a != char_b),
            ))
        previous = current
    return previous[-1]


def textSimilarityPct(target: str, recognized: str) -> float:
    target = normalize_text(target)
    recognized = normalize_text(recognized)
    if not recognized:
        return 0.0
    max_len = max(len(target), len(recognized))
    return round(clamp((1.0 - levenshtein(target, recognized) / max_len) * 100.0), 2)


def classifyThaiChar(ch: str) -> str:
    if ch in "่้๊๋":
        return "วรรณยุกต์"
    if "\u0e01" <= ch <= "\u0e2e":
        return "พยัญชนะ"
    if ch in "ะาิีึืุูเแโใไั็์ํๅ":
        return "สระ/เครื่องหมาย"
    return "อื่น ๆ"


def levenshteinOps(a: str, b: str) -> list[dict[str, str | None]]:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] if a[i-1] == b[j-1] else 1 + min(
                dp[i-1][j-1], dp[i-1][j], dp[i][j-1]
            )
    i, j = m, n
    ops: list[dict[str, str | None]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i-1] == b[j-1] and dp[i][j] == dp[i-1][j-1]:
            ops.append({"type":"match", "from":a[i-1], "to":b[j-1]}); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append({"type":"substitute", "from":a[i-1], "to":b[j-1]}); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append({"type":"delete", "from":a[i-1], "to":None}); i -= 1
        else:
            ops.append({"type":"insert", "from":None, "to":b[j-1]}); j -= 1
    return list(reversed(ops))


def explainDiff(target: str, heard: str) -> list[str]:
    messages: list[str] = []
    for op in levenshteinOps(target, heard):
        typ, src, dst = op["type"], op["from"], op["to"]
        if typ == "match": continue
        if typ == "delete" and src:
            messages.append(f'ขาด{classifyThaiChar(src)} “{src}”')
        elif typ == "insert" and dst:
            messages.append(f'มี{classifyThaiChar(dst)} “{dst}” เกินมา')
        elif typ == "substitute" and src and dst:
            messages.append(f'{classifyThaiChar(src)} “{src}” ถูกเปลี่ยนเป็น “{dst}”')
    return messages


def stripToneMarks(text: str) -> str:
    return "".join(ch for ch in text if ch not in "่้๊๋")


def splitBySyllableLengths(recognized: str, syllableTexts: list[str]) -> list[str]:
    recognized = normalize_text(recognized)
    if not syllableTexts: return [recognized]
    total = sum(max(1, len(s)) for s in syllableTexts)
    result, pos = [], 0
    for i, syllable in enumerate(syllableTexts):
        if i == len(syllableTexts) - 1:
            result.append(recognized[pos:]); break
        length = round(len(recognized) * max(1, len(syllable)) / total)
        result.append(recognized[pos:pos+length]); pos += length
    return result


def analyzeTone(target: str, recognized: str, ref: dict[str, Any]) -> dict[str, Any]:
    target, recognized = normalize_text(target), normalize_text(recognized)
    if target == recognized:
        return {"hasIssue":False, "issues":[], "diffIssues":[]}
    syllables = ref.get("syllables") or []
    if len(syllables) > 1:
        texts = [str(s["text"]) for s in syllables]
        segments = splitBySyllableLengths(recognized, texts)
        issues, diffs = [], []
        for index, syllable in enumerate(syllables):
            wanted = str(syllable["text"])
            heard = segments[index] if index < len(segments) else ""
            if heard == wanted or not heard: continue
            if stripToneMarks(heard) == stripToneMarks(wanted):
                issues.append({"syllable":wanted,"tone":syllable.get("tone", ref.get("tone","")),"heard":heard,"index":index+1})
            else:
                diffs.append({"syllable":wanted,"heard":heard,"index":index+1,"explanations":explainDiff(wanted,heard)})
        return {"hasIssue":bool(issues), "issues":issues, "diffIssues":diffs}
    if stripToneMarks(target) == stripToneMarks(recognized):
        return {"hasIssue":True,"issues":[{"syllable":target,"tone":ref.get("tone",""),"heard":recognized,"index":1}],"diffIssues":[]}
    return {"hasIssue":False,"issues":[],"diffIssues":[{"syllable":target,"heard":recognized,"index":1,"explanations":explainDiff(target,recognized)}]}


def component_score(value: float, mean: float, std: float) -> tuple[float, float]:
    if value <= 0: return 0.0, 99.0
    safe_std = max(abs(std), 1e-6)
    z = abs(value - mean) / safe_std
    return clamp(100.0 - z * 22.0), z


def computeScores(pitch: float, mouthOpen: float, duration: float, textPct: float, ref: dict[str, Any]) -> dict[str, float]:
    pitch_score, pitch_z = component_score(pitch, float(ref["multi"]["pitch"]["mean"]), float(ref["multi"]["pitch"]["std"]))
    duration_score, duration_z = component_score(duration, float(ref["multi"]["duration"]["mean"]), float(ref["multi"]["duration"]["std"]))
    target_mouth = max(float(ref["mouth"]["jawOpenMax"]), 1e-6)
    mouth_score = clamp(100.0 - abs(mouthOpen - target_mouth) / target_mouth * 100.0) if mouthOpen > 0 else 0.0
    acoustic_score = pitch_score * 0.80 + mouth_score * 0.10 + duration_score * 0.10
    overall = textPct * 0.50 + acoustic_score * 0.50
    return {
        "overallScore":round(overall,2), "textScore":round(textPct,2),
        "acousticScore":round(acoustic_score,2), "pitchScore":round(pitch_score,2),
        "mouthScore":round(mouth_score,2), "durationScore":round(duration_score,2),
        "pitchZScore":round(pitch_z,4), "durationZScore":round(duration_z,4),
    }


def renderScores(raw: dict[str, float], textPct: float, toneAnalysis: dict[str, Any], target: str, transcript: str, ref: dict[str, Any]) -> dict[str, Any]:
    advice: list[dict[str,str]] = []
    if toneAnalysis["hasIssue"]:
        for issue in toneAnalysis["issues"]:
            advice.append({"type":"tone","severity":"warning","title":"วรรณยุกต์","message":f'พยางค์ “{issue["syllable"]}” ควรใช้เสียง{issue["tone"]} (ระบบได้ยินเป็น “{issue["heard"]}”)'})
    elif toneAnalysis.get("diffIssues"):
        for diff in toneAnalysis["diffIssues"]:
            details = "; ".join(diff.get("explanations",[])[:3]) or "กรุณาพูดให้ชัดเจนขึ้น"
            advice.append({"type":"transcript","severity":"warning","title":"คำที่ได้ยิน","message":f'ระบบได้ยิน “{diff["heard"] or "ไม่พบคำพูด"}” แทน “{diff["syllable"]}” — {details}'})
    elif textPct < 90:
        advice.append({"type":"transcript","severity":"warning","title":"ความชัดเจนของคำ","message":f'คำที่ได้ยินคล้าย “{target}” {textPct:.0f}% ลองออกเสียงพยัญชนะ สระ และตัวสะกดให้ชัดขึ้น'})

    pitch_ref = ref["multi"]["pitch"]
    if raw["pitchHz"] < float(pitch_ref["mean"]) - float(pitch_ref["std"]):
        advice.append({"type":"pitch","severity":"warning","title":"ระดับเสียง","message":f'ระดับเสียงต่ำกว่าช่วงอ้างอิง ลองเน้นเสียง{ref.get("tone","")}ให้ชัดขึ้น'})
    elif raw["pitchHz"] > float(pitch_ref["mean"]) + float(pitch_ref["std"]):
        advice.append({"type":"pitch","severity":"warning","title":"ระดับเสียง","message":"ระดับเสียงสูงกว่าช่วงอ้างอิง ลองผ่อนเสียงลงเล็กน้อย"})

    mouth_target = float(ref["mouth"]["jawOpenMax"])
    if raw["mouthOpenMax"] <= 0:
        advice.append({"type":"mouth","severity":"warning","title":"รูปปาก","message":"ไม่พบข้อมูลการอ้าปาก กรุณาจัดใบหน้าให้อยู่กลางกล้อง"})
    elif raw["mouthOpenMax"] < mouth_target * 0.55:
        advice.append({"type":"mouth","severity":"warning","title":"รูปปาก","message":f'อ้าปากให้กว้างขึ้น (วัดได้ {raw["mouthOpenMax"]:.2f}, ต้นแบบสูงสุดประมาณ {mouth_target:.2f})'})

    duration_ref = ref["multi"]["duration"]
    if raw["durationSeconds"] < float(duration_ref["mean"]) - float(duration_ref["std"]):
        advice.append({"type":"duration","severity":"warning","title":"ความยาวเสียง","message":f'ลากเสียง “{target}” ให้นานขึ้นอีกเล็กน้อย'})
    elif raw["durationSeconds"] > float(duration_ref["mean"]) + float(duration_ref["std"]):
        advice.append({"type":"duration","severity":"warning","title":"ความยาวเสียง","message":"ลดความยาวเสียงลงเล็กน้อย"})

    energy_ref = ref["multi"]["energy"]
    if raw["energy"] < max(0.0, float(energy_ref["mean"]) - float(energy_ref["std"])):
        advice.append({"type":"energy","severity":"warning","title":"ความดังของเสียง","message":"ลองพูดให้ดังและชัดขึ้นเล็กน้อย"})
    elif raw["energy"] > float(energy_ref["mean"]) + float(energy_ref["std"]) * 3:
        advice.append({"type":"energy","severity":"warning","title":"ความดังของเสียง","message":"เสียงดังหรืออยู่ใกล้ไมโครโฟนมากเกินไป ลองถอยออกเล็กน้อย"})

    is_good = raw["overallScore"] >= PASS_THRESHOLD
    if is_good and not advice:
        advice.append({"type":"success","severity":"good","title":"ทำได้ดี","message":f'ออกเสียง “{target}” ได้ดี'})
    feedback = f'ออกเสียง “{target}” ผ่านเกณฑ์' if is_good else (advice[0]["message"] if advice else f'ลองออกเสียง “{target}” อีกครั้ง')
    return {
        "score":round(raw["overallScore"] / 100.0,4), "scorePercent":round(raw["overallScore"]),
        "isGood":is_good, "feedback":feedback, "transcript":transcript, "advice":advice[:5],
        "breakdown":{"textScore":raw["textScore"],"acousticScore":raw["acousticScore"],"pitchScore":raw["pitchScore"],"mouthScore":raw["mouthScore"],"durationScore":raw["durationScore"]},
        "weights":WEIGHTS,
        "metrics":{"pitchHz":raw["pitchHz"],"energy":raw["energy"],"durationSeconds":raw["durationSeconds"],"mouthOpenMax":raw["mouthOpenMax"],"mouthOpenMean":raw["mouthOpenMean"],"pitchZScore":raw["pitchZScore"],"durationZScore":raw["durationZScore"]},
    }


def finalizeWithTranscript(text: str, target: str, pitch: float, energy: float, duration: float, mouthOpen: float, mouthOpenMean: float, ref: dict[str,Any]) -> dict[str,Any]:
    transcript = normalize_text(text)
    text_pct = textSimilarityPct(target, transcript)
    tone = analyzeTone(target, transcript, ref)
    scores = computeScores(pitch, mouthOpen, duration, text_pct, ref)
    raw = {**scores,"pitchHz":pitch,"energy":energy,"durationSeconds":duration,"mouthOpenMax":mouthOpen,"mouthOpenMean":mouthOpenMean}
    return renderScores(raw, text_pct, tone, target, transcript, ref)


def _read_wav(path: Path) -> tuple[np.ndarray,int]:
    try:
        with wave.open(str(path),"rb") as f:
            sr, channels, width = f.getframerate(), f.getnchannels(), f.getsampwidth()
            frames = f.readframes(f.getnframes())
    except wave.Error as exc:
        raise HTTPException(415, detail=f"INVALID_WAV: {exc}") from exc
    if width != 2: raise HTTPException(415, detail="WAV_MUST_BE_PCM16")
    samples = np.frombuffer(frames,dtype=np.int16).astype(np.float32)/32768.0
    if channels > 1: samples = samples.reshape(-1,channels).mean(axis=1)
    return samples,sr


def _convert_to_wav(source: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg: raise HTTPException(415, detail="FFMPEG_REQUIRED_FOR_NON_WAV_AUDIO")
    result = subprocess.run([ffmpeg,"-y","-i",str(source),"-ac","1","-ar","16000","-c:a","pcm_s16le",str(destination)],capture_output=True,text=True)
    if result.returncode != 0: raise HTTPException(415, detail=f"AUDIO_CONVERSION_FAILED: {result.stderr[-300:]}")


def _audio_metrics(samples: np.ndarray, sample_rate: int) -> dict[str,float]:
    if samples.size == 0: return {"pitchHz":0.0,"energy":0.0,"durationSeconds":0.0}
    frame_len, hop = max(1,int(sample_rate*.04)), max(1,int(sample_rate*.02))
    pitches, rms_values, voiced = [], [], 0
    for start in range(0,max(1,samples.size-frame_len+1),hop):
        frame=samples[start:start+frame_len]
        if frame.size<frame_len: break
        frame=frame-float(frame.mean()); rms=float(np.sqrt(np.mean(frame*frame)+1e-12)); rms_values.append(rms)
        if rms<.008: continue
        voiced += 1; corr=np.correlate(frame,frame,mode="full")[frame_len-1:]
        min_lag=max(1,int(sample_rate/400)); max_lag=min(len(corr)-1,int(sample_rate/70))
        if max_lag<=min_lag: continue
        lag=int(np.argmax(corr[min_lag:max_lag+1]))+min_lag
        if corr[0]>0 and corr[lag]/corr[0]>=.25: pitches.append(sample_rate/lag)
    return {"pitchHz":round(float(np.median(pitches)) if pitches else 0.0,3),"energy":round(float(np.mean(rms_values)) if rms_values else 0.0,6),"durationSeconds":round(voiced*hop/sample_rate,3)}

def _transcribe_local_sync(path: Path) -> str:
    model = _get_local_whisper_model()

    segments, _ = model.transcribe(
        str(path),
        language="th",
        beam_size=5,
        vad_filter=True,
    )

    return "".join(segment.text for segment in segments).strip()


async def _transcribe_local(path: Path) -> str:
    return await asyncio.to_thread(
        _transcribe_local_sync,
        path,
    )

async def _transcribe(
    path: Path,
    filename: str,
    content_type: str,
) -> tuple[str, str]:
    api_error: str | None = None

    # 1. ลอง Modal API ก่อน
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            with path.open("rb") as audio:
                response = await client.post(
                    WHISPER_URL,
                    files={
                        "file": (
                            filename,
                            audio,
                            content_type,
                        )
                    },
                )

        if response.status_code < 400:
            data = response.json()

            transcript = str(
                data.get("text")
                or data.get("transcript")
                or data.get("transcription")
                or ""
            ).strip()

            if transcript:
                return transcript, "modal"

            api_error = "Modal returned an empty transcript"
        else:
            api_error = (
                f"Modal HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

    except Exception as exc:
        api_error = str(exc)

    # 2. ถ้า Modal ใช้ไม่ได้ ค่อยใช้ Faster-Whisper
    try:
        transcript = await _transcribe_local(path)

        if transcript:
            return transcript, "faster-whisper"

        raise RuntimeError("Faster-Whisper returned an empty transcript")

    except Exception as local_exc:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "TRANSCRIPTION_UNAVAILABLE",
                "modalError": api_error,
                "localError": str(local_exc),
            },
        ) from local_exc

@app.get("/")
def root() -> dict[str,str]: return {"message":"TASE001 API is running","docs":"/docs"}

@app.get("/health")
def health() -> dict[str,Any]: return {"ok":True,"supportedWords":len(WORD_BANK),"whisperConfigured":bool(WHISPER_URL)}

@app.get("/evaluation/words/{word}")
def get_word(word: str) -> dict[str,Any]:
    ref=WORD_BANK.get(word)
    if ref is None: raise HTTPException(404,detail="WORD_NOT_FOUND")
    return {"vocabText":word,**ref,"weights":WEIGHTS,"passThreshold":PASS_THRESHOLD}

@app.post("/evaluate")
async def evaluate(file: UploadFile=File(...), vocab_text: str=Form(...), mouth_summary: str=Form("{}")) -> dict[str,Any]:
    ref=WORD_BANK.get(vocab_text)
    if ref is None: raise HTTPException(400,detail="WORD_NOT_SUPPORTED")
    try: mouth=json.loads(mouth_summary)
    except json.JSONDecodeError as exc: raise HTTPException(400,detail="INVALID_MOUTH_SUMMARY_JSON") from exc
    content=await file.read()
    if not content: raise HTTPException(400,detail="EMPTY_AUDIO_FILE")
    suffix=Path(file.filename or "recording.wav").suffix.lower() or ".bin"
    with tempfile.TemporaryDirectory() as tmp:
        source=Path(tmp)/f"input{suffix}"; source.write_bytes(content)
        wav_path=source
        if suffix != ".wav":
            wav_path=Path(tmp)/"converted.wav"; _convert_to_wav(source,wav_path)
        samples,sr=_read_wav(wav_path); metrics=_audio_metrics(samples,sr)
        transcript, transcriptionSource = await _transcribe(
            wav_path,
            file.filename or "recording.wav",
            file.content_type or "audio/wav",
        )
    mouth_max=float(mouth.get("mouthOpenMax") or mouth.get("jawOpenMax") or 0.0)
    mouth_mean=float(mouth.get("mouthOpenMean") or mouth.get("jawOpenMean") or 0.0)
    result=finalizeWithTranscript(transcript,vocab_text,metrics["pitchHz"],metrics["energy"],metrics["durationSeconds"],mouth_max,mouth_mean,ref)
    return {"success":True,"vocabText":vocab_text,"guide":{"instruction":ref.get("pron",""),"tone":ref.get("tone",""),"syllables":ref.get("syllables",[])},**result}


@app.post("/ask-advice")
async def ask_advice(
    vocab_text: str = Form(...),
    question: str = Form(""),
) -> dict[str, Any]:
    ref = WORD_BANK.get(vocab_text)
    if ref is None:
        raise HTTPException(400, detail="WORD_NOT_SUPPORTED")

    tips: list[str] = ref.get("tips") or []
    if tips:
        answer = tips[0]
    else:
        tone = ref.get("tone", "")
        answer = f'ลองพูดคำว่า "{vocab_text}" ช้าๆ ทีละพยางค์ก่อน สังเกตวรรณยุกต์{tone}ให้ชัดเจน แล้วค่อยๆ เพิ่มความเร็วให้เป็นธรรมชาติ'

    return {"success": True, "vocabText": vocab_text, "answer": answer}
