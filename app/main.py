import os
import tempfile
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from app.config import settings
from app.model_manager import aligner_slot, asr_slot

app = FastAPI()


def _resolve_model(model: str) -> str:
    return settings.default_model if model == "whisper-1" else model


def _srt_time(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_time(seconds: float) -> str:
    return _srt_time(seconds).replace(",", ".")


def _to_srt(segments: Any) -> str:
    parts = []
    for i, seg in enumerate(segments, 1):
        parts.append(f"{i}\n{_srt_time(seg.start_time)} --> {_srt_time(seg.end_time)}\n{seg.text.strip()}\n")
    return "\n".join(parts)


def _to_vtt(segments: Any) -> str:
    parts = ["WEBVTT\n"]
    for seg in segments:
        parts.append(f"{_vtt_time(seg.start_time)} --> {_vtt_time(seg.end_time)}\n{seg.text.strip()}\n")
    return "\n".join(parts)


@app.post("/v1/audio/transcriptions")
async def transcribe(request: Request):
    form = await request.form()

    file = form.get("file")
    if file is None:
        raise HTTPException(status_code=422, detail="Missing required field: file")

    model = form.get("model")
    if not model:
        raise HTTPException(status_code=422, detail="Missing required field: model")

    language = form.get("language") or None
    context = form.get("prompt") or ""
    response_format = form.get("response_format") or "json"
    timestamp_granularities = form.getlist("timestamp_granularities[]")

    needs_timestamps = bool(timestamp_granularities) or response_format in ("srt", "vtt", "verbose_json")
    if needs_timestamps and not settings.enable_timestamps:
        raise HTTPException(
            status_code=400,
            detail="Timestamp support is disabled. Set ENABLE_TIMESTAMPS=true to enable.",
        )

    model_name = _resolve_model(model)
    suffix = os.path.splitext(getattr(file, "filename", "") or "audio")[1] or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        asr_result = await asr_slot.transcribe(model_name, tmp_path, language, context)
        text: str = asr_result[0].text
        detected_language: str = asr_result[0].language

        segments = None
        if needs_timestamps:
            align_result = await aligner_slot.align(tmp_path, text, detected_language)
            segments = align_result[0]
    finally:
        os.unlink(tmp_path)

    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "srt":
        return PlainTextResponse(_to_srt(segments), media_type="text/plain")
    if response_format == "vtt":
        return PlainTextResponse(_to_vtt(segments), media_type="text/vtt")
    if response_format == "verbose_json":
        seg_list = []
        if segments:
            for i, seg in enumerate(segments):
                seg_list.append({"id": i, "start": seg.start_time, "end": seg.end_time, "text": seg.text})
        return {"task": "transcribe", "language": detected_language, "text": text, "segments": seg_list}
    return {"text": text}
