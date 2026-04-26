import asyncio
import logging
from typing import Any

import torch
from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner

from app.config import settings

logger = logging.getLogger(__name__)


def _effective_device() -> str:
    if settings.device != "cuda":
        return settings.device
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to cpu")
        return "cpu"
    major, minor = torch.cuda.get_device_capability(0)
    device_sm = major * 10 + minor
    arch_list = torch.cuda.get_arch_list()
    compiled_sms = [int(a.replace("sm_", "")) for a in arch_list if a.startswith("sm_")]
    if compiled_sms and device_sm < min(compiled_sms):
        name = torch.cuda.get_device_name(0)
        logger.warning(
            "%s (sm_%d%d) is not supported by this PyTorch build (min: sm_%d). Falling back to cpu.",
            name, major, minor, min(compiled_sms),
        )
        return "cpu"
    return "cuda"


_DEVICE = _effective_device()


def _dtype() -> torch.dtype:
    if _DEVICE == "cpu":
        return torch.float32
    if _DEVICE == "mps":
        return torch.float16
    return torch.bfloat16


class _ModelSlot:
    def __init__(self) -> None:
        self._model: Any = None
        self._model_name: str | None = None
        self._lock = asyncio.Lock()
        self._timer: asyncio.Task | None = None

    def _cancel_timer(self) -> None:
        if self._timer and not self._timer.done():
            self._timer.cancel()
        self._timer = None

    def _schedule_unload(self) -> None:
        self._timer = asyncio.create_task(self._unload_after_timeout())

    async def _unload_after_timeout(self) -> None:
        await asyncio.sleep(settings.model_unload_timeout)
        async with self._lock:
            self._do_unload()

    def _do_unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._model_name = None
            if _DEVICE == "cuda":
                torch.cuda.empty_cache()

    def _load(self, model_name: str) -> None:
        self._model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=_dtype(),
            device_map=_DEVICE,
        )
        self._model_name = model_name

    def _run_transcribe(self, audio_path: str, language: str | None, context: str) -> Any:
        return self._model.transcribe(audio=audio_path, language=language, context=context)

    async def transcribe(
        self,
        model_name: str,
        audio_path: str,
        language: str | None,
        context: str = "",
    ) -> Any:
        self._cancel_timer()
        loop = asyncio.get_running_loop()
        async with self._lock:
            if self._model_name != model_name:
                self._do_unload()
                await loop.run_in_executor(None, self._load, model_name)
            result = await loop.run_in_executor(None, self._run_transcribe, audio_path, language, context)
        self._schedule_unload()
        return result


class _AlignerSlot:
    def __init__(self) -> None:
        self._model: Any = None
        self._lock = asyncio.Lock()
        self._timer: asyncio.Task | None = None

    def _cancel_timer(self) -> None:
        if self._timer and not self._timer.done():
            self._timer.cancel()
        self._timer = None

    def _schedule_unload(self) -> None:
        self._timer = asyncio.create_task(self._unload_after_timeout())

    async def _unload_after_timeout(self) -> None:
        await asyncio.sleep(settings.model_unload_timeout)
        async with self._lock:
            self._do_unload()

    def _do_unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            if _DEVICE == "cuda":
                torch.cuda.empty_cache()

    def _load(self) -> None:
        self._model = Qwen3ForcedAligner.from_pretrained(
            settings.aligner_model,
            dtype=_dtype(),
            device_map=_DEVICE,
        )

    def _run_align(self, audio_path: str, text: str, language: str) -> Any:
        return self._model.align(audio=audio_path, text=text, language=language)

    async def align(self, audio_path: str, text: str, language: str) -> Any:
        self._cancel_timer()
        loop = asyncio.get_running_loop()
        async with self._lock:
            if self._model is None:
                await loop.run_in_executor(None, self._load)
            result = await loop.run_in_executor(None, self._run_align, audio_path, text, language)
        self._schedule_unload()
        return result


asr_slot = _ModelSlot()
aligner_slot = _AlignerSlot()
