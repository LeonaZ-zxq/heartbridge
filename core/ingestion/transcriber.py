"""音频 → 文字稿。faster-whisper 的薄封装。

为什么本地跑 Whisper 而不是调云端 ASR API：
和 embedding 的理由一样——**隐私约束推导技术选型**。
本项目的原则是用户数据不出本机，那么语音转写也不能例外。
faster-whisper 在 Apple Silicon 上用 int8 量化跑 small 模型完全够用。

重依赖是懒加载的：没装 faster-whisper 不影响其余模块和测试套件。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Transcript:
    text: str
    language: str = "zh"
    duration_s: float | None = None

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text, encoding="utf-8")
        return path


def transcribe(
    audio_path: Path,
    model_size: str = "small",
    language: str = "zh",
    compute_type: str = "int8",
) -> Transcript:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise RuntimeError(
            "需要 faster-whisper：pip install faster-whisper"
        ) from exc

    model = WhisperModel(model_size, compute_type=compute_type)
    segments, info = model.transcribe(str(audio_path), language=language)
    text = "".join(seg.text for seg in segments).strip()
    return Transcript(text=text, language=info.language, duration_s=info.duration)
