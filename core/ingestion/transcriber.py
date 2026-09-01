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
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        # Whisper 的中文默认输出**不带标点、且常常给繁体**。这不只是不好看：
        # 下游分块器按 。！？ 找句子边界，没有标点就整篇是一个"句子"，
        # 分块直接退化成不切。initial_prompt 是唯一能同时把这两件事扳回来的
        # 旋钮——它给解码器一个风格样例，让它照着这个样子往下写。
        initial_prompt=(
            "以下是普通话的口语内容，请使用简体中文转写，并加上标点符号。"
            if language == "zh" else None
        ),
        # 短视频几乎都有背景音乐。VAD 先把非语音段切掉，
        # 能明显减少模型对着音乐硬编词句的情况。
        vad_filter=True,
    )
    text = "".join(seg.text for seg in segments).strip()
    return Transcript(text=text, language=info.language, duration_s=info.duration)
