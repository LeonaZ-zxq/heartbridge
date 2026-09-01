"""视频/音频下载。yt-dlp 的薄封装。

设计要点（面试点：来源可信性是**代码保证**的，不是模型自觉）：

下载器返回的不只是音频文件，还有一份 `SourceMeta`——平台、作者、标题、URL。
这份元数据在蒸馏阶段由**代码**盖到每张卡的 source 字段上，
模型完全没有机会去"生成"一个来源。
把 provenance 变成管道的结构性产物，而不是 prompt 里的一条请求。
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

_URL = re.compile(r"^https?://", re.I)


class IngestionError(RuntimeError):
    pass


@dataclass
class SourceMeta:
    platform: str
    url: str | None = None
    author: str | None = None
    title: str | None = None
    duration_s: float | None = None
    date_ingested: date = field(default_factory=date.today)

    def to_source_dict(self) -> dict:
        return {
            "platform": self.platform,
            "url": self.url,
            "author": self.author,
            "title": self.title,
            "date_ingested": str(self.date_ingested),
        }


def _platform_of(url: str) -> str:
    host = url.split("/")[2].lower() if "//" in url else ""
    for key, name in (
        ("douyin", "douyin"), ("xiaohongshu", "xiaohongshu"), ("xhslink", "xiaohongshu"),
        ("bilibili", "bilibili"), ("b23.tv", "bilibili"), ("youtube", "youtube"), ("youtu.be", "youtube"),
    ):
        if key in host:
            return name
    return "web"


def download_audio(url: str, out_dir: Path) -> tuple[Path, SourceMeta]:
    """下载并抽取音频。需要系统里有 yt-dlp 和 ffmpeg。"""
    if not _URL.match(url):
        raise IngestionError(f"不是合法 URL: {url!r}")
    if shutil.which("yt-dlp") is None:
        raise IngestionError("找不到 yt-dlp。装一下：pip install yt-dlp")
    if shutil.which("ffmpeg") is None:
        raise IngestionError("找不到 ffmpeg。装一下：brew install ffmpeg")

    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "%(id)s.%(ext)s")

    # 先只取元数据，不下载。这样即使抽音频失败，我们也知道来源是什么。
    meta: dict = {}
    try:
        probe = subprocess.run(
            ["yt-dlp", "--dump-single-json", "--no-warnings", url],
            capture_output=True, text=True, timeout=120,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            meta = json.loads(probe.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        meta = {}

    source = SourceMeta(
        platform=_platform_of(url),
        url=meta.get("webpage_url") or url,
        author=meta.get("uploader") or meta.get("channel"),
        title=meta.get("title"),
        duration_s=meta.get("duration"),
    )

    proc = subprocess.run(
        ["yt-dlp", "-x", "--audio-format", "mp3", "--no-warnings", "-o", template, url],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        raise IngestionError(
            "yt-dlp 下载失败。小红书链接经常失效——"
            "兜底方案：手机录屏保存视频 → AirDrop 到 Mac → "
            "用 --audio 参数直接喂本地文件。\n" + proc.stderr[-500:]
        )

    audio = sorted(out_dir.glob("*.mp3"), key=lambda p: p.stat().st_mtime)
    if not audio:
        raise IngestionError("yt-dlp 报告成功但没有产出 mp3")
    return audio[-1], source


def local_audio(path: Path, platform: str = "local", author: str | None = None) -> tuple[Path, SourceMeta]:
    """本地文件入口（录屏兜底路径）。

    管道的输入契约从一开始就是「任意音频文件」，不是「一个能下载的链接」。
    这让平台反爬变成一个**可绕过的运营问题**，而不是架构级的失败点。
    """
    if not path.exists():
        raise IngestionError(f"文件不存在: {path}")
    return path, SourceMeta(platform=platform, author=author, title=path.stem)
