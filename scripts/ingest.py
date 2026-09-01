#!/usr/bin/env python3
"""摄取 CLI：把一个视频/音频/文字稿变成候选知识卡。

    # 从链接（抖音等，需要 yt-dlp + ffmpeg）
    python scripts/ingest.py --url "https://v.douyin.com/xxxx/"

    # 从本地音频（小红书录屏兜底路径）
    python scripts/ingest.py --audio ~/Desktop/clip.m4a

    # 从已有文字稿（跳过 ASR，调 prompt 时最常用）
    python scripts/ingest.py --transcript-file notes.txt

产出**不会自动进知识库**。它写到 data/distilled/ 下，人工看过再合并。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import CONFIG  # noqa: E402
from core.ingestion.pipeline import ingest  # noqa: E402
from core.knowledge.retrieval import build_retriever  # noqa: E402
from core.knowledge.schema import load_cards  # noqa: E402
from core.utils.llm import get_llm  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url")
    src.add_argument("--audio", type=Path)
    src.add_argument("--transcript-file", type=Path)
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--start-index", type=int, default=None,
                    help="新卡 id 起始序号。默认接在现有卡片之后")
    ap.add_argument("--no-dedup", action="store_true")
    args = ap.parse_args()

    existing = load_cards(CONFIG.cards_dir)
    start = args.start_index
    if start is None:
        comm = [int(c.id.split("_")[1]) for c in existing if c.id.startswith("comm_")]
        start = (max(comm) + 1) if comm else 1

    retriever = None if args.no_dedup else build_retriever(existing, backend="bm25")

    result = ingest(
        url=args.url,
        audio=args.audio,
        transcript_text=args.transcript_file.read_text(encoding="utf-8")
        if args.transcript_file else None,
        llm=get_llm(CONFIG),
        retriever=retriever,
        start_index=start,
        whisper_model=args.whisper_model,
    )

    print(f"来源: {result.source.platform} | {result.source.author or '?'} | {result.source.title or '?'}")
    print(result.report.summary())
    if result.transcript_path:
        print(f"文字稿:   {result.transcript_path}")
    if result.cards_path:
        print(f"新卡:     {result.cards_path}")
    if result.review_path:
        print(f"待复核:   {result.review_path}")
        for card, dup, score in result.report.needs_review:
            print(f"    ~ {card.scenario[:34]}  ≈ {dup} ({score})")
    for r in result.report.rejected:
        print(f"    ✗ 校验未通过: {r}")
    print("\n人工复核后，把新卡合并进 knowledge_base/cards/ 并重跑 pytest 与检索评测。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
