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



MEDIA_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi",
             ".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg"}
TEXT_EXT = {".txt", ".md"}


def run_batch(args) -> int:
    """把一整个文件夹跑完。

    为什么单独写一个批量入口，而不是让用户 for 循环调单个：
    **卡片 id 必须全局唯一且连续**。一个一个手动跑，每次都要自己算
    `--start-index`，算错就会产生重复 id，而重复 id 会让检索里出现
    两张同号卡——这种错很难在事后发现。批量模式在进程内维护这个计数器，
    顺便让去重索引在每处理完一个文件后就更新，
    这样**同一批素材内部的重复也能被抓到**（同一个博主在两条视频里
    讲同一个技巧，是最常见的情况）。
    """
    files = sorted(
        f for f in args.dir.iterdir()
        if f.is_file() and f.suffix.lower() in (MEDIA_EXT | TEXT_EXT)
    )
    if not files:
        print(f"{args.dir} 里没有可处理的文件。"
              f"\n  支持：{' '.join(sorted(MEDIA_EXT | TEXT_EXT))}")
        return 1

    existing = load_cards(CONFIG.cards_dir)
    comm = [int(c.id.split("_")[1]) for c in existing if c.id.startswith("comm_")]
    next_index = args.start_index or ((max(comm) + 1) if comm else 1)

    llm = get_llm(CONFIG)
    pool = list(existing)          # 累积：包含本批已经产出的新卡
    total_new = total_review = total_rejected = 0

    print(f"批量处理 {len(files)} 个文件，新卡 id 从 comm_{next_index:03d} 开始\n")
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}")
        retriever = None if args.no_dedup else build_retriever(pool, backend="bm25")
        try:
            is_text = f.suffix.lower() in TEXT_EXT
            result = ingest(
                audio=None if is_text else f,
                transcript_text=f.read_text(encoding="utf-8") if is_text else None,
                llm=llm,
                retriever=retriever,
                start_index=next_index,
                whisper_model=args.whisper_model,
            )
        except Exception as exc:   # noqa: BLE001
            # 一个文件失败不该让整批停下来——素材质量参差是常态。
            print(f"    ✗ 跳过：{exc}\n")
            continue

        new_cards = result.report.cards
        print("    " + result.report.summary())
        if result.cards_path:
            print(f"    → {result.cards_path}")
        for card, dup, score in result.report.needs_review:
            print(f"    ~ 疑似重复 {card.scenario[:28]} ≈ {dup} ({score})")

        pool.extend(new_cards)
        next_index += len(new_cards) + len(result.report.needs_review)
        total_new += len(new_cards)
        total_review += len(result.report.needs_review)
        total_rejected += len(result.report.rejected)
        print()

    print(f"━━ 完成：新卡 {total_new} 张，待复核 {total_review} 张，校验未通过 {total_rejected} 张")
    print(f"产出在 {CONFIG.private_dir / 'distilled'}/，**人工复核后**再合并进 "
          f"knowledge_base/cards/")
    print("合并完记得重跑：pytest -q && python scripts/eval_retrieval.py "
          "--backends bm25 --set both")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url")
    src.add_argument("--audio", type=Path)
    src.add_argument("--transcript-file", type=Path)
    src.add_argument("--dir", type=Path,
                     help="批量：把这个目录下所有音视频/文字稿依次处理")
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--start-index", type=int, default=None,
                    help="新卡 id 起始序号。默认接在现有卡片之后")
    ap.add_argument("--no-dedup", action="store_true")
    args = ap.parse_args()

    if args.dir:
        return run_batch(args)

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
