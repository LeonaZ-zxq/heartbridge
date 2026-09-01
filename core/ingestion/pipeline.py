"""端到端摄取管道：链接（或本地音频）→ 音频 → 文字稿 → 校验过的卡片。

管道的输入契约是「任意音频文件」而不是「一个能下载的链接」。
这是刻意的：小红书之类平台的下载经常失效，把录屏兜底路径设计进契约里，
平台反爬就只是一个运营麻烦，而不是架构级的失败点。

输出两个文件：
    <slug>.json              通过校验、且不与现有知识库重复的新卡
    <slug>.needs_review.json 疑似重复的卡 + 撞上了哪张、相似度多少
人工看完 needs_review 再决定合并还是丢弃。**没有任何一步是自动入库的。**
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from core.config import CONFIG, Config
from core.ingestion.distiller import DistillReport, cards_to_json, distill
from core.ingestion.downloader import SourceMeta, download_audio, local_audio
from core.ingestion.transcriber import Transcript, transcribe
from core.knowledge.retrieval import Retriever
from core.utils.llm import LLMProvider


@dataclass
class IngestResult:
    source: SourceMeta
    transcript_path: Path | None
    cards_path: Path | None
    review_path: Path | None
    report: DistillReport


def _slug(meta: SourceMeta) -> str:
    base = (meta.title or meta.author or meta.platform or "clip")[:40]
    return re.sub(r"[^\w一-鿿-]+", "_", base).strip("_") or "clip"


def ingest(
    *,
    url: str | None = None,
    audio: Path | None = None,
    transcript_text: str | None = None,
    llm: LLMProvider,
    retriever: Retriever | None = None,
    start_index: int = 1,
    cfg: Config | None = None,
    whisper_model: str = "small",
    meta: SourceMeta | None = None,
    evidence_tier: str = "lived_experience",
) -> IngestResult:
    cfg = cfg or CONFIG
    cfg.ensure_dirs()

    # ── 1. 拿到音频与来源元数据 ──
    if transcript_text is not None:
        # 调用方能说清来源就用它的。说不清才退回 "manual"——
        # 但 platform="manual" 是一个**信号**：这批卡的 provenance 是断的，
        # 合并进知识库前应该先看一眼。来源盖章是代码的职责，
        # 不能因为走了"已有文字稿"这条捷径就悄悄丢掉。
        meta = meta or SourceMeta(platform="manual", title="pasted transcript")
        tx = Transcript(text=transcript_text)
        tx_path = None
    else:
        if url:
            audio_path, probed = download_audio(url, cfg.private_dir / "audio")
            meta = meta or probed
        elif audio:
            audio_path, probed = local_audio(audio)
            meta = meta or probed
        else:
            raise ValueError("url / audio / transcript_text 至少给一个")
        tx = transcribe(audio_path, model_size=whisper_model)
        tx_path = tx.save(cfg.transcripts_dir / f"{_slug(meta)}.txt")

    # ── 2. 蒸馏。来源由代码盖章 ──
    report = distill(
        tx.text, llm, meta.to_source_dict(),
        start_index=start_index, retriever=retriever,
        evidence_tier=evidence_tier,
    )

    # ── 3. 落盘。新卡和待复核卡分开 ──
    out_dir = cfg.private_dir / "distilled"
    out_dir.mkdir(parents=True, exist_ok=True)
    cards_path = review_path = None

    if report.cards:
        cards_path = out_dir / f"{_slug(meta)}.json"
        cards_path.write_text(cards_to_json(report.cards), encoding="utf-8")

    if report.needs_review:
        review_path = out_dir / f"{_slug(meta)}.needs_review.json"
        payload = [
            {
                "card": json.loads(card.model_dump_json()),
                "duplicate_of": dup_id,
                "similarity": score,
            }
            for card, dup_id, score in report.needs_review
        ]
        review_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    return IngestResult(meta, tx_path, cards_path, review_path, report)
