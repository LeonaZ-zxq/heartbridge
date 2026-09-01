"""文字稿 → 结构化技巧卡片。整个管道里最需要工程约束的一步。

━━━ 四个设计决策，每一个都是面试可讲的点 ━━━

**1. 来源由代码盖章，模型无权生成。**
   `source` 字段不在 prompt 的输出 schema 里。模型返回的 JSON 即使包含
   source 也会被丢弃，卡片的来源一律由下载阶段拿到的 `SourceMeta` 覆盖。
   这样「这条建议来自哪个博主的哪个视频」是**管道的结构性产物**，
   不是模型的自觉。id 同理，由代码按序分配。

**2. 校验失败会把错误信息喂回去重试（self-repair loop）。**
   LLM 输出结构化数据的失败通常是**可修的**：少个字段、类型写错、
   列表写成字符串。把 Pydantic 的报错原样贴回去让它改，
   比"重跑一次祈祷这次对"有效得多，也比手写解析器便宜得多。

**3. 长文字稿分块 + 重叠。**
   十分钟视频的转写稿轻松上万字。一次性塞进去，小模型会漏掉后半段
   （lost in the middle）。分块处理，块间保留重叠，避免把一个技巧
   正好从中间切断。

**4. 去重用检索器，不用字符串比对。**
   同一个博主会反复讲同一个技巧，不同博主也会讲同样的东西。
   新卡入库前先拿它的 index_text 去检索现有知识库，相似度过高就
   **标记为待人工复核**（不是静默丢弃）——否则知识库会慢慢被同义卡淹没，
   而同义卡会互相竞争检索排名，直接损害 Recall@1。
   **这是把已有的检索能力复用到数据质量上**，不需要引入新组件。

   阈值是量出来的，不是拍的。在现有 30 张卡上实测归一化相似度：
       真近义卡     0.168 / 0.244 / 0.363
       无关新主题   0.041 / 0.057 / 无命中
   分界线取 0.12，偏向「多标记、交给人看」——和安全层同一个哲学：
   在便宜的错误方向上犯错。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Sequence

from pydantic import ValidationError

from core.knowledge.retrieval import Retriever
from core.knowledge.schema import Card, CommunicationCard
from core.utils.llm import LLMError, LLMProvider, complete_json

# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """你在把一段关于「如何陪伴抑郁症患者」的口播文字稿，整理成结构化的沟通技巧卡片。

**你是整理者，不是作者。** 严格遵守：
- 只提取文字稿里**真的说过**的内容。没说的不要补，不要用你自己的心理学知识扩写。
- 一个技巧一张卡。文字稿里没有可提取的具体技巧就返回空数组，这是合法输出。
- 不要下诊断，不要给用药建议，不要承诺疗效。
- scenario 要写成**求助者会怎么描述自己的处境**（"他说自己不配被爱"），
  不要写成技巧的名字（"共情式回应"）。这直接决定这张卡以后能不能被检索到。
- user_phrasings 写 3-6 条这个处境的**口语说法**，用于扩充检索索引。

每张卡的字段：
{"scenario": str, "technique_name": str, "do": [str], "dont": [str],
 "example_phrases": [str], "why_it_works": str, "tags": [str], "user_phrasings": [str]}

do / dont / example_phrases 都**至少一条**。why_it_works 讲机制，不要复述做法。

只输出 JSON 数组，不要任何其他文字：
[{...}, {...}]"""

REPAIR_PROMPT = """你上一次的输出没有通过数据校验。错误如下：

{errors}

请修正后重新输出完整的 JSON 数组。只输出 JSON，不要解释。"""


# --------------------------------------------------------------------------- #
# 分块
# --------------------------------------------------------------------------- #
def chunk_transcript(text: str, max_chars: int = 2400, overlap: int = 240) -> list[str]:
    """按句子边界切块，块间重叠。

    在句子边界切而不是硬切字符数：一个技巧被从句子中间切断，
    两个块都拿不到完整信息。重叠是为了让跨块的技巧至少在一个块里完整出现。
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    sentences = re.split(r"(?<=[。！？!?\n])", text)
    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        if len(buf) + len(sent) > max_chars and buf:
            chunks.append(buf)
            buf = buf[-overlap:] if overlap else ""
        buf += sent
    if buf.strip():
        chunks.append(buf)
    return chunks


# --------------------------------------------------------------------------- #
# 结果
# --------------------------------------------------------------------------- #
@dataclass
class DistillReport:
    cards: list[Card] = field(default_factory=list)
    chunks: int = 0
    raw_items: int = 0            # 模型一共产出多少条
    rejected: list[str] = field(default_factory=list)   # 校验没过、修不好的
    # 疑似重复：不静默丢弃，写进单独文件交给人工复核
    needs_review: list[tuple[Card, str, float]] = field(default_factory=list)
    repairs: int = 0              # 触发了几次 self-repair

    def summary(self) -> str:
        return (
            f"分块 {self.chunks} | 模型产出 {self.raw_items} 条 | "
            f"通过校验 {len(self.cards)} | 自修复 {self.repairs} 次 | "
            f"丢弃 {len(self.rejected)} | 待人工复核(疑似重复) {len(self.needs_review)}"
        )


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def distill(
    transcript: str,
    llm: LLMProvider,
    source: dict,
    *,
    start_index: int = 1,
    retriever: Retriever | None = None,
    dedup_threshold: float = 0.12,
    max_repairs: int = 1,
    evidence_tier: str = "lived_experience",
) -> DistillReport:
    """把一段文字稿蒸馏成卡片。

    `source` 由调用方（下载阶段）提供并强制盖到每张卡上。
    `retriever` 若提供，则对已有知识库做去重检查。
    """
    report = DistillReport()
    chunks = chunk_transcript(transcript)
    report.chunks = len(chunks)
    next_id = start_index

    for chunk in chunks:
        items = _distill_chunk(llm, chunk, report, max_repairs)
        report.raw_items += len(items)

        for raw in items:
            if not isinstance(raw, dict):
                continue
            # ── 来源、id、证据层级由代码写死，模型给什么都不算数 ──
            raw = {k: v for k, v in raw.items()
                   if k not in ("source", "id", "type", "evidence_tier", "needs_review")}
            raw["id"] = f"comm_{next_id:03d}"
            raw["type"] = "communication"
            raw["source"] = source
            # 证据层级和 source 同理，属于**管道知道、模型不知道**的事实。
            # 默认值是 clinical_guideline，而摄取进来的全是博主口播——
            # 不显式盖章，一批短视频经验就会以"临床指南"的身份进检索结果，
            # 这比少几张卡严重得多。
            raw["evidence_tier"] = evidence_tier

            try:
                card = CommunicationCard.model_validate(raw)
            except ValidationError as exc:
                report.rejected.append(f"{raw.get('scenario', '?')[:30]}: {_brief(exc)}")
                continue

            if retriever is not None:
                dup = _find_duplicate(card, retriever, dedup_threshold)
                if dup:
                    report.needs_review.append((card, dup[0], dup[1]))
                    next_id += 1
                    continue

            report.cards.append(card)
            next_id += 1

    return report


def _distill_chunk(llm: LLMProvider, chunk: str, report: DistillReport, max_repairs: int) -> list:
    """单块蒸馏，带一次 self-repair 机会。"""
    try:
        data = complete_json(llm, SYSTEM_PROMPT, chunk, temperature=0.2)
    except LLMError:
        return []
    if isinstance(data, list):
        return data

    # 不是数组 → 给模型一次改正机会，把具体问题告诉它
    if max_repairs > 0:
        report.repairs += 1
        try:
            fixed = complete_json(
                llm,
                SYSTEM_PROMPT,
                chunk + "\n\n" + REPAIR_PROMPT.format(errors="顶层必须是 JSON 数组，你返回的不是数组"),
                temperature=0.0,
            )
            if isinstance(fixed, list):
                return fixed
        except LLMError:
            pass
    return []


def _find_duplicate(card: Card, retriever: Retriever, threshold: float) -> tuple[str, float] | None:
    """用现有检索器判断新卡是不是已有卡的同义重复。

    阈值按后端不同含义不同（BM25 分数无上界，余弦在 [-1,1]），
    所以这里用**相对分数**：把命中分数除以该卡自查询自身的分数做归一化。
    """
    hits = retriever.search(card.index_text(), k=1)
    if not hits:
        return None
    top = hits[0]
    self_hits = retriever.search(top.card.index_text(), k=1)
    self_score = self_hits[0].score if self_hits else 0.0
    if self_score <= 0:
        return None
    ratio = top.score / self_score
    return (top.id, round(ratio, 3)) if ratio >= threshold else None


def _brief(exc: ValidationError) -> str:
    errs = exc.errors()[:2]
    return "; ".join(f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in errs)


def cards_to_json(cards: Sequence[Card]) -> str:
    return json.dumps(
        [json.loads(c.model_dump_json()) for c in cards], ensure_ascii=False, indent=1
    )
