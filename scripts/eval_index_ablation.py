#!/usr/bin/env python3
"""消融实验：索引文本里放哪些字段，对检索命中率影响有多大。

这个脚本回答一个具体的设计问题——
「RAG 里应该把整个文档扔进向量库，还是只索引其中一部分？」

我的假设：索引文本应该贴近**查询**的语言分布，而不是贴近**答案**的语言分布。
用户会说「他说自己不配被爱」（场景描述），不会说「先接住情绪不急着反驳」（技巧名）。
所以 scenario / symptom / aliases 这些字段的价值应该远高于 do / why_it_works。

跑一下就知道假设对不对。这就是把「我觉得」变成「我测过」的过程。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import CONFIG  # noqa: E402
from core.knowledge.evaluator import evaluate, load_eval_set  # noqa: E402
from core.knowledge.retrieval import BM25Retriever  # noqa: E402
from core.knowledge.schema import Card, CommunicationCard, SomaticCard, load_cards  # noqa: E402


def fields(card: Card, names: set[str]) -> str:
    """按字段名拼接索引文本。不存在的字段自动跳过（两种卡字段名不同）。"""
    out: list[str] = []
    for n in names:
        v = getattr(card, n, None)
        if v is None:
            continue
        out.extend(v if isinstance(v, list) else [str(v)])
    return " ".join(out)


VARIANTS: dict[str, set[str]] = {
    # 最朴素的做法：只索引「这张卡叫什么」
    "A 只有名称": {"technique_name", "symptom"},
    # 只索引场景描述 —— 最贴近用户查询的字段
    "B 只有场景": {"scenario", "symptom", "aliases"},
    # 场景 + 反例。反例里往往写着用户实际会说的原话
    "C 场景+反例": {"scenario", "symptom", "aliases", "dont", "avoid_saying"},
    # 场景 + 主题标签
    "D 场景+标签": {"scenario", "symptom", "aliases", "tags"},
    # 场景 + 例句（例句是「你该说的话」，语言分布偏答案侧）
    "D2 场景+例句": {"scenario", "symptom", "aliases", "example_phrases", "say"},
    # 当前生产配置 index_text()
    "P 生产配置": set(),
    # 全字段：把整张卡都扔进索引（很多人默认的做法）
    "E 全字段": {
        "scenario", "technique_name", "do", "dont", "example_phrases", "why_it_works",
        "symptom", "aliases", "what_it_is", "in_person", "remote", "say",
        "avoid_saying", "seek_help_if", "tags",
    },
}


def main() -> int:
    cards = load_cards(CONFIG.cards_dir)
    queries = load_eval_set(Path(__file__).parent.parent / "tests/fixtures/retrieval_eval.json")
    para = [q for q in queries if q.get("paraphrase")]

    print(f"索引文本消融实验 | 后端 BM25 | {len(cards)} 张卡 | {len(queries)} 条查询\n")
    print(f"{'变体':<14}{'Recall@1':>10}{'Recall@3':>10}{'MRR':>8}{'改写式R@3':>12}")
    print("-" * 54)
    for label, names in VARIANTS.items():
        fn = None if not names else (lambda c, n=names: fields(c, n))
        r = BM25Retriever(cards, text_fn=fn)
        res = evaluate(r, queries)
        sub = evaluate(r, para)
        print(f"{label:<14}{res.recall_at_1:>9.1%}{res.recall_at_3:>10.1%}"
              f"{res.mrr:>8.3f}{sub.recall_at_3:>11.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
