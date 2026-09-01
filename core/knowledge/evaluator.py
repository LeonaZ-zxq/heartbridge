"""检索评测。

为什么必须有这个文件（面试里几乎必被问「你怎么知道你的 RAG 好不好」）：

没有评测集的 RAG 系统，所有的「我觉得挺准的」都是自我安慰。
评测集把「调参」从玄学变成实验：改了 index_text 的字段组合、换了 embedding
模型、调了 RRF 的 k，命中率是升是降立刻可见。

指标选择：
- Recall@k：前 k 个结果里有没有正确的卡。**这是最贴近业务的指标**——
  下游生成器会拿到 top-k 全部内容，只要正确的卡在里面，生成就有依据。
- MRR (Mean Reciprocal Rank)：正确答案排第几的倒数的平均。
  Recall@3 只告诉你「在不在前三」，MRR 还能区分「排第一」和「排第三」。
  两个一起看，才能发现「命中率没变但排序变差了」这种退化。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from core.knowledge.retrieval import Retriever


@dataclass
class EvalResult:
    backend: str
    n: int
    recall_at_1: float
    recall_at_3: float
    mrr: float
    failures: list[dict]

    def summary(self) -> str:
        return (
            f"{self.backend:<8} n={self.n:<3} "
            f"Recall@1={self.recall_at_1:.1%}  "
            f"Recall@3={self.recall_at_3:.1%}  "
            f"MRR={self.mrr:.3f}"
        )


def load_eval_set(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["queries"]


def evaluate(retriever: Retriever, queries: list[dict], k: int = 3) -> EvalResult:
    hit1 = hitk = 0
    rr_total = 0.0
    failures: list[dict] = []
    for item in queries:
        hits = retriever.search(item["q"], k=k)
        ids = [h.id for h in hits]
        gold = item["expect"]
        if ids and ids[0] == gold:
            hit1 += 1
        if gold in ids:
            hitk += 1
            rr_total += 1.0 / (ids.index(gold) + 1)
        else:
            failures.append({"q": item["q"], "expect": gold, "got": ids,
                             "paraphrase": item.get("paraphrase", False)})
    n = len(queries)
    return EvalResult(
        backend=getattr(retriever, "name", "?"),
        n=n,
        recall_at_1=hit1 / n if n else 0.0,
        recall_at_3=hitk / n if n else 0.0,
        mrr=rr_total / n if n else 0.0,
        failures=failures,
    )
