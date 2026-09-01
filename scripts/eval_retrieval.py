#!/usr/bin/env python3
"""跑检索评测，对比三种后端。

用法:
    python scripts/eval_retrieval.py                # 全部后端（dense 需装 sentence-transformers）
    python scripts/eval_retrieval.py --backends bm25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import CONFIG  # noqa: E402
from core.knowledge.evaluator import evaluate, load_eval_set  # noqa: E402
from core.knowledge.retrieval import build_retriever  # noqa: E402
from core.knowledge.schema import load_cards  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="*", default=["bm25", "dense", "hybrid"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--show-failures", action="store_true")
    ap.add_argument("--set", default="dev", choices=["dev", "holdout", "both"],
                    help="dev=开发集（写文档扩展时参考过，分数偏乐观）; holdout=留出集（诚实数字）")
    args = ap.parse_args()

    cards = load_cards(CONFIG.cards_dir)
    fx = Path(__file__).parent.parent / "tests/fixtures"
    sets = {"dev": "retrieval_eval.json", "holdout": "retrieval_holdout.json"}
    chosen = list(sets) if args.set == "both" else [args.set]
    for name in chosen:
        _run(cards, load_eval_set(fx / sets[name]), name, args)
    return 0


def _run(cards, queries, set_name, args):
    print(f"[{set_name}] 知识库 {len(cards)} 张卡 | {len(queries)} 条查询 "
          f"（改写式 {sum(q.get('paraphrase', False) for q in queries)} 条）")

    results = []
    for backend in args.backends:
        try:
            r = build_retriever(cards, backend=backend)
            res = evaluate(r, queries, k=args.k)
        except ImportError as exc:
            print(f"{backend:<8} 跳过（缺依赖: {exc}）")
            continue
        results.append(res)
        print(res.summary())
        # 改写式查询单独看：这是检验语义检索价值的子集
        para = [q for q in queries if q.get("paraphrase")]
        if para:
            sub = evaluate(r, para, k=args.k)
            print(f"{'':<8} └─ 仅改写式查询 (n={sub.n}): Recall@3={sub.recall_at_3:.1%}")
        if args.show_failures and res.failures:
            for f in res.failures:
                print(f"    ✗ {f['q']}  期望 {f['expect']}  实际 {f['got']}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
