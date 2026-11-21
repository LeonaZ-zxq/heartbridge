#!/usr/bin/env python3
"""跑检索评测，对比各个后端。

用法:
    python scripts/eval_retrieval.py                       # 全部后端（dense 需装 sentence-transformers）
    python scripts/eval_retrieval.py --backends bm25
    python scripts/eval_retrieval.py --backends rerank --set both

    # T1-1 的对照：精确检索 vs HNSW 近似检索，召回和延迟一起看
    python scripts/eval_retrieval.py --backends dense --dense-index flat --latency 20 --set holdout
    python scripts/eval_retrieval.py --backends dense --dense-index hnsw --latency 20 --set holdout

注意：--dense-index flat 是**精确**检索，召回率必须和 numpy 完全一致；
对不上说明实现有 bug，不是近似误差。只有 hnsw 才允许掉召回。
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
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
    ap.add_argument("--dense-index", default=None, choices=["numpy", "flat", "hnsw"],
                    help="向量索引实现。flat/hnsw 需装 faiss-cpu")
    ap.add_argument("--latency", type=int, default=0, metavar="N",
                    help="每条查询重复 N 次测延迟（默认 0 = 不测）")
    ap.add_argument("--set", default="dev",
                    choices=["dev", "holdout", "test", "both", "all"],
                    help="dev=开发集（调参用，已污染）; holdout=留出集（选型用，已污染）; "
                         "test=测试集（只报告，碰了就作废）; both=dev+holdout; all=三份都跑")
    args = ap.parse_args()

    cards = load_cards(CONFIG.cards_dir)
    fx = Path(__file__).parent.parent / "tests/fixtures"
    sets = {"dev": "retrieval_eval.json",
            "holdout": "retrieval_holdout.json",
            "test": "retrieval_test.json"}
    if args.set == "both":
        chosen = ["dev", "holdout"]        # 保持旧行为不变
    elif args.set == "all":
        chosen = list(sets)
    else:
        chosen = [args.set]
    if "test" in chosen:
        print("⚠️  test 集只用于**报告**。任何基于它的选型、调参或诊断都会让它失效，\n"
              "    届时必须作废重做，而不是继续引用它的数字。见 docs/EVALUATION.md §5。\n")
    for name in chosen:
        _run(cards, load_eval_set(fx / sets[name]), name, args)
    return 0


def _run(cards, queries, set_name, args):
    print(f"[{set_name}] 知识库 {len(cards)} 张卡 | {len(queries)} 条查询 "
          f"（改写式 {sum(q.get('paraphrase', False) for q in queries)} 条）")

    results = []
    for backend in args.backends:
        cfg = CONFIG if not args.dense_index else replace(CONFIG, dense_index=args.dense_index)
        try:
            r = build_retriever(cards, backend=backend, cfg=cfg)
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
        if args.latency:
            # 先跑一次把模型和索引都加载好，不然第一条查询会把权重加载的时间
            # 算进延迟里，测出来的是冷启动不是检索。
            r.search(queries[0]["q"], k=args.k)
            times = []
            for item in queries:
                for _ in range(args.latency):
                    t0 = time.perf_counter()
                    r.search(item["q"], k=args.k)
                    times.append((time.perf_counter() - t0) * 1000)
            times.sort()
            p50 = times[len(times) // 2]
            p95 = times[min(int(len(times) * 0.95), len(times) - 1)]
            label = f"{backend}/{args.dense_index}" if args.dense_index else backend
            print(f"{'':<8} └─ 延迟 {label} (n={len(times)}): "
                  f"p50={p50:.2f}ms  p95={p95:.2f}ms  mean={sum(times)/len(times):.2f}ms")
        if args.show_failures and res.failures:
            for f in res.failures:
                print(f"    ✗ {f['q']}  期望 {f['expect']}  实际 {f['got']}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
