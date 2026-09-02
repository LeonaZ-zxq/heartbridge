#!/usr/bin/env python3
"""命令行问一句话，看检索返回哪几张卡。**不调用任何 LLM。**

    python scripts/ask.py "她已读不回好几天了，我不知道该不该继续发消息"
    python scripts/ask.py "他说心口像被石头压着" --type somatic
    python scripts/ask.py "她说她不想活了" --backend bm25 --k 5

为什么值得单独有这个入口：

1. **验收检索不该需要 LLM。** 网页版最主要那页要调模型生成回复，于是
   「检索准不准」和「模型答得好不好」被绑在了一起——免费额度用完的时候，
   你连检索有没有坏都验证不了。这个脚本把两件事拆开：它只跑检索。

2. **复核卡片最省事的方式是按查询看，而不是按文件看。** 一张卡写得好不好，
   取决于它会在什么问题下被召回。翻 JSON 看不出这个，问一句话就看得出。

3. **它是评测集的前置工序。** 第三份评测集要标「这个查询的正确答案是哪张卡」，
   而人是想不出查询的——得先看到检索结果，才知道自己想问什么。
   注意：**用它探索出来的查询不能直接进评测集**，那又是照着检索结果写标注。
   它的作用是帮你熟悉知识库覆盖了什么，然后**合上它**再去写查询。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import CONFIG  # noqa: E402
from core.knowledge.retrieval import build_retriever  # noqa: E402
from core.knowledge.schema import load_cards  # noqa: E402

_TIER_LABEL = {
    "clinical_guideline": "临床指南",
    "practitioner": "从业者",
    "lived_experience": "亲历经验",
}


def _provenance(card) -> str:
    """一张卡的来源与分量，一行说清。

    证据层级要和来源一起显示：「这条建议出自 Beyond Blue」和
    「这条出自某个博主」是两种不同的东西，看的人有权当场知道自己在看哪一种。
    """
    s = card.source
    who = s.authority or s.author or s.platform or "?"
    tier = _TIER_LABEL.get(getattr(card, "evidence_tier", ""), getattr(card, "evidence_tier", "?"))
    flag = " ⚠未人工复核" if getattr(card, "needs_review", False) else ""
    return f"[{tier}] {who}{flag}"


def _show(hit, rank: int, verbose: bool) -> None:
    c = hit.card
    print(f"\n{'━' * 66}")
    title = getattr(c, "technique_name", None) or getattr(c, "symptom", "")
    print(f"{rank}. {c.id}  (score {hit.score:.3f})  {title}")
    print(f"   {_provenance(c)}")

    if c.type == "communication":
        print(f"\n   场景：{c.scenario}")
        print("\n   这样做：")
        for x in (c.do if verbose else c.do[:3]):
            print(f"     · {x}")
        print("\n   不要：")
        for x in (c.dont if verbose else c.dont[:2]):
            print(f"     ✗ {x}")
        print("\n   可以直接说：")
        for x in (c.example_phrases if verbose else c.example_phrases[:2]):
            print(f"     「{x}」")
        if verbose:
            print(f"\n   为什么有效：{c.why_it_works}")
    else:  # somatic
        print(f"\n   口语说法：{ '、'.join(c.aliases) }")
        print(f"\n   这是什么：{c.what_it_is}")
        print("\n   在他身边：")
        for x in (c.in_person if verbose else c.in_person[:3]):
            print(f"     · {x}")
        print("\n   可以直接说：")
        for x in (c.say if verbose else c.say[:2]):
            print(f"     「{x}」")
        # 就医条件永远完整显示，不截断：这是躯体化卡里唯一不能少看一条的部分
        print("\n   ⚠ 这些情况必须就医／走危机流程：")
        for x in c.seek_help_if:
            print(f"     ! {x}")


def main() -> int:
    ap = argparse.ArgumentParser(description="问一句话，看检索返回哪几张卡（不调 LLM）")
    ap.add_argument("query", help="用户会怎么说这件事，用口语写")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--backend", default=None,
                    help="bm25 / dense / hybrid，默认读配置")
    ap.add_argument("--type", dest="type_filter", default=None,
                    choices=["communication", "somatic"],
                    help="只在某一类卡里检索。躯体化科普那一页就是这么做的")
    ap.add_argument("-v", "--verbose", action="store_true", help="显示完整卡片")
    args = ap.parse_args()

    cards = load_cards(CONFIG.cards_dir)
    backend = args.backend or CONFIG.retrieval_backend
    retriever = build_retriever(cards, backend=backend)
    hits = retriever.search(args.query, k=args.k, type_filter=args.type_filter)

    scope = f"，限定 {args.type_filter}" if args.type_filter else ""
    print(f"问：{args.query}")
    print(f"（{len(cards)} 张卡 | {backend}{scope}）")

    if not hits:
        print("\n没有检索到任何卡片。")
        print("这本身是一条信息：要么这个处境知识库还没覆盖，要么索引文本和你的说法对不上。")
        return 1

    for i, h in enumerate(hits, 1):
        _show(h, i, args.verbose)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
