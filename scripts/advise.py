#!/usr/bin/env python3
"""HeartBridge CLI：粘贴一段聊天记录，拿到带解释的回复选项。

用法：
    # demo 模式（假 LLM，不需要 API key，用来看流程和跑演示）
    python scripts/advise.py --demo --text "小鱼 23:45
    我觉得我就是个废物"

    # 真实模式（需要 .env 里配好 HB_LLM_PROVIDER 和 API key）
    python scripts/advise.py --file chat.txt --profile examples/sample_profile.json

    # 从标准输入读（方便直接粘贴）
    pbpaste | python scripts/advise.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import CONFIG  # noqa: E402
from core.engine.pipeline import advise  # noqa: E402
from core.knowledge.retrieval import build_retriever  # noqa: E402
from core.knowledge.schema import load_cards  # noqa: E402
from core.profile.models import PartnerProfile  # noqa: E402
from core.utils.llm import MockProvider, get_llm  # noqa: E402

DEMO_REPLY = {
    "options": [
        {"text": "我听到了。你现在真的这么觉得，那一定很沉。",
         "why": "先验证情绪、不反驳。抑郁认知下直接反驳会被读成'你不理解我'，反而增加孤立感。",
         "card_id": "comm_001", "style": "共情"},
        {"text": "你不用现在就相信我说的。我就在这儿，不走。",
         "why": "把'相信'的压力拿掉，只承诺在场。稳定的在场比说服更能反驳'所有人都会走'。",
         "card_id": "comm_001", "style": "陪伴"},
        {"text": "先不聊这个了好不好，我给你放首歌，你躺着听。",
         "why": "温和打断反刍循环，把注意力从自我评判转到当下的身体和环境。",
         "card_id": "comm_007", "style": "转移"},
    ]
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--file", type=Path)
    ap.add_argument("--situation", default="", help="补充描述，比如'他在国内，我只能打字'")
    ap.add_argument("--profile", type=Path)
    ap.add_argument("--backend", default="bm25", help="bm25 | dense | hybrid")
    ap.add_argument("--demo", action="store_true", help="用假 LLM，不需要 API key")
    args = ap.parse_args()

    raw = args.text or (args.file.read_text(encoding="utf-8") if args.file else sys.stdin.read())
    if not raw.strip():
        print("没有输入。用 --text / --file，或者把聊天记录管道进来。", file=sys.stderr)
        return 2

    profile = None
    if args.profile:
        data = json.loads(args.profile.read_text(encoding="utf-8"))
        data.pop("_note", None)
        profile = PartnerProfile.model_validate(data)

    if args.demo:
        llm = MockProvider()
        llm.register("硬性要求", lambda s, u: json.dumps(DEMO_REPLY, ensure_ascii=False))
        llm.register("危机信号识别器", lambda s, u: json.dumps({"level": "none", "reason": "demo"}))
    else:
        llm = get_llm(CONFIG)

    retriever = build_retriever(load_cards(CONFIG.cards_dir), backend=args.backend)
    print(advise(raw, retriever, llm, profile, situation=args.situation).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
