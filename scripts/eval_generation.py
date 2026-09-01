#!/usr/bin/env python3
"""生成质量评测：盲测 A/B（有知识库 vs 无知识库）+ LLM-as-judge。

三步走：

    # 1) 生成候选（需要真实 LLM，会调用 API）
    python scripts/eval_generation.py generate

    # 2) 人工盲评：打开生成的 rating_sheet.md，逐条填 Y/N
    #    ——这是唯一的验收标准，judge 只是哨兵

    # 3) 汇总
    python scripts/eval_generation.py score
    python scripts/eval_generation.py judge     # 可选，LLM 打分做交叉参考

产出在 data/generation_eval/ 下（已 gitignore，因为里面有模型输出，不是代码）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import CONFIG  # noqa: E402
from core.engine.generator import generate_options  # noqa: E402
from core.evaluation.rubric import (  # noqa: E402
    PRIMARY_QUESTION, RUBRIC, BlindItem, build_blind_sheet, dump_key,
    judge_option, score_blind_sheet,
)
from core.knowledge.retrieval import build_retriever  # noqa: E402
from core.knowledge.schema import load_cards  # noqa: E402
from core.profile.models import PartnerProfile  # noqa: E402
from core.safety.detector import RiskLevel, rule_scan  # noqa: E402
from core.utils.llm import CassetteProvider, get_llm  # noqa: E402
from core.utils.text import parse_transcript  # noqa: E402

OUT = CONFIG.private_dir / "generation_eval"
FIXTURE = Path(__file__).parent.parent / "tests/fixtures/generation_eval.json"


def _profile() -> PartnerProfile:
    data = json.loads((Path(__file__).parent.parent / "examples/sample_profile.json")
                      .read_text(encoding="utf-8"))
    data.pop("_note", None)
    return PartnerProfile.model_validate(data)


def _preflight(llm) -> None:
    """先打一发真实请求，确认 LLM 通道是通的。

    为什么要这一步：`generate_options` 在 LLMError 时返回空列表（对线上是
    正确的降级——宁可不给建议，也不给编的建议），但对评测脚本来说，
    「全部失败」和「模型认真回答了但都不合格」会产出一模一样的空评分表。
    评测工具必须能区分**系统坏了**和**系统表现差**，否则一份 0/0 的报告
    会被误读成结论。所以这里让通道故障**大声失败**。
    """
    from core.utils.llm import LLMError

    try:
        llm.complete("只输出 JSON。", '返回 {"ok": true}', temperature=0.0)
    except LLMError as exc:
        print(f"\nLLM 通道不通，评测中止（没有生成任何候选）：\n  {exc}\n")
        print("常见原因：\n"
              "  - OpenRouter 免费模型需要在 Settings → Privacy 里允许 prompt 记录，"
              "否则返回 404 no endpoints\n"
              "  - 免费额度限速（429），等几分钟或换一个 :free 模型\n"
              "  - key 失效 / 已轮换（401）\n"
              "  - 模型名写错或已下线（400/404）\n")
        raise SystemExit(2)


def cmd_generate(args) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    situations = json.loads(FIXTURE.read_text(encoding="utf-8"))["situations"]
    if args.cassette:
        # 回放模式：不联网、不花钱、结果完全可复现。
        llm = CassetteProvider(Path(args.cassette), mode="replay", strict=True)
        print(f"回放 cassette：{args.cassette}（{len(llm)} 条记录）")
    else:
        llm = get_llm(CONFIG)
        _preflight(llm)
    retriever = build_retriever(load_cards(CONFIG.cards_dir), backend=args.backend)
    profile = _profile()

    records, blind = [], []
    for s in situations:
        risk = rule_scan(s["text"])
        if risk.level >= RiskLevel.CRISIS:
            # 危机情境不进生成评测：那条路径根本不生成文本。
            print(f"  {s['id']} 触发危机分支，跳过生成评测（这是正确行为）")
            records.append({"id": s["id"], "crisis": True, "signals": risk.signals})
            continue

        turns = parse_transcript(s["text"])
        hits = retriever.search(s["text"], k=CONFIG.top_k)

        for arm, arm_hits in (("grounded", hits), ("ungrounded", [])):
            opts = generate_options(llm, "", turns, arm_hits, profile, drop_ungrounded=False)
            for o in opts[: args.per_arm]:
                blind.append(BlindItem(s["id"], s["text"], o.text, o.why, arm, o.card_id))
            records.append({
                "id": s["id"], "arm": arm,
                "options": [{"text": o.text, "why": o.why, "card_id": o.card_id,
                             "grounded": o.grounded} for o in opts],
                "hits": [h.id for h in arm_hits],
            })
        print(f"  {s['id']} ✓")

    sheet, key = build_blind_sheet(blind)
    (OUT / "candidates.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "rating_sheet.md").write_text(sheet, encoding="utf-8")
    (OUT / "answer_key.json").write_text(dump_key(key), encoding="utf-8")

    if not blind:
        print("\n生成了 0 条候选：通道是通的，但每个情境的输出都没通过校验"
              "（缺 why / JSON 不合法 / 引用了不存在的 card_id）。"
              "\n看 data/generation_eval/candidates.json 里的 options 字段定位。")
        return 2

    print(f"\n候选已生成：{len(blind)} 条待评")
    print(f"  评分表   {OUT / 'rating_sheet.md'}   ← 现在去填这个")
    print(f"  答案表   {OUT / 'answer_key.json'}   ← 填完再看")
    print(f"\n主问题：{PRIMARY_QUESTION}")
    return 0


def cmd_score(args) -> int:
    sheet = (OUT / "rating_sheet.md").read_text(encoding="utf-8")
    key = json.loads((OUT / "answer_key.json").read_text(encoding="utf-8"))
    res = score_blind_sheet(sheet, key)

    n_sit = len(json.loads(FIXTURE.read_text(encoding="utf-8"))["situations"])
    print(f"人工盲评结果（{n_sit} 个情境，两组对照）\n")
    print(f"{'组别':<14}{'已评':>6}{'会发出去':>10}{'可发送率':>10}{'≥1条可用的情境数':>18}")
    print("-" * 60)
    for arm, label in (("grounded", "有知识库"), ("ungrounded", "无知识库(对照)")):
        d = res.get(arm, {})
        rate = d.get("would_send_rate")
        print(f"{label:<14}{d.get('rated', 0):>6}{d.get('would_send', 0):>10}"
              f"{(f'{rate:.1%}' if rate is not None else '—'):>10}"
              f"{d.get('situations_with_at_least_one_yes', 0):>18}")

    print("\n分维度均分（1-5）")
    for d in RUBRIC:
        g = res.get("grounded", {}).get("dimension_means", {}).get(d.key)
        u = res.get("ungrounded", {}).get("dimension_means", {}).get(d.key)
        if g is not None or u is not None:
            print(f"  {d.name:<14} 有知识库 {g if g is not None else '—':<6} "
                  f"无知识库 {u if u is not None else '—'}")

    (OUT / "human_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {OUT / 'human_results.json'}")
    return 0


def cmd_judge(args) -> int:
    records = json.loads((OUT / "candidates.json").read_text(encoding="utf-8"))
    llm = get_llm(CONFIG)
    sit_map = {s["id"]: s["text"]
               for s in json.loads(FIXTURE.read_text(encoding="utf-8"))["situations"]}

    from core.engine.generator import ReplyOption
    totals: dict[str, list[float]] = {"grounded": [], "ungrounded": []}
    for rec in records:
        if rec.get("crisis"):
            continue
        for o in rec["options"]:
            r = judge_option(llm, sit_map[rec["id"]],
                             ReplyOption(o["text"], o["why"], o["card_id"]))
            if r.ok:
                totals[rec["arm"]].append(r.mean)

    print("LLM-as-judge（仅作回归哨兵，不是验收标准）\n")
    for arm, label in (("grounded", "有知识库"), ("ungrounded", "无知识库")):
        vals = totals[arm]
        print(f"  {label:<10} n={len(vals):<4} 平均 {sum(vals)/len(vals):.2f}/5" if vals
              else f"  {label:<10} 无数据")
    print("\n⚠️ judge 存在自我偏好与位置偏差，且与人类判断的相关性在本场景下未验证。")
    print("   验收看 `score` 的人工盲评结果。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--backend", default=CONFIG.retrieval_backend)
    g.add_argument("--per-arm", type=int, default=2)
    g.add_argument("--cassette", default=None,
                   help="回放一份录好的 LLM 输出，不联网、可复现")
    g.set_defaults(func=cmd_generate)
    s = sub.add_parser("score"); s.set_defaults(func=cmd_score)
    j = sub.add_parser("judge"); j.set_defaults(func=cmd_judge)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
