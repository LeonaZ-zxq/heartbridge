#!/usr/bin/env python3
"""危机检测评测。

输出三个数字，对应三种不同代价的错误：
  · crisis 召回率   —— 漏掉一个真危机。**零容忍，必须 100%**
  · 误报率 (FPR)    —— 把普通消息判成危机。代价是告警疲劳，要控制但可接受
  · 降级错误        —— 把危机判成 elevated。等同于漏报，同样零容忍

面试点：这三个指标不是「准确率」一个数能概括的，因为三种错误的代价
差了几个数量级。**指标设计必须反映真实代价结构**——只报 accuracy
在这种场景里是失职的。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.safety.detector import RiskLevel, rule_scan  # noqa: E402

LEVELS = {"none": RiskLevel.NONE, "elevated": RiskLevel.ELEVATED, "crisis": RiskLevel.CRISIS}


def main() -> int:
    path = Path(__file__).parent.parent / "tests/fixtures/safety_eval.json"
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]

    confusion: Counter = Counter()
    missed, false_alarms, downgrades = [], [], []

    for c in cases:
        got = rule_scan(c["text"]).level
        want = LEVELS[c["expect"]]
        confusion[(c["expect"], got.name.lower())] += 1
        if want == RiskLevel.CRISIS and got < RiskLevel.CRISIS:
            (missed if got == RiskLevel.NONE else downgrades).append((c, got))
        if want == RiskLevel.NONE and got >= RiskLevel.CRISIS:
            false_alarms.append((c, got))

    n_crisis = sum(1 for c in cases if c["expect"] == "crisis")
    n_none = sum(1 for c in cases if c["expect"] == "none")
    caught = n_crisis - len(missed) - len(downgrades)

    print(f"评测集: {len(cases)} 条  (crisis {n_crisis} / elevated "
          f"{sum(1 for c in cases if c['expect']=='elevated')} / none {n_none})\n")
    print(f"  crisis 召回率        {caught}/{n_crisis} = {caught/n_crisis:.1%}   ← 必须 100%")
    print(f"  误报率 (none→crisis)  {len(false_alarms)}/{n_none} = {len(false_alarms)/n_none:.1%}")
    print(f"  降级错误 (crisis→低)  {len(downgrades) + len(missed)}\n")

    for label, group in (("漏报", missed), ("降级", downgrades), ("误报", false_alarms)):
        for c, got in group:
            print(f"  ✗ [{label}] {c['text']}  期望 {c['expect']} 实际 {got.name.lower()}")

    return 0 if not (missed or downgrades) else 1


if __name__ == "__main__":
    raise SystemExit(main())
