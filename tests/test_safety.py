"""安全层测试。

这个文件里最重要的不是覆盖率，而是几条**不变量（invariant）**：
系统无论怎么改，这几条都不能被破坏。把它们写成测试，
等于把安全设计固化进代码，而不是留在某个人的记忆里。
"""
import json
from pathlib import Path

import pytest

from core.config import Config
from core.safety.detector import RiskLevel, assess, rule_scan
from core.safety.templates import RESOURCES_AU, render_crisis_response
from core.utils.llm import MockProvider

FIXTURE = Path(__file__).parent / "fixtures/safety_eval.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
CRISIS = [c for c in CASES if c["expect"] == "crisis"]
NORMAL = [c for c in CASES if c["expect"] == "none"]


# --------------------------------------------------------------------------- #
# 不变量 1：危机召回零容忍
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", CRISIS, ids=lambda c: c["text"][:16])
def test_every_crisis_case_is_caught(case):
    """逐条参数化，而不是算一个总召回率。

    这样任何一条漏报都会单独报一个红，测试报告里能直接看到是哪一句——
    在安全模块里，'93% 通过' 是没有意义的，必须知道漏的是哪一条。
    """
    assert rule_scan(case["text"]).level == RiskLevel.CRISIS


# --------------------------------------------------------------------------- #
# 不变量 2：误报率受控（中文「死」的程度副词用法）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", NORMAL, ids=lambda c: c["text"][:16])
def test_normal_messages_do_not_trigger_crisis(case):
    """告警疲劳本身是一种安全失效：用户学会忽略警告之后，真警告也会被忽略。"""
    assert rule_scan(case["text"]).level < RiskLevel.CRISIS


def test_degree_adverb_death_is_masked():
    for text in ["累死了", "笑死我了", "困死了", "热死了", "电脑死机了"]:
        assert rule_scan(text).level == RiskLevel.NONE, text


# --------------------------------------------------------------------------- #
# 不变量 3：否定式降级，但绝不清零
# --------------------------------------------------------------------------- #
def test_negation_downgrades_but_never_to_zero():
    r = rule_scan("他说他不会自杀的")
    assert r.level == RiskLevel.ELEVATED, "否定不该把风险清零"
    assert r.level < RiskLevel.CRISIS, "否定应当降一级"


# --------------------------------------------------------------------------- #
# 不变量 4（最关键）：LLM 只能升级，永远不能降级
# --------------------------------------------------------------------------- #
def test_llm_cannot_downgrade_a_rule_crisis():
    """一个非确定性组件不允许关掉一个已经亮起的安全信号。

    这条测试防的是一整类真实事故：模型幻觉、prompt 注入、
    或者未来某个人「顺手」把 LLM 结论当成最终结论。
    """
    llm = MockProvider()
    llm.register("危机信号识别器", lambda s, u: json.dumps({"level": "none", "reason": "我觉得没事"}))
    result = assess("他说他不想活了", llm=llm)
    assert result.level == RiskLevel.CRISIS


def test_llm_can_escalate_what_rules_miss():
    """反向：规则没抓到的隐晦表达，LLM 应该能把它抬上来。"""
    llm = MockProvider()
    llm.register("危机信号识别器",
                 lambda s, u: json.dumps({"level": "crisis", "reason": "在交代身后事"}))
    subtle = "他今天把猫送去我妈那儿了，说以后麻烦她"
    assert rule_scan(subtle).level < RiskLevel.CRISIS      # 规则确实抓不到
    assert assess(subtle, llm=llm).level == RiskLevel.CRISIS


def test_rule_crisis_skips_llm_call():
    """规则已判危机时不该再调 LLM：省延迟，且 LLM 本来也无权改变结论。"""
    llm = MockProvider()
    assess("他说他想自杀", llm=llm)
    assert llm.calls == []


def test_llm_failure_does_not_break_safety_layer():
    """LLM 挂掉时安全层必须降级为规则层，而不是抛异常让整条链路失败。"""
    class Broken:
        name = "broken"

        def complete(self, system, user, *, temperature=0.3):
            raise RuntimeError("网络炸了")

    from core.utils.llm import LLMError, LLMProvider  # noqa: F401

    class Failing:
        name = "failing"

        def complete(self, system, user, *, temperature=0.3):
            from core.utils.llm import LLMError
            raise LLMError("429 限速")

    assert assess("他今天心情不好", llm=Failing()).level == RiskLevel.NONE
    assert assess("他说他不想活了", llm=Failing()).level == RiskLevel.CRISIS


def test_second_pass_can_be_disabled_by_config():
    llm = MockProvider()
    llm.register("危机信号识别器", lambda s, u: json.dumps({"level": "crisis", "reason": "x"}))
    cfg = Config()
    object.__setattr__(cfg, "crisis_llm_second_pass", False)
    assert assess("他今天有点累", llm=llm, cfg=cfg).level == RiskLevel.NONE


# --------------------------------------------------------------------------- #
# 不变量 5：危机模板的内容必须完整且可预测
# --------------------------------------------------------------------------- #
def test_crisis_template_always_contains_emergency_resources():
    out = render_crisis_response("测试")
    for res in RESOURCES_AU:
        assert res.contact in out, f"危机回复必须包含 {res.name}"
    assert "000" in out and "13 11 14" in out
    assert "12356" in out  # 恋人在国内时的通道


def test_crisis_template_is_deterministic():
    """同样输入永远同样输出——这是硬编码模板相对生成的核心价值。"""
    assert render_crisis_response("x") == render_crisis_response("x")


def test_crisis_template_carries_disclaimer_and_escalation():
    out = render_crisis_response()
    assert "不能替代专业干预" in out
    assert "不做诊断" in out
