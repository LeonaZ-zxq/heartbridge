"""安全层的两个漏洞：来世式表达漏报，以及二级校验 fail-open。

━━━ 触发这组测试的真实案例 ━━━

输入「下辈子会好的吧 我说不出话了」，危机检测没拦住，
这句话一路走到生成器，用户拿回来一条「你刚才说'下辈子会好的吧'，我记下来了」。

两个独立的缺陷叠在一起：
1. 规则词表覆盖的是「怎么用语言表达想死」，漏掉了「把希望寄到下一辈子」——
   被动自杀意念的标准表述，一个死字都不含。
2. LLM 二级校验失败时返回一个干净的 NONE，assess() 当成「安全」放行。
   一次静默的网络失败就等于整层安全检测被关掉，用户完全看不到。
"""
import pytest

from core.safety.detector import RiskLevel, assess, llm_scan
from core.utils.llm import LLMError


@pytest.mark.parametrize("text", [
    "他说下辈子会好的吧",
    "下辈子会好的吧 我说不出话了",
    "他说下辈子不想再做人了",
    "他说来生别再做人了",
    "他说这辈子就这样了吧",
    "他说他这辈子没救了",
])
def test_来世式表达算危机(text):
    """这些句子一个死字都不含，但说的是「这辈子没救了」。"""
    assert assess(text, llm=None).is_crisis, f"漏报：{text}"


@pytest.mark.parametrize("text", [
    "他开玩笑说下辈子想做只猫",
    "他说来世我们还做朋友",
    "他说下辈子想当医生",
    "下辈子再见啦",
])
def test_来世的日常用法不误报(text):
    """规则收窄到绝望语义，不是见到「下辈子」就报。

    误报的代价不小：每一次都会把生成挡掉、弹出危机响应，
    用户很快就会学会忽略它——那时真正的危机也拦不住了。
    """
    assert not assess(text, llm=None).is_crisis, f"误报：{text}"


class _DeadLLM:
    """一个永远调不通的 provider，模拟网络/配额故障。"""
    name = "dead"

    def complete(self, system, user, **kw):
        raise LLMError("模拟故障：403 Forbidden")


def test_二级校验失败不能当成安全():
    """安全检查失败时唯一诚实的输出是「我没查成」，不是「没问题」。"""
    r = llm_scan("他今天心情不太好", _DeadLLM())
    assert r.second_pass_ok is False
    assert r.level is RiskLevel.NONE   # 不能凭空捏造风险等级


def test_二级失败的事实会带到最终结论里():
    """上层要能如实告诉用户「这次只有规则层过了一遍」。"""
    r = assess("他今天心情不太好", llm=_DeadLLM())
    assert r.second_pass_ok is False
    assert "跳过二级" in r.rationale


def test_规则判危机时不受二级故障影响():
    """规则层已经判危机就直接返回，根本不问 LLM。"""
    r = assess("他说他想死", llm=_DeadLLM())
    assert r.is_crisis
    assert r.second_pass_ok is True


def test_二级正常时不该被标记():
    class _OKLLM:
        name = "ok"

        def complete(self, system, user, **kw):
            return '{"level":"none","reason":"没有风险信号"}'

    assert assess("他今天心情不太好", llm=_OKLLM()).second_pass_ok is True
