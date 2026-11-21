"""ReAct Agent 的三条硬边界。

会写 ReAct 循环的人很多。这组测试想证明的是另一件事：
**这个 Agent 的安全保证不依赖模型的配合。**

最重要的一条是 test_危机输入下模型一次都不会被调用——
它断言的不是「模型选择了先查危机」，而是「模型根本没有得到选择的机会」。
前者靠 prompt，后者靠代码结构，只有后者能写成断言。
"""
import json

import pytest

from core.agent import AgentTrace, run_agent
from core.config import CONFIG
from core.knowledge.retrieval import BM25Retriever
from core.knowledge.schema import load_cards
from core.utils.llm import LLMError

NORMAL = "他说他什么都做不好，最近话很少"
CRISIS = "他说他想死"

# 一条结构合法的生成结果：text/why 齐全就能通过校验，
# 这组测试关心的是循环，不是生成质量（那是 test_grounding 的事）。
_OPTIONS = json.dumps({"options": [
    {"text": "你说你什么都做不好，可你今天还是把这句话说出口了。",
     "why": "接住他的原话，不替他下结论", "anchor": "什么都做不好",
     "card_id": "comm_001", "style": "共情"},
]}, ensure_ascii=False)


class _FakeLLM:
    """按 system prompt 分流的假模型，并记录每一次调用。

    三种 prompt 会打到同一个 complete()：安全二级校验、ReAct 步骤、回复生成。
    按内容分流比按调用顺序分流稳——顺序会随实现变，内容不会。
    """

    name = "fake"

    def __init__(self, steps, fail_with=None):
        self.steps = list(steps)
        self.fail_with = fail_with
        self.calls = []          # 记下每次调用的类型

    def complete(self, system, user, *, temperature=0.3):
        if "每一步只输出一个 JSON" in system:
            self.calls.append("react")
            if self.fail_with:
                raise self.fail_with
            return self.steps.pop(0) if self.steps else json.dumps({"action": "finish"})
        if "卡片提供技巧" in system:
            self.calls.append("generate")
            return _OPTIONS
        self.calls.append("safety")
        return json.dumps({"level": "none", "reason": "没有风险信号"})


def _step(action, **args):
    return json.dumps({"thought": "t", "action": action, "action_input": args},
                      ensure_ascii=False)


@pytest.fixture(scope="module")
def retriever():
    return BM25Retriever(load_cards(CONFIG.cards_dir))


# --------------------------------------------------------------------------- #
# 边界 1：危机检查前置，Agent 无权跳过
# --------------------------------------------------------------------------- #
def test_危机输入下模型一次都不会被调用(retriever):
    """这是整个模块最重要的一条断言。

    一个「想跳过安全检查」的 Agent——这里用一个上来就 finish 的脚本模拟——
    依然不可能跳过，因为危机判定发生在循环外面，在模型拿到第一个回合之前。
    模型不是「选择了遵守」，是**没有得到选择的机会**。
    """
    llm = _FakeLLM([_step("finish")])
    adv = run_agent(llm, CRISIS, retriever)

    assert llm.calls == []                  # 模型一次都没被调用
    assert adv.is_crisis
    assert adv.crisis_response
    assert adv.options == []                # 没有走生成
    assert adv.trace.stopped == "crisis"
    assert adv.trace.tool_calls == 0


def test_前置检查在任何路径上都执行过(retriever):
    llm = _FakeLLM([_step("search_knowledge", query="自我否定"), _step("finish")])
    adv = run_agent(llm, NORMAL, retriever)
    assert adv.trace.gate_ran is True


# --------------------------------------------------------------------------- #
# 边界 2：步数与工具预算
# --------------------------------------------------------------------------- #
def test_永不作答的模型会被步数上限掐停(retriever):
    """模型一直调工具不肯 finish，循环必须自己停，而不是转到天荒地老。"""
    # 查询必须真能召回到卡片，否则会先走「没收集到东西」的降级分支，
    # 测不到步数上限。第一版用的是「查询0/1/2」，BM25 一张都召不到。
    qs = ["他总说自己没用", "他不想说话", "他晚上睡不着", "他不肯出门"]
    llm = _FakeLLM([_step("search_knowledge", query=qs[i % len(qs)]) for i in range(50)])
    adv = run_agent(llm, NORMAL, retriever, max_steps=3)
    assert adv.trace.stopped == "max_steps"
    assert len(adv.trace.steps) == 3
    assert adv.options                      # 停下来之后仍然给出了回答


def test_单个工具的调用预算会被执行(retriever):
    """get_partner_profile 上限是 1。第二次调用必须被挡，并且告诉模型原因。"""
    llm = _FakeLLM([_step("get_partner_profile"), _step("get_partner_profile"),
                    _step("search_knowledge", query="自我否定"), _step("finish")])
    adv = run_agent(llm, NORMAL, retriever, max_steps=6)
    blocked = [s for s in adv.trace.steps if "调用次数已用完" in s.observation]
    assert len(blocked) == 1
    assert blocked[0].ok is False


def test_预算耗尽不是崩溃而是收敛到作答(retriever):
    llm = _FakeLLM([_step("get_partner_profile")] * 10)
    adv = run_agent(llm, NORMAL, retriever, max_steps=8)
    assert adv.trace.stopped in ("max_steps", "tool_budget", "degraded")
    assert adv.options or adv.is_crisis


# --------------------------------------------------------------------------- #
# 边界 3：失败降级，不是失败崩溃
# --------------------------------------------------------------------------- #
def test_工具抛异常且无任何检索结果时降级回退(retriever):
    class _Boom:
        name = "boom"

        def search(self, query, k=3, type_filter=None):
            raise RuntimeError("索引炸了")

    llm = _FakeLLM([_step("search_knowledge", query="自我否定"), _step("finish")])
    adv = run_agent(llm, NORMAL, _Boom())

    # 降级路径和 Agent 共用同一个 retriever，所以这里回退也会失败。
    # 要求是「不抛异常、并且说清楚两边都怎么坏的」，
    # 而不是硬造一个回答出来——检索器全坏时没有诚实的回答可给。
    assert adv.trace.stopped == "degraded"
    assert "索引炸了" in adv.trace.note
    assert "直通路径同样失败" in adv.trace.note
    assert adv.options == []
    assert not adv.is_crisis


def test_模型调用失败时降级而不是抛出(retriever):
    llm = _FakeLLM([], fail_with=LLMError("429 配额用完"))
    adv = run_agent(llm, NORMAL, retriever)
    assert adv.trace.stopped == "degraded"
    assert "429" in adv.trace.note
    assert adv.options


def test_不存在的工具名不会让循环崩掉(retriever):
    llm = _FakeLLM([_step("rm_minus_rf"), _step("search_knowledge", query="自我否定"),
                    _step("finish")])
    adv = run_agent(llm, NORMAL, retriever)
    bad = [s for s in adv.trace.steps if s.action == "rm_minus_rf"]
    assert bad and bad[0].ok is False
    assert "没有叫 rm_minus_rf 的工具" in bad[0].observation
    assert adv.options


def test_模型返回非对象时不会崩(retriever):
    llm = _FakeLLM(['["不是对象"]', _step("search_knowledge", query="自我否定"),
                    _step("finish")])
    adv = run_agent(llm, NORMAL, retriever)
    assert adv.options


# --------------------------------------------------------------------------- #
# Agent 相对直通管道的实际价值
# --------------------------------------------------------------------------- #
def test_多次检索的结果取并集且去重(retriever):
    """这才是 Agent 值得存在的理由：一次检索解决不了的复合问题可以拆开问。"""
    llm = _FakeLLM([_step("search_knowledge", query="他总说自己没用"),
                    _step("search_knowledge", query="他不想说话"),
                    _step("finish")])
    adv = run_agent(llm, NORMAL, retriever)
    ids = [h.id for h in adv.hits]
    assert len(ids) == len(set(ids))        # 去重
    assert len(ids) > CONFIG.top_k          # 确实比单次检索拿到更多


def test_返回类型与直通管道完全一致(retriever):
    """UI / CLI / 测试都不需要知道这次走的是 Agent。"""
    from core.engine.pipeline import Advice

    llm = _FakeLLM([_step("search_knowledge", query="自我否定"), _step("finish")])
    adv = run_agent(llm, NORMAL, retriever)
    assert isinstance(adv, Advice)
    assert isinstance(adv.trace, AgentTrace)


# --------------------------------------------------------------------------- #
# 回归：Agent 不得把危机卡带进生成
# --------------------------------------------------------------------------- #
def test_agent_检索永远不会带出危机卡(retriever):
    """这条是这个 Agent 实打实引入过的一次安全回归的回归测试。

    第一版的 search_knowledge 直接调 retriever.search()，绕过了
    pipeline._retrieve_for_generation 的危机卡过滤。于是像
    「他说他撑不下去了」这种判为 ELEVATED（不是 CRISIS）、走正常路径的输入，
    可以把 crisis_001 检索进生成 prompt——而整个安全架构的立足点是
    「危机内容确定性交付、永不生成」。模型一旦拿到危机卡，就能改写
    「什么时候必须立刻叫救护车」这类升级条件。

    pipeline 里早就修过这个泄漏。Agent 从底层 API 绕过去，就把它放了回来。
    **安全过滤必须长在所有调用方都绕不开的那一层**——这条测试就是那个保证。
    """
    llm = _FakeLLM([_step("search_knowledge", query="他说他撑不下去了"),
                    _step("search_knowledge", query="他不想活了怎么办"),
                    _step("finish")])
    adv = run_agent(llm, "他最近话很少，说自己撑不下去了", retriever)

    if adv.is_crisis:
        pytest.skip("这条输入被判为危机，走的是短路路径，不经过检索")
    assert adv.hits, "应该检索到了东西，否则这条测试没有验证到过滤"
    leaked = [h.id for h in adv.hits if h.card.type == "crisis"]
    assert not leaked, f"危机卡泄漏进生成候选：{leaked}"
