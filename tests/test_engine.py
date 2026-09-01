"""回复引擎测试。

这里体现的是「怎么测一个会调 LLM 的模块」：
把模型换成 MockProvider，**断言我们对模型输出做的处理**，
而不是断言模型说了什么。所有测试都是确定性的。
"""
import json

import pytest

from core.engine.generator import build_user_prompt, generate_options
from core.engine.pipeline import advise
from core.knowledge.retrieval import BM25Retriever
from core.profile.models import PartnerProfile
from core.utils.llm import MockProvider

MARKER = "硬性要求"  # SYSTEM_PROMPT 里的一段，用来路由 mock


def mock_llm(payload):
    llm = MockProvider()
    llm.register(MARKER, lambda s, u: json.dumps(payload, ensure_ascii=False))
    return llm


@pytest.fixture
def retriever(cards):
    return BM25Retriever(cards)


# --------------------------------------------------------------------------- #
# 反幻觉：引用校验
# --------------------------------------------------------------------------- #
def test_ungrounded_citations_are_dropped(cards, retriever):
    """模型引用了没检索到的卡 = 在编。必须丢掉。"""
    hits = retriever.search("他说自己不配被爱", k=3)
    llm = mock_llm({"options": [
        {"text": "真的", "why": "有依据", "card_id": hits[0].id},
        {"text": "编的", "why": "无依据", "card_id": "comm_888"},
    ]})
    opts = generate_options(llm, "", [], hits)
    assert [o.card_id for o in opts] == [hits[0].id]


def test_all_ungrounded_keeps_them_but_flags(cards, retriever):
    """全部未接地时保留并标记——给用户空结果比给带警告的结果更糟。"""
    hits = retriever.search("他说自己不配被爱", k=3)
    llm = mock_llm({"options": [{"text": "x", "why": "y", "card_id": "comm_888"}]})
    opts = generate_options(llm, "", [], hits)
    assert len(opts) == 1 and opts[0].grounded is False


def test_option_without_why_is_rejected(cards, retriever):
    """没有「为什么」的选项退化成话术抄写，违背产品设计，直接丢。"""
    hits = retriever.search("他说自己不配被爱", k=3)
    llm = mock_llm({"options": [
        {"text": "只有话术", "why": "", "card_id": hits[0].id},
        {"text": "有解释", "why": "机制说明", "card_id": hits[0].id},
    ]})
    assert [o.text for o in generate_options(llm, "", [], hits)] == ["有解释"]


def test_at_most_three_options(cards, retriever):
    hits = retriever.search("他说自己不配被爱", k=3)
    llm = mock_llm({"options": [
        {"text": f"o{i}", "why": "w", "card_id": hits[0].id} for i in range(9)
    ]})
    assert len(generate_options(llm, "", [], hits)) == 3


def test_llm_failure_returns_empty_not_crash(cards, retriever):
    class Failing:
        name = "failing"

        def complete(self, system, user, *, temperature=0.3):
            from core.utils.llm import LLMError
            raise LLMError("down")

    hits = retriever.search("他说自己不配被爱", k=3)
    assert generate_options(Failing(), "", [], hits) == []


# --------------------------------------------------------------------------- #
# Prompt 构造
# --------------------------------------------------------------------------- #
def test_prompt_only_includes_non_empty_profile_fields(cards, retriever):
    hits = retriever.search("他说自己不配被爱", k=3)
    p = PartnerProfile(nickname="小鱼", landmines=["他前任"], long_distance=True)
    prompt = build_user_prompt("", [], hits, p)
    assert "他前任" in prompt and "异地" in prompt
    assert "诊断" not in prompt  # 空字段不该占 token、稀释注意力


def test_prompt_carries_voice_samples(cards, retriever):
    hits = retriever.search("他说自己不配被爱", k=3)
    p = PartnerProfile(my_voice_samples=["我在呢", "别怕"])
    assert "我在呢" in build_user_prompt("", [], hits, p)


def test_prompt_lists_only_retrieved_card_ids(cards, retriever):
    hits = retriever.search("他说自己不配被爱", k=3)
    prompt = build_user_prompt("", [], hits, None)
    for h in hits:
        assert h.id in prompt


# --------------------------------------------------------------------------- #
# 端到端：安全分支优先
# --------------------------------------------------------------------------- #
def test_crisis_short_circuits_before_retrieval_and_generation(retriever):
    """危机命中时，检索和生成都不该发生——用户看到的必须是审查过的模板。"""
    llm = mock_llm({"options": [{"text": "不该出现", "why": "w", "card_id": "comm_001"}]})
    adv = advise("小鱼 23:00\n我不想活了", retriever, llm)
    assert adv.is_crisis
    assert adv.hits == [] and adv.options == []
    assert "13 11 14" in adv.render()


def test_normal_path_produces_options_with_reasons(retriever):
    llm = mock_llm({"options": [
        {"text": "我在", "why": "先验证情绪", "card_id": "comm_001", "style": "共情"}
    ]})
    adv = advise("小鱼 23:00\n我觉得自己不配被爱", retriever, llm)
    assert not adv.is_crisis
    out = adv.render()
    assert "为什么" in out and "先验证情绪" in out
    assert "不做诊断" in out  # disclaimer 必须每次都在


def test_somatic_query_routes_to_somatic_cards(retriever):
    llm = mock_llm({"options": []})
    adv = advise("小鱼 23:00\n我喘不上气，手好麻", retriever, llm)
    assert adv.hits and all(h.card.type == "somatic" for h in adv.hits)


def test_elevated_risk_is_surfaced_to_user(retriever):
    llm = mock_llm({"options": [{"text": "x", "why": "y", "card_id": "comm_001"}]})
    adv = advise("小鱼 23:00\n他说他撑不下去了", retriever, llm)
    assert not adv.is_crisis
    assert "留意" in adv.render()
