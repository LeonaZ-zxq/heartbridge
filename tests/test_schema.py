"""知识库的数据契约测试。

这些测试守的不是「代码能跑」，而是「知识库内容合法」——
它们会在我手写或 LLM 蒸馏出一张烂卡时立刻失败。
"""
import pytest
from pydantic import ValidationError

from core.knowledge.schema import CommunicationCard, SomaticCard, parse_card

SOMATIC_TARGETS = {"胸闷", "过度换气", "发抖", "恶心", "疲惫", "疼痛", "麻木", "失眠"}


def test_all_cards_load_and_validate(cards):
    assert len(cards) >= 30


def test_ids_unique(cards):
    ids = [c.id for c in cards]
    assert len(ids) == len(set(ids))


def test_every_card_has_provenance(cards):
    """每张卡都要能回答「这条建议是哪来的」。"""
    for c in cards:
        assert any([c.source.authority, c.source.url, c.source.platform, c.source.author]), c.id


def test_somatic_cards_require_authority(cards):
    """医疗相邻内容的硬约束：必须有权威机构或 URL。"""
    for c in cards:
        if c.type == "somatic":
            assert c.source.authority or c.source.url, f"{c.id} 缺权威来源"


def test_somatic_cards_have_three_mandatory_blocks(cards):
    """在场 / 远程 / 何时就医——躯体化卡缺任何一块都不合格。"""
    for c in cards:
        if c.type == "somatic":
            assert c.in_person and c.remote and c.seek_help_if, c.id


def test_somatic_symptom_coverage(cards):
    """8 类目标症状必须全覆盖，防止悄悄漏掉一类。"""
    blob = " ".join(c.symptom + " ".join(c.aliases) for c in cards if c.type == "somatic")
    missing = [t for t in SOMATIC_TARGETS if t not in blob]
    assert not missing, f"未覆盖的症状类别: {missing}"


def test_somatic_card_without_source_is_rejected():
    """反向测试：没有权威来源的躯体化卡必须被拒绝，不能悄悄放行。"""
    with pytest.raises(ValidationError):
        SomaticCard(
            id="soma_999", source={"platform": "douyin"}, symptom="胸闷",
            what_it_is="x", in_person=["a"], remote=["b"], say=["c"],
            avoid_saying=["d"], seek_help_if=["e"],
        )


def test_bad_id_rejected():
    with pytest.raises(ValidationError):
        CommunicationCard(
            id="随便写的id", source={"platform": "web"}, scenario="s",
            technique_name="t", do=["a"], dont=["b"], example_phrases=["c"],
            why_it_works="w",
        )


def test_empty_list_rejected():
    """do/dont 不许为空——一张没有具体做法的卡对用户毫无价值。"""
    with pytest.raises(ValidationError):
        CommunicationCard(
            id="comm_999", source={"platform": "web"}, scenario="s",
            technique_name="t", do=[], dont=["b"], example_phrases=["c"], why_it_works="w",
        )


def test_unknown_type_rejected():
    with pytest.raises(ValueError):
        parse_card({"type": "nonsense"})
