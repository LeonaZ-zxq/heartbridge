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


# --------------------------------------------------------------------------- #
# 危机卡：整个知识库里风险最高的内容
# --------------------------------------------------------------------------- #
def test_危机卡必须有权威来源(cards):
    """和躯体化卡同样的硬约束，理由更强：危机指导说错的代价不可逆。"""
    from core.knowledge.schema import CrisisCard

    crisis = [c for c in cards if isinstance(c, CrisisCard)]
    assert crisis, "知识库里应当有危机卡"
    for c in crisis:
        assert c.source.authority or c.source.url, f"{c.id} 没有权威来源"


def test_危机卡不接受博主来源():
    """用类型系统挡住「某博主说」——不是靠 prompt 祈祷模型遵守。"""
    import pytest
    from pydantic import ValidationError

    from core.knowledge.schema import CrisisCard

    with pytest.raises(ValidationError):
        CrisisCard(
            id="crisis_999", signal="测试", what_it_means="测试",
            do_now=["x"], say=["x"], avoid_saying=["x"], escalate_if=["x"],
            source={"platform": "douyin", "author": "某博主"},
        )


def test_每张危机卡都要说清楚什么时候升级(cards):
    """这是危机卡区别于沟通卡的关键字段。缺了它，这张卡是危险的。"""
    from core.knowledge.schema import CrisisCard

    for c in (c for c in cards if isinstance(c, CrisisCard)):
        assert c.escalate_if, f"{c.id} 没有写升级条件"
        assert c.do_now and c.say and c.avoid_saying


def test_危机卡不得列举具体方式(cards):
    """危机内容里不应出现具体的自伤方式。

    即使是在「移开危险物品」这种正当建议里也不列举：
    这类信息本身有风险，而陪伴者本来就知道自己家里有什么。
    """
    from core.knowledge.schema import CrisisCard

    banned = ["割腕", "上吊", "跳楼", "安眠药", "农药", "煤气", "一氧化碳"]
    for c in (c for c in cards if isinstance(c, CrisisCard)):
        blob = " ".join([c.signal, c.what_it_means, *c.do_now, *c.say,
                         *c.avoid_saying, *c.escalate_if])
        hit = [w for w in banned if w in blob]
        assert not hit, f"{c.id} 出现了具体方式：{hit}"
