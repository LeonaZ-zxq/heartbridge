"""反幻觉：回复里不许出现他没说过的事实。

━━━ 这组测试锁的是一个真实的失败案例 ━━━

用户输入的全是情绪，没有任何具体事实：
    「你要是跟正常人谈恋爱就好了 / 跟我在一起很累吧 我太敏感了 / 我什么都做不好」

而 SYSTEM_PROMPT 要求「必须具体」。模型被要求具体、输入里又没有具体的东西——
于是它编了：编出失眠（抄 soma_008 卡）、编出语音数呼吸（抄 comm_011 卡）、
编出一个知识库里根本不存在的「课堂糗事」。

这是一个 **Goodhart 案例**：一条防套话的规则，在输入抽象时会主动制造幻觉。
所以「不许编」也必须翻译成代码能执行的检查，而不是只写在 prompt 里。
"""
import pytest

from core.engine.generator import (
    _fact_tokens,
    _has_basis,
    _invented_facts,
    _validate,
    build_user_prompt,
    generate_options,
)

# 用户真实输入：三句纯情绪，零具体事实
SAID = "你要是跟正常人谈恋爱就好了 跟我在一起很累吧 我太敏感了 我什么都做不好"

# 模型真实产出的三条回复，全部不合格
BAD_SLEEP = (
    "宝宝，听到你说「我什么都做不好」，我感受到你很累。"
    "睡不着不是你的错，这也是抑郁常见的症状。"
    "我们可以先把每天上床时间固定下来。"
)
BAD_VOICE = (
    "宝宝，先深呼吸，我现在给你数：吸——一二三。"
    "再说说你房间里现在能看到的三样东西，我会一直在语音里陪着。"
)
BAD_CLASS = (
    "其实连正常人也会被工作压得喘不过气。"
    "咱们可以聊点轻松的，今晚一起分享一个好笑的课堂糗事吧。"
)


@pytest.mark.parametrize("text", [BAD_SLEEP, BAD_VOICE, BAD_CLASS])
def test_编造他没说过的场景会被抓出来(text):
    assert _invented_facts(text, SAID), f"没抓到编造：{text}"


def test_抓出来的是具体哪个词():
    """报告要能说清楚「编了什么」，不能只说「不合格」。

    上一版 UI 之所以让人困惑，就是因为它只说「都没通过校验」，
    不说是哪一条、为什么。
    """
    assert "症状" in _invented_facts(BAD_SLEEP, SAID)
    assert "语音" in _invented_facts(BAD_VOICE, SAID)
    assert "课堂" in _invented_facts(BAD_CLASS, SAID)


@pytest.mark.parametrize("text", [
    "你说你什么都做不好，可你今天还是把这句话说出口了。累不是敏感，我没觉得累。",
    "我不觉得累。你说你太敏感，可我要的就是这个。",
    "正常人这三个字是你加的，我没这么想过。",
])
def test_只用他说过的东西不会被误伤(text):
    assert _invented_facts(text, SAID) == ()


def test_同源词算有依据():
    """他说「今天面试又挂了」，回复里写「面试官」应该算有依据。

    整词比对会把它误判成编造，所以用 2-gram 重叠。
    """
    assert _has_basis("面试官", "今天面试又挂了")
    assert not _has_basis("课堂", "今天面试又挂了")


def test_常用名词不算编造事实():
    """「宝宝」「今天」这类词出现在任何回复里都不构成编造。"""
    assert _fact_tokens("宝宝，今天的事情我知道了") == []


def test_纯情绪输入下宁可朴素也不要编():
    """两条选项：一条编了失眠，一条朴素但真实。要留朴素的那条。"""
    raw = {"options": [
        {"text": BAD_SLEEP, "why": "共情", "card_id": "soma_008", "anchor": "我什么都做不好"},
        {"text": "我不觉得累。你说你太敏感，可我要的就是这个。",
         "why": "直接否认他的预设，而不是安慰他", "card_id": "soma_008",
         "anchor": "跟我在一起很累吧"},
    ]}
    opts = _validate(raw, {"soma_008"}, SAID)
    assert opts[0].invented          # 编了
    assert opts[1].invented == ()    # 没编


def test_全部编造时不给空结果只给标记():
    """兜底哲学跟 grounded/specific 一致：空结果比带警告的结果更糟。"""
    raw = {"options": [
        {"text": BAD_SLEEP, "why": "w", "card_id": "c", "anchor": "我什么都做不好"},
        {"text": BAD_CLASS, "why": "w", "card_id": "c", "anchor": "我什么都做不好"},
    ]}
    llm = _FakeLLM(raw)
    opts = generate_options(llm, "", [_FakeTurn(SAID)], [_FakeHit("c")])
    assert len(opts) == 2
    assert all(o.invented for o in opts)


def test_复读开头会被排到后面():
    """「听到你说X」不是硬性不合格，但有不复读的选项时优先给不复读的。"""
    raw = {"options": [
        {"text": "听到你说我什么都做不好，我想说累不是敏感。", "why": "w",
         "card_id": "c", "anchor": "我什么都做不好"},
        {"text": "累不是敏感，我没觉得累。", "why": "w",
         "card_id": "c", "anchor": "跟我在一起很累吧"},
    ]}
    llm = _FakeLLM(raw)
    opts = generate_options(llm, "", [_FakeTurn(SAID)], [_FakeHit("c")])
    assert opts[0].quotes_back is False   # 不复读的排前面
    assert opts[1].quotes_back is True


def test_纯情绪输入会在_prompt_里显式解除必须具体的要求():
    """「必须具体」和「不许编」会打架，要在输入抽象时明确告诉模型该让哪一条。

    不加这段的话，模型在两条冲突的硬性要求之间会选择编一个场景出来——
    这正是「课堂糗事」的来源。
    """
    from core.engine.generator import build_user_prompt

    prompt = build_user_prompt("", [_FakeTurn(SAID)], [_FakeHit("c")], None)
    assert "全是情绪" in prompt
    assert "不许提睡眠" in prompt


def test_泛指人群不算具体事实():
    """「正常人」是他用来贬低自己的抽象参照，不是他处境里的一个事实。

    算成事实的话，一段纯情绪的输入会被误判成「有具体的东西可接」，
    上面那条动态提醒就不会触发。
    """
    assert _fact_tokens("你要是跟正常人谈恋爱就好了") == []


def test_卡片示范被标成不许照抄():
    """卡片提供技巧，他的话提供事实——这个区分要写在喂给模型的文本里。"""
    prompt = build_user_prompt("", [_FakeTurn(SAID)], [_FakeHit("c")], None)
    assert "学结构不要抄内容" in prompt


def test_没有原话时不做编造判定():
    """拿不到他说的话时不能凭空判不合格。"""
    opts = _validate(
        {"options": [{"text": BAD_CLASS, "why": "w", "card_id": "c", "anchor": ""}]},
        {"c"}, "",
    )
    assert opts[0].invented == ()


# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #
class _FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, user, **kw):
        import json
        return json.dumps(self.payload, ensure_ascii=False)


class _FakeTurn:
    def __init__(self, text):
        self.text = text
        self.speaker = "他"

    def render(self):
        return f"他：{self.text}"


class _FakeCard:
    type = "communication"
    scenario = "他在自我否定"
    technique_name = "不反驳，改成提问"
    do = ["把「都」这个词拎出来问"]
    dont = ["直接夸他"]
    example_phrases = ["你说「什么都做不好」——这个「都」是从什么时候开始算的？"]
    why_it_works = "让他自己去够那个反例，比别人塞给他更站得住"

    def __init__(self, cid):
        self.id = cid


class _FakeHit:
    def __init__(self, cid):
        self.id = cid
        self.card = _FakeCard(cid)
        self.score = 1.0
