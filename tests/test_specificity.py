"""anchor 校验：把「回复要具体」变成一条代码能执行的检查。

写在 prompt 里的「不要说套话」是没有约束力的——模型会答应，然后继续
生成「我在，有什么都跟我说」。因为「具体」对模型不是可验证的目标。

所以换成一个可验证的代理目标：你必须指出这条回复在回应他说的哪一句原话，
而那句话我会去他的消息里查。查不到，这条不算数。
"""
import pytest

from core.engine.generator import _check_anchor, _validate

SAID = "今天面试又挂了\n我大概真的什么都做不好"


@pytest.mark.parametrize("anchor", ["面试又挂了", "今天面试又挂了，", "什么都做不好"])
def test_原话摘录通过(anchor):
    assert _check_anchor(anchor, SAID)


@pytest.mark.parametrize("anchor", ["", "我", "你的工作压力", "他妈妈打电话"])
def test_编造或过短的_anchor_不通过(anchor):
    assert not _check_anchor(anchor, SAID)


def test_标点差异不影响判定():
    """模型摘原文时经常带上或漏掉标点，比的应该是内容不是排版。"""
    assert _check_anchor("今天面试，又挂了！", SAID)


def test_通用套话会被标成不具体():
    raw = {"options": [
        {"text": "我在呢，有什么都跟我说", "why": "表达陪伴", "card_id": "comm_001", "anchor": ""},
        {"text": "面试官只见了你三十分钟，我见了你两年半", "why": "用具体立场代替笼统夸奖",
         "card_id": "comm_001", "anchor": "面试又挂了"},
    ]}
    opts = _validate(raw, {"comm_001"}, SAID)
    assert opts[0].specific is False   # 「我在」放到任何情境都成立 → 锚不住
    assert opts[1].specific is True


def test_没有原文时不做具体性判定():
    """拿不到他说的话（比如只填了情境描述）时不能凭空判不合格。"""
    opts = _validate({"options": [{"text": "t", "why": "w", "card_id": "c", "anchor": ""}]},
                     {"c"}, "")
    assert opts[0].specific is True
