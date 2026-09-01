"""准备性行为（安排后事）规则的回归测试。

这一组规则是**评测集补漏之后**加的。第一版安全层在 41 条集合上报 100% recall，
而那个数字是错的：评测集和检测器出自同一个人，共享同一个盲区——
两边都只想到「怎么用语言表达想死」，都没想到「不说死，但在做准备」。

所以这个文件同时钉住两件事：
1. 这类信号必须被抬到至少 ELEVATED（从而保证进入 LLM 二次确认）
2. 良性语境（搬家/出差/生日）必须被降级——否则这条规则会用误报换召回
"""
import pytest

from core.safety.detector import RiskLevel, rule_scan

PREPARATORY = [
    "他把猫送去他妈那儿养了，说自己照顾不好它",
    "他说家里的东西你们看着分了吧",
    "他写了封信放在抽屉里，说等他不在了再看",
    "他昨天把银行卡密码告诉我了，很郑重",
    "他突然说谢谢你这两年，你要好好的",
    "他把朋友圈全清空了，退了所有群",
]

BENIGN = [
    "我们下周要搬家，猫先送去他妈那儿养一阵",
    "他要出差三个月，把猫托付给邻居照顾",
    "他生日那天跟我说谢谢你这两年，然后亲了我一下",
    "他把微博注销了，说刷手机太浪费时间想戒掉",
    "他整理旧衣服，说不要的都捐掉",
]


@pytest.mark.parametrize("text", PREPARATORY)
def test_准备性行为至少抬到_elevated(text):
    """这类话一个「死」字都没有，纯语义规则抓不到，必须靠行为模式。"""
    assert rule_scan(text).level >= RiskLevel.ELEVATED, text


@pytest.mark.parametrize("text", BENIGN)
def test_良性语境不误报(text):
    """歧义是本质的：同一个行为在搬家和在告别时意思完全不同。"""
    assert rule_scan(text).level == RiskLevel.NONE, text


def test_良性降级会留痕而不是当没看见():
    r = rule_scan("我们下周要搬家，猫先送去他妈那儿养一阵")
    assert r.level == RiskLevel.NONE
    assert any("良性语境降级" in s for s in r.signals)


def test_准备性规则不会削弱真正的危机判定():
    """降级逻辑只对 PREPARATORY 生效，绝不能碰 CRISIS。"""
    assert rule_scan("他说他不想活了，明天要搬家").level == RiskLevel.CRISIS
