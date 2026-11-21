"""评测集本身的完整性检查。

一个评测集如果 expect 指向不存在的卡、或者指向危机卡（生成路径永远拿不到它），
那么它报出来的召回率就是错的——而且**错得很安静**：数字照常产出，
只是永远差那么几个点，没人知道为什么。

这组测试把三份集合都当成需要校验的输入来对待，
和 generator 把 LLM 输出当不可信输入是同一条原则。
"""
import json
from pathlib import Path

import pytest

from core.config import CONFIG
from core.knowledge.schema import load_cards

FIXTURES = Path(__file__).parent / "fixtures"
SETS = ["retrieval_eval.json", "retrieval_holdout.json", "retrieval_test.json"]


@pytest.fixture(scope="module")
def card_types():
    return {c.id: c.type for c in load_cards(CONFIG.cards_dir)}


@pytest.mark.parametrize("name", SETS)
def test_每条_expect_都指向真实存在的卡片(name, card_types):
    qs = json.loads((FIXTURES / name).read_text(encoding="utf-8"))["queries"]
    missing = sorted({q["expect"] for q in qs} - set(card_types))
    assert not missing, f"{name} 指向了不存在的卡片：{missing}"


@pytest.mark.parametrize("name", SETS)
def test_没有一条_expect_指向危机卡(name, card_types):
    """危机卡被 _retrieve_for_generation 挡在生成路径之外。

    标注成危机卡的查询等于一个**永远不可能答对**的题，
    它只会无声地压低所有后端的分数。
    """
    qs = json.loads((FIXTURES / name).read_text(encoding="utf-8"))["queries"]
    bad = [q["q"] for q in qs if card_types.get(q["expect"]) == "crisis"]
    assert not bad, f"{name} 把危机卡标成了期望答案：{bad}"


@pytest.mark.parametrize("name", SETS)
def test_查询不重复(name):
    qs = json.loads((FIXTURES / name).read_text(encoding="utf-8"))["queries"]
    texts = [q["q"] for q in qs]
    dupes = {t for t in texts if texts.count(t) > 1}
    assert not dupes, f"{name} 有重复查询：{dupes}"


@pytest.mark.parametrize("name", SETS)
def test_每份集合都带协议说明(name):
    """_note 不是装饰。三份集合的区别全在使用纪律上，

    而纪律写在代码之外就会丢——dev 集当年就是这么被污染的。
    """
    payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert payload.get("_note", "").strip(), f"{name} 缺少 _note"


def test_test_集的规模够报一个数():
    """少于 30 条时，一条对错就能把召回率动 3 个点以上。"""
    qs = json.loads((FIXTURES / "retrieval_test.json").read_text(encoding="utf-8"))["queries"]
    assert len(qs) >= 30


def test_test_集没有和另外两份重叠():
    """重叠一条，这份集合就被污染一条。"""
    def texts(n):
        return {q["q"] for q in json.loads(
            (FIXTURES / n).read_text(encoding="utf-8"))["queries"]}

    test = texts("retrieval_test.json")
    assert not (test & texts("retrieval_eval.json"))
    assert not (test & texts("retrieval_holdout.json"))
