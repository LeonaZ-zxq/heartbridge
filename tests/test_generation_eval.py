"""生成质量评测装置的测试。

评测工具本身出 bug，比被评测的系统出 bug 更危险——
它会让你对一个坏系统充满信心。所以盲评表的「盲」是要被测试的。
"""
import json

from core.engine.generator import ReplyOption
from core.evaluation.rubric import (
    PRIMARY_QUESTION, RUBRIC, BlindItem, build_blind_sheet, judge_option,
    score_blind_sheet,
)
from core.safety.detector import RiskLevel, rule_scan
from core.utils.llm import MockProvider

SITS = json.loads(
    (__import__("pathlib").Path(__file__).parent / "fixtures/generation_eval.json")
    .read_text(encoding="utf-8")
)["situations"]


def sample_items():
    return [
        BlindItem("g01", "我今天面试挂了", "有知识库的回复", "机制解释A", "grounded", "comm_016"),
        BlindItem("g01", "我今天面试挂了", "没知识库的回复", "机制解释B", "ungrounded"),
        BlindItem("g02", "我睡不着", "有知识库的回复2", "机制C", "grounded", "soma_008"),
        BlindItem("g02", "我睡不着", "没知识库的回复2", "机制D", "ungrounded"),
    ]


# --------------------------------------------------------------------------- #
# 盲评表：「盲」必须是真的
# --------------------------------------------------------------------------- #
def test_sheet_hides_the_arm_and_card_id():
    sheet, _ = build_blind_sheet(sample_items())
    for token in ("grounded", "ungrounded", "comm_016", "soma_008"):
        assert token not in sheet, f"盲评表泄漏了 {token}"


def test_sheet_contains_every_option_and_the_primary_question():
    sheet, key = build_blind_sheet(sample_items())
    for it in sample_items():
        assert it.option_text in sheet
    assert PRIMARY_QUESTION in sheet
    assert len(key) == 4


def test_answer_key_maps_labels_back_to_arms():
    _, key = build_blind_sheet(sample_items())
    assert set(key.values()) == {"grounded", "ungrounded"}
    assert all(lbl.startswith(("g01-", "g02-")) for lbl in key)


def test_shuffle_is_seeded_and_reproducible():
    a, ka = build_blind_sheet(sample_items(), seed=7)
    b, kb = build_blind_sheet(sample_items(), seed=7)
    assert a == b and ka == kb


def test_different_seeds_can_reorder():
    _, k1 = build_blind_sheet(sample_items(), seed=1)
    _, k2 = build_blind_sheet(sample_items(), seed=999)
    assert set(k1) == set(k2)  # 标签集合一样
    # 至少存在一个种子组合让顺序不同（否则打乱是假的）
    assert any(
        build_blind_sheet(sample_items(), seed=s)[1] != k1 for s in range(2, 40)
    )


# --------------------------------------------------------------------------- #
# 评分解析
# --------------------------------------------------------------------------- #
def test_scoring_separates_the_two_arms():
    sheet, key = build_blind_sheet(sample_items())
    filled = sheet
    for label, arm in key.items():
        verdict = "Y" if arm == "grounded" else "N"
        filled = filled.replace(
            f"### {label}\n", f"### {label}\n<<{verdict}>>\n", 1
        )
    # 把占位符换成真正的答案行
    lines = []
    current = None
    for line in filled.splitlines():
        if line.startswith("<<"):
            current = line.strip("<>")
            continue
        if line.startswith("我会发吗:") and current:
            line = f"我会发吗: {current}"
            current = None
        lines.append(line)
    res = score_blind_sheet("\n".join(lines), key)

    assert res["grounded"]["would_send"] == 2
    assert res["grounded"]["would_send_rate"] == 1.0
    assert res["ungrounded"]["would_send"] == 0
    assert res["ungrounded"]["would_send_rate"] == 0.0


def test_scoring_counts_situations_with_at_least_one_usable_option():
    """主验收标准是「有多少情境里至少有一条我真会发」，不是逐条通过率。"""
    items = sample_items()
    sheet, key = build_blind_sheet(items)
    g_labels = [l for l, a in key.items() if a == "grounded"]
    filled = sheet
    for label in g_labels:
        head = f"### {label}"
        idx = filled.index(head)
        filled = filled[:idx] + filled[idx:].replace("我会发吗: ", "我会发吗: Y", 1)
    res = score_blind_sheet(filled, key)
    assert res["grounded"]["situations_with_at_least_one_yes"] == 2


def test_unfilled_sheet_yields_no_score_not_a_crash():
    sheet, key = build_blind_sheet(sample_items())
    res = score_blind_sheet(sheet, key)
    assert res["grounded"]["rated"] == 0
    assert res["grounded"]["would_send_rate"] is None


def test_dimension_scores_are_parsed():
    sheet, key = build_blind_sheet(sample_items())
    label = next(iter(key))
    idx = sheet.index(f"### {label}")
    filled = sheet[:idx] + sheet[idx:].replace("我会发吗: ", "我会发吗: Y", 1).replace(
        "validation: ", "validation: 5", 1)
    res = score_blind_sheet(filled, key)
    arm = key[label]
    assert res[arm]["dimension_means"]["validation"] == 5.0


# --------------------------------------------------------------------------- #
# LLM-as-judge
# --------------------------------------------------------------------------- #
def judge_llm(payload):
    llm = MockProvider()
    llm.register("评估一条", lambda s, u: json.dumps(payload, ensure_ascii=False))
    return llm


def test_judge_returns_all_rubric_dimensions():
    llm = judge_llm({"scores": {d.key: 4 for d in RUBRIC}, "note": "还行"})
    r = judge_option(llm, "情境", ReplyOption("回复", "为什么", "comm_001"))
    assert set(r.scores) == {d.key for d in RUBRIC}
    assert r.mean == 4.0


def test_judge_clamps_out_of_range_scores():
    llm = judge_llm({"scores": {d.key: 99 for d in RUBRIC}})
    assert all(v == 5 for v in judge_option(llm, "s", ReplyOption("t", "w", "c")).scores.values())


def test_judge_defaults_missing_dimensions_to_neutral():
    llm = judge_llm({"scores": {"validation": 5}})
    r = judge_option(llm, "s", ReplyOption("t", "w", "c"))
    assert r.scores["validation"] == 5
    assert r.scores["sendability"] == 3


def test_judge_failure_is_marked_not_silently_zero():
    class Failing:
        name = "f"

        def complete(self, system, user, *, temperature=0.3):
            from core.utils.llm import LLMError
            raise LLMError("down")

    r = judge_option(Failing(), "s", ReplyOption("t", "w", "c"))
    assert r.ok is False and r.scores == {}


# --------------------------------------------------------------------------- #
# 评测集本身
# --------------------------------------------------------------------------- #
def test_fixture_has_enough_situations():
    assert len(SITS) >= 20
    assert len({s["id"] for s in SITS}) == len(SITS)


def test_no_situation_overlaps_the_retrieval_eval_sets():
    """生成评测集必须与检索评测集无重叠，否则又是一次泄漏。"""
    import pathlib

    fx = pathlib.Path(__file__).parent / "fixtures"
    other = set()
    for name in ("retrieval_eval.json", "retrieval_holdout.json"):
        for q in json.loads((fx / name).read_text(encoding="utf-8"))["queries"]:
            other.add(q["q"])
    for s in SITS:
        assert s["text"] not in other


def test_crisis_situations_are_excluded_from_generation_scoring():
    """危机情境根本不走生成路径，把它算进生成质量里是没有意义的。"""
    crisis = [s for s in SITS if rule_scan(s["text"]).level >= RiskLevel.CRISIS]
    for s in crisis:
        assert s["id"] in {sit["id"] for sit in SITS}
    # 至少留一个边界样本，确保这条分支在评测里真的被走到
    assert any("边界" in " ".join(s.get("tags", [])) for s in SITS)
