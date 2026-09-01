"""摄取管道测试。

重点测的是**管道对模型输出的约束**，不是模型的输出质量：
来源能不能被篡改、id 谁分配、坏卡会不会漏进去、重复卡会不会被静默吞掉。
"""
import json

import pytest

from core.ingestion.distiller import chunk_transcript, distill
from core.ingestion.downloader import IngestionError, SourceMeta, local_audio
from core.knowledge.retrieval import BM25Retriever
from core.utils.llm import MockProvider

MARKER = "你是整理者"
GOOD = {
    "scenario": "他说自己不配被爱，全盘否定自己",
    "technique_name": "先接住情绪",
    "do": ["共情确认"], "dont": ["立刻反驳"], "example_phrases": ["我听到了"],
    "why_it_works": "降低防御", "tags": ["自我否定"],
    "user_phrasings": ["说自己没用"],
}
FRESH = {**GOOD, "scenario": "他开始沉迷赌博借了很多钱还不敢告诉家里",
         "tags": ["赌博"], "user_phrasings": ["他欠债了"]}
SOURCE = {"platform": "douyin", "author": "某博主", "url": "https://v.douyin.com/x"}


def llm_returning(payload):
    llm = MockProvider()
    llm.register(MARKER, lambda s, u: json.dumps(payload, ensure_ascii=False))
    return llm


# --------------------------------------------------------------------------- #
# 分块
# --------------------------------------------------------------------------- #
def test_short_transcript_is_one_chunk():
    assert len(chunk_transcript("很短的一段话。")) == 1


def test_empty_transcript_yields_no_chunks():
    assert chunk_transcript("   ") == []


def test_long_transcript_chunks_with_overlap():
    text = "。".join(f"这是第{i}句话，讲一个关于陪伴的技巧要点" for i in range(400))
    chunks = chunk_transcript(text)
    assert len(chunks) > 1
    assert all(len(c) <= 2500 for c in chunks)
    # 相邻块必须有重叠，否则跨块的技巧两边都拿不全
    assert chunks[0][-60:] in chunks[1]


# --------------------------------------------------------------------------- #
# 来源与 id 由代码掌控（最重要的一组）
# --------------------------------------------------------------------------- #
def test_model_cannot_forge_the_source():
    """模型返回 source 也没用，一律被下载阶段的元数据覆盖。"""
    poisoned = {**FRESH, "source": {"authority": "世界卫生组织"}, "id": "comm_999"}
    report = distill("稿子", llm_returning([poisoned]), SOURCE, start_index=50)
    card = report.cards[0]
    assert card.source.platform == "douyin"
    assert card.source.author == "某博主"
    assert card.source.authority is None, "模型伪造的权威机构必须被丢弃"


def test_ids_are_assigned_by_code_sequentially():
    two = [FRESH, {**FRESH, "scenario": "他妈妈住院了他要请假回国照顾"}]
    report = distill("稿子", llm_returning(two), SOURCE, start_index=50)
    assert [c.id for c in report.cards] == ["comm_050", "comm_051"]


def test_model_cannot_change_card_type():
    report = distill("稿子", llm_returning([{**FRESH, "type": "somatic"}]), SOURCE)
    assert report.cards[0].type == "communication"


def test_ingested_cards_default_to_lived_experience():
    """摄取进来的是博主口播，不是临床指南。

    Card 的默认层级是 clinical_guideline（知识库里手写的卡大多如此），
    所以这条**必须由摄取管道显式盖章**，否则一批短视频经验会以
    "临床指南"的身份混进检索结果——用户没法再分辨自己看的是哪一层。
    """
    report = distill("稿子", llm_returning([FRESH]), SOURCE)
    assert report.cards[0].evidence_tier == "lived_experience"


def test_model_cannot_forge_the_evidence_tier():
    poisoned = {**FRESH, "evidence_tier": "clinical_guideline", "needs_review": False}
    report = distill("稿子", llm_returning([poisoned]), SOURCE)
    assert report.cards[0].evidence_tier == "lived_experience"


# --------------------------------------------------------------------------- #
# 校验：坏卡不许漏进来
# --------------------------------------------------------------------------- #
def test_invalid_card_is_rejected_not_silently_kept():
    bad = {"scenario": "", "technique_name": "x", "do": [], "dont": [],
           "example_phrases": [], "why_it_works": ""}
    report = distill("稿子", llm_returning([bad]), SOURCE)
    assert report.cards == [] and len(report.rejected) == 1


def test_mixed_batch_keeps_good_drops_bad():
    report = distill("稿子", llm_returning([FRESH, {"scenario": "只有这一个字段"}]), SOURCE)
    assert len(report.cards) == 1 and len(report.rejected) == 1


def test_empty_array_is_a_legal_answer():
    """文字稿里没有可提取的技巧时，返回空数组是正确行为，不是失败。"""
    report = distill("今天天气不错。", llm_returning([]), SOURCE)
    assert report.cards == [] and report.rejected == []


def test_non_list_output_triggers_self_repair():
    """模型返回了对象而不是数组 → 触发一次自修复重试。"""
    llm = MockProvider()
    state = {"n": 0}

    def handler(s, u):
        state["n"] += 1
        if state["n"] == 1:
            return json.dumps({"cards": [FRESH]}, ensure_ascii=False)  # 错误形状
        return json.dumps([FRESH], ensure_ascii=False)                  # 修好了

    llm.register(MARKER, handler)
    report = distill("稿子", llm, SOURCE)
    assert report.repairs == 1
    assert len(report.cards) == 1


def test_llm_failure_yields_empty_report_not_crash():
    class Failing:
        name = "failing"

        def complete(self, system, user, *, temperature=0.3):
            from core.utils.llm import LLMError
            raise LLMError("down")

    report = distill("稿子", Failing(), SOURCE)
    assert report.cards == []


# --------------------------------------------------------------------------- #
# 去重：标记复核，不静默丢弃
# --------------------------------------------------------------------------- #
def test_near_duplicate_goes_to_review_not_into_cards(cards):
    report = distill("稿子", llm_returning([GOOD]), SOURCE, retriever=BM25Retriever(cards))
    assert report.cards == [], "近义卡不该直接入库"
    assert len(report.needs_review) == 1
    card, dup_id, score = report.needs_review[0]
    assert dup_id == "comm_001"
    assert score >= 0.12


def test_genuinely_new_topic_is_not_flagged(cards):
    report = distill("稿子", llm_returning([FRESH]), SOURCE, retriever=BM25Retriever(cards))
    assert len(report.cards) == 1 and report.needs_review == []


def test_dedup_disabled_when_no_retriever(cards):
    report = distill("稿子", llm_returning([GOOD]), SOURCE, retriever=None)
    assert len(report.cards) == 1


# --------------------------------------------------------------------------- #
# 下载器
# --------------------------------------------------------------------------- #
def test_local_audio_entrypoint(tmp_path):
    f = tmp_path / "clip.m4a"
    f.write_bytes(b"x")
    path, meta = local_audio(f, platform="xiaohongshu", author="某博主")
    assert path == f and meta.platform == "xiaohongshu" and meta.title == "clip"


def test_local_audio_missing_file_raises(tmp_path):
    with pytest.raises(IngestionError):
        local_audio(tmp_path / "nope.m4a")


def test_source_meta_serialises_for_schema():
    d = SourceMeta(platform="douyin", author="a", url="https://x").to_source_dict()
    assert d["platform"] == "douyin" and "date_ingested" in d
