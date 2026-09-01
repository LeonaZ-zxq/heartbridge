"""检索质量的回归测试。

这里断言的是**指标下限**而不是具体数值。
理由：知识库会不断加卡、评测集会不断加查询，硬编码 87.5% 会天天变红。
但如果哪次改动让 Recall@3 掉到 80% 以下，那一定是真的退化了，必须拦住。

这是「LLM 应用怎么做回归测试」的一个具体答案——
不是断言模型输出等于某个字符串，而是断言系统级指标不低于基线。
"""
from core.knowledge.evaluator import evaluate
from core.knowledge.retrieval import BM25Retriever, HybridRetriever, tokenize

# ━━━ 两个评测集，两个完全不同的数字，这本身就是这个项目的一个结论 ━━━
#
# dev  (retrieval_eval.json)    BM25 Recall@3 = 100.0%
# holdout (retrieval_holdout.json) BM25 Recall@3 =  36.7%
#
# 差距的原因是**测试集泄漏**：卡片上的 user_phrasings（文档扩展）是在
# 看过 dev 集失败案例之后写的，等于把答案抄进了索引。dev 集的分数因此
# 永久性地偏乐观，只能用作回归哨兵，不能对外报。
# holdout 集是之后独立撰写、全部为改写式查询的，它才是诚实的数字，
# 也定量地证明了：**纯词法检索在同义改写上会崩，语义检索不是可选项。**
RECALL_AT_3_FLOOR = 0.80          # dev 集回归下限
MRR_FLOOR = 0.75
HOLDOUT_RECALL_FLOOR = 0.33       # holdout 集当前 BM25 水平；接入 dense 后应大幅提高


def test_tokenizer_handles_chinese_and_punctuation():
    toks = tokenize("他说：自己不配被爱……")
    assert "不配" in "".join(toks)
    assert "：" not in toks and "的" not in toks  # 标点和停用词都要被清掉


def test_bm25_meets_recall_floor(cards, eval_queries):
    res = evaluate(BM25Retriever(cards), eval_queries, k=3)
    assert res.recall_at_3 >= RECALL_AT_3_FLOOR, res.summary()
    assert res.mrr >= MRR_FLOOR, res.summary()


def test_holdout_set_does_not_regress(cards):
    """留出集的回归哨兵。

    下限设得低，因为纯 BM25 在这个集上本来就弱——
    重点不是分数高，而是**不许悄悄变得更差**，以及提醒任何读代码的人：
    对外要报的是这个数字，不是 dev 集那个 100%。
    """
    from pathlib import Path

    from core.knowledge.evaluator import load_eval_set

    holdout = load_eval_set(Path(__file__).parent / "fixtures/retrieval_holdout.json")
    res = evaluate(BM25Retriever(cards), holdout, k=3)
    assert res.recall_at_3 >= HOLDOUT_RECALL_FLOOR, res.summary()


def test_type_filter_restricts_results(cards):
    """/soma 这类速查入口依赖 type 过滤，必须只返回躯体化卡。"""
    hits = BM25Retriever(cards).search("他喘不上气手麻", k=3, type_filter="somatic")
    assert hits and all(h.card.type == "somatic" for h in hits)


def test_somatic_query_hits_right_card(cards):
    hits = BM25Retriever(cards).search("他喘不上气，手都麻了", k=3)
    assert "soma_002" in [h.id for h in hits]


def test_rrf_fusion_prefers_consensus():
    """RRF 的核心行为：被多个检索器同时排在前面的文档应该胜出。

    用两个假检索器验证融合逻辑本身，不依赖任何模型——
    这正是把 LLM/embedding 关在接口之外带来的好处。
    """
    class Fake:
        name = "fake"

        def __init__(self, order):
            self.order = order

        def search(self, query, k=3, type_filter=None):
            from core.knowledge.retrieval import Hit
            return [Hit(c, 1.0, "fake") for c in self.order[:k]]

    class C:
        def __init__(self, i):
            self.id = f"comm_{i:03d}"
            self.type = "communication"

    a, b, c = C(1), C(2), C(3)
    # 检索器一: a,b,c   检索器二: b,a,c  → b 综合排名最好
    fused = HybridRetriever([Fake([a, b, c]), Fake([b, a, c])]).search("q", k=3)
    assert fused[0].id in {"comm_001", "comm_002"}
    assert fused[-1].id == "comm_003"  # 两边都垫底的必须排最后
