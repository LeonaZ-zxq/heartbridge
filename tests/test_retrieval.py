"""检索质量的回归测试。

这里断言的是**指标下限**而不是具体数值。
理由：知识库会不断加卡、评测集会不断加查询，硬编码 87.5% 会天天变红。
但如果哪次改动让 Recall@3 掉到 80% 以下，那一定是真的退化了，必须拦住。

这是「LLM 应用怎么做回归测试」的一个具体答案——
不是断言模型输出等于某个字符串，而是断言系统级指标不低于基线。
"""
from core.knowledge.evaluator import evaluate
from core.knowledge.retrieval import BM25Retriever, HybridRetriever, tokenize

# 基线由 scripts/eval_retrieval.py 测得（BM25，32 条查询）：
# Recall@1=78.1%  Recall@3=87.5%  MRR=0.828
RECALL_AT_3_FLOOR = 0.80
MRR_FLOOR = 0.75


def test_tokenizer_handles_chinese_and_punctuation():
    toks = tokenize("他说：自己不配被爱……")
    assert "不配" in "".join(toks)
    assert "：" not in toks and "的" not in toks  # 标点和停用词都要被清掉


def test_bm25_meets_recall_floor(cards, eval_queries):
    res = evaluate(BM25Retriever(cards), eval_queries, k=3)
    assert res.recall_at_3 >= RECALL_AT_3_FLOOR, res.summary()
    assert res.mrr >= MRR_FLOOR, res.summary()


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
