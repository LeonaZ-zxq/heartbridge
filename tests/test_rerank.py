"""两级重排：第一级宽召回，第二级 cross-encoder 精排。

这组测试全部**离线**跑完：注入 score_fn 就不会加载 bge-reranker，
所以没有 torch、没有模型权重、没有网络时，重排逻辑本身依然被验证。
这和 DenseRetriever 的懒加载是同一个约束——
「199 条测试，无需 API Key / GPU」是这个仓库的硬性承诺，
加一个重依赖不能把它破坏掉。
"""
from dataclasses import replace

import pytest

from core.config import Config
from core.knowledge.retrieval import (
    BM25Retriever,
    RerankRetriever,
    build_retriever,
)
from core.knowledge.schema import load_cards

QUERY = "他说他什么都做不好"


@pytest.fixture(scope="module")
def cards():
    from core.config import CONFIG
    return load_cards(CONFIG.cards_dir)


def _reversing(query, cs):
    """给靠后的候选更高的分——把第一级的顺序整个倒过来。

    用一个「明显错」的打分函数，是为了让断言只可能因为
    **重排真的生效了**而通过。打真分的模型做不到这种判别力。
    """
    return list(range(len(cs)))


def test_重排真的改变了顺序(cards):
    base = BM25Retriever(cards)
    rr = RerankRetriever(base=base, candidate_k=10, score_fn=_reversing)
    before = [h.id for h in base.search(QUERY, k=10)]
    after = [h.id for h in rr.search(QUERY, k=3)]
    assert after == before[-3:][::-1]


def test_重排只能重排不能凭空变出卡片(cards):
    """第二级是精排，不是检索。第一级没召回的卡，它无权引入。

    这和 generator 里的 card_id 校验是同一条原则：
    链路上每一环都只能收窄，不能偷偷扩大候选集。
    """
    base = BM25Retriever(cards)
    rr = RerankRetriever(base=base, candidate_k=8, score_fn=_reversing)
    allowed = {h.id for h in base.search(QUERY, k=8)}
    assert {h.id for h in rr.search(QUERY, k=3)} <= allowed


def test_candidate_k_决定有多少候选进第二级(cards):
    seen = {}

    def spy(query, cs):
        seen["n"] = len(cs)
        return [1.0] * len(cs)

    RerankRetriever(base=BM25Retriever(cards), candidate_k=5, score_fn=spy).search(QUERY, k=3)
    assert seen["n"] <= 5
    RerankRetriever(base=BM25Retriever(cards), candidate_k=20, score_fn=spy).search(QUERY, k=3)
    assert seen["n"] > 5


def test_type_filter_由第一级执行(cards):
    """在第二级再过滤会先花精排的算力再丢掉结果，

    而且会让实际参与重排的候选数悄悄小于 candidate_k。
    """
    rr = RerankRetriever(
        base=BM25Retriever(cards), candidate_k=20,
        score_fn=lambda q, cs: [1.0] * len(cs),
    )
    hits = rr.search("他胸口发闷喘不上气", k=3, type_filter="somatic")
    assert hits
    assert all(h.card.type == "somatic" for h in hits)


def test_第一级空结果时不加载模型也不报错(cards):
    """没有候选就没有可排的。这里**不能**去碰模型——

    一个拼写错误的查询不应该触发一次 400MB 的权重加载。
    """
    class _Empty:
        name = "empty"

        def search(self, query, k=3, type_filter=None):
            return []

    rr = RerankRetriever(base=_Empty())        # 注意：没有注入 score_fn
    assert rr.search(QUERY, k=3) == []
    assert rr._model is None                   # 真模型从未被加载


def test_注入打分函数时绝不加载真模型(cards):
    """离线承诺的回归测试。"""
    rr = RerankRetriever(base=BM25Retriever(cards), score_fn=_reversing)
    rr.search(QUERY, k=3)
    assert rr._model is None


def test_打分数量对不上要响亮地失败(cards):
    """静默截断会让重排结果和候选错位——那是最难查的一类 bug。"""
    rr = RerankRetriever(
        base=BM25Retriever(cards), candidate_k=10,
        score_fn=lambda q, cs: [1.0],          # 只给一个分
    )
    with pytest.raises(ValueError, match="数量对不上"):
        rr.search(QUERY, k=3)


def test_工厂按配置把重排包在第一级上(cards):
    cfg = replace(Config(), rerank_base="bm25", rerank_candidate_k=17)
    r = build_retriever(cards, backend="rerank", cfg=cfg)
    assert isinstance(r, RerankRetriever)
    assert isinstance(r.base, BM25Retriever)
    assert r.candidate_k == 17


def test_第一级不能配成重排自己(cards):
    cfg = replace(Config(), rerank_base="rerank")
    with pytest.raises(ValueError, match="无限递归"):
        build_retriever(cards, backend="rerank", cfg=cfg)
