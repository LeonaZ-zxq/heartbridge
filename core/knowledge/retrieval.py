"""检索层：BM25（词法）、Dense（语义）、Hybrid（RRF 融合）三种后端。

为什么不是「直接上向量库」就完事（面试最想听的部分）：

稠密向量擅长语义相似（'不配被爱' ↔ '觉得自己没价值'），
但对**罕见词、专有名词、精确症状名**反而不如老式的关键词匹配——
用户说"过度换气"，BM25 能精确命中，而小模型的中文 embedding 可能把它
和"呼吸困难""胸闷"混在一起排不出先后。

两者的失败模式是**互补**的，所以工业界的标准做法是混合检索 + RRF 融合。
本模块把三种后端都实现出来，并用同一套评测集量化对比——
「我不是因为看到别人这么做才用 hybrid，是因为我测出来它更好」，
这是面试里最值钱的一句话。

RRF (Reciprocal Rank Fusion, Cormack et al. 2009)：
    score(d) = Σ_over_retrievers  1 / (k + rank_r(d))
只用**排名**不用分数，所以不需要在不同量纲的分数之间做归一化——
这正是它比加权求和更稳健的原因。
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol, Sequence

from core.config import CONFIG, Config
from core.knowledge.schema import Card

# --------------------------------------------------------------------------- #
# 中文分词
# --------------------------------------------------------------------------- #
_PUNCT = re.compile(r"[\s，。、！？；：''""（）()《》【】…—·,.!?;:'\"/\\|-]+")
# 极简停用词表。中文检索里这些高频虚词几乎不携带区分度，
# 留着会让 BM25 的 IDF 被稀释。
_STOP = {
    "的", "了", "是", "我", "你", "他", "她", "在", "和", "也", "都", "就", "不",
    "有", "很", "会", "要", "着", "过", "吗", "呢", "啊", "吧", "把", "被", "让",
    "这", "那", "个", "上", "下", "来", "去", "说", "什么", "怎么", "自己",
}


def tokenize(text: str) -> list[str]:
    """中文分词。用 jieba；没装则退化为单字切分（保证核心逻辑永远可跑）。

    单字退化不是摆设：中文 BM25 用 unigram 其实也有相当可用的效果，
    这让这个模块的测试不依赖任何第三方包。
    """
    text = _PUNCT.sub(" ", text)
    try:
        import jieba

        toks = jieba.lcut(text)
    except ImportError:
        toks = list(text)
    return [t for t in (tok.strip() for tok in toks) if t and t not in _STOP]


# --------------------------------------------------------------------------- #
# 通用结构
# --------------------------------------------------------------------------- #
@dataclass
class Hit:
    card: Card
    score: float
    retriever: str = ""

    @property
    def id(self) -> str:
        return self.card.id


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 3, type_filter: str | None = None) -> list[Hit]: ...


def _filtered(cards: Sequence[Card], type_filter: str | None) -> list[Card]:
    if not type_filter:
        return list(cards)
    return [c for c in cards if c.type == type_filter]


# --------------------------------------------------------------------------- #
# BM25：词法检索
# --------------------------------------------------------------------------- #
@dataclass
class BM25Retriever:
    """Okapi BM25，纯 Python 实现，零重依赖。

    参数含义（会被追问）：
    - k1=1.5 控制词频饱和：一个词出现 10 次不该比出现 2 次重要 5 倍。
    - b=0.75 控制文档长度归一化：长文档天然含更多词，要惩罚，但不能罚满。
    这是 IR 领域的经典默认值，除非有评测数据支持，否则不该乱调。
    """

    cards: Sequence[Card]
    k1: float = 1.5
    b: float = 0.75
    name: str = "bm25"
    # 索引文本构造函数。默认用卡片自己的 index_text()；
    # 传入别的函数就能做「索引里放哪些字段」的消融实验。
    text_fn: Callable[[Card], str] | None = None
    _docs: list[list[str]] = field(default_factory=list, init=False)
    _df: dict[str, int] = field(default_factory=dict, init=False)
    _avgdl: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        fn = self.text_fn or (lambda c: c.index_text())
        self._docs = [tokenize(fn(c)) for c in self.cards]
        self._avgdl = (sum(len(d) for d in self._docs) / len(self._docs)) if self._docs else 0.0
        df: dict[str, int] = defaultdict(int)
        for doc in self._docs:
            for term in set(doc):
                df[term] += 1
        self._df = dict(df)

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = self._df.get(term, 0)
        # BM25 的概率式 IDF，加 0.5 平滑；max(…, 极小正数) 防止高频词变负分
        return max(math.log((n - df + 0.5) / (df + 0.5) + 1.0), 1e-9)

    def search(self, query: str, k: int = 3, type_filter: str | None = None) -> list[Hit]:
        q_terms = tokenize(query)
        scored: list[Hit] = []
        for card, doc in zip(self.cards, self._docs):
            if type_filter and card.type != type_filter:
                continue
            if not doc:
                continue
            tf: dict[str, int] = defaultdict(int)
            for t in doc:
                tf[t] += 1
            dl = len(doc)
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
                score += self._idf(term) * (f * (self.k1 + 1)) / denom
            if score > 0:
                scored.append(Hit(card, score, self.name))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]


# --------------------------------------------------------------------------- #
# Dense：语义向量检索
# --------------------------------------------------------------------------- #
@dataclass
class DenseRetriever:
    """sentence-transformers + 余弦相似度。

    模型选 BAAI/bge-small-zh-v1.5：中文优化、只有 ~95MB、本地 CPU 就能跑，
    符合项目「全免费 + 数据不出本机」的隐私约束——
    这不是妥协，是隐私需求推导出的技术选型（面试里要这样讲）。

    重依赖（torch）是**懒加载**的：没装也不影响 BM25 和整个测试套件。
    """

    cards: Sequence[Card]
    model_name: str = "BAAI/bge-small-zh-v1.5"
    name: str = "dense"
    text_fn: Callable[[Card], str] | None = None
    # 向量索引后端：numpy | flat | hnsw
    #
    # ━━━ 三个都实现了，而默认依然是 numpy ━━━
    #
    #   numpy：全量点积。O(N·d)，N=101 时一次查询是几十微秒。
    #   flat ：faiss.IndexFlatIP。**同样是全量精确检索**，只是 SIMD 实现更好。
    #          结果必须和 numpy 逐位一致——不一致就说明实现有 bug，测试锁死了这一点。
    #   hnsw ：faiss.IndexHNSWFlat。近似检索，用召回率换延迟。
    #
    # 近似检索解决的是「候选集大到线性扫描不可接受」的问题。
    # 这个语料 101 张卡，线性扫描本来就不是瓶颈，
    # 上 HNSW 只会**用召回率换一个不存在的性能问题**。
    # 所以默认留在 numpy：faiss 是可选依赖，缺了也不影响整个测试套件。
    #
    # 把三条路都跑通再选最朴素的那条，和「hybrid 输给 dense」是同一件事：
    # 让测量来决定，而不是让流行度决定。
    index: str = "numpy"
    hnsw_m: int = 32            # 每个节点的出边数，faiss 默认量级
    hnsw_ef_search: int = 64    # 查询时的候选队列深度，越大越准越慢
    _model: object | None = field(default=None, init=False)
    _emb: object | None = field(default=None, init=False)
    _index: object | None = field(default=None, init=False)

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        fn = self.text_fn or (lambda c: c.index_text())
        self._emb = self._model.encode(
            [fn(c) for c in self.cards],
            normalize_embeddings=True,  # 归一化后点积 == 余弦相似度，省一次除法
            show_progress_bar=False,
        )
        if self.index != "numpy":
            self._build_faiss()

    def _build_faiss(self) -> None:
        import faiss
        import numpy as np

        emb = np.ascontiguousarray(np.asarray(self._emb), dtype="float32")
        dim = emb.shape[1]
        if self.index == "flat":
            # 向量已归一化，所以内积就是余弦相似度——和 numpy 路径同一个度量
            idx = faiss.IndexFlatIP(dim)
        elif self.index == "hnsw":
            idx = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efSearch = self.hnsw_ef_search
        else:
            raise ValueError(f"未知向量索引: {self.index}（可选 numpy | flat | hnsw）")
        idx.add(emb)
        self._index = idx

    def search(self, query: str, k: int = 3, type_filter: str | None = None) -> list[Hit]:
        self._ensure()
        import numpy as np

        qv = self._model.encode([query], normalize_embeddings=True)[0]  # type: ignore[union-attr]

        if self.index == "numpy":
            sims = np.asarray(self._emb) @ qv
            order = [int(i) for i in np.argsort(-sims)]
            scores = {i: float(sims[i]) for i in order}
        else:
            # 有 type_filter 时要多取：faiss 只会返回全局 top-n，
            # 过滤之后可能一张都不剩。101 张卡全取回来的代价可以忽略。
            n = len(self.cards) if type_filter else min(k, len(self.cards))
            q = np.ascontiguousarray(qv.reshape(1, -1), dtype="float32")
            sims, idxs = self._index.search(q, n)  # type: ignore[union-attr]
            order = [int(i) for i in idxs[0] if i != -1]
            scores = {int(i): float(sc) for i, sc in zip(idxs[0], sims[0]) if i != -1}

        out: list[Hit] = []
        for i in order:
            card = self.cards[i]
            if type_filter and card.type != type_filter:
                continue
            out.append(Hit(card, scores[i], self.name))
            if len(out) >= k:
                break
        return out


# --------------------------------------------------------------------------- #
# Hybrid：RRF 融合
# --------------------------------------------------------------------------- #
@dataclass
class HybridRetriever:
    """把多个检索器的**排名**用 RRF 融合。

    关键：只用 rank，不用 score。
    BM25 的分数是无上界的（可以是 8.3），余弦相似度在 [-1,1]，
    直接加权求和需要先归一化，而归一化本身又要调参、且对分布敏感。
    RRF 绕开了整个问题——这就是它成为工业标准的原因。
    """

    retrievers: Sequence[Retriever]
    rrf_k: int = 60
    name: str = "hybrid"

    def search(self, query: str, k: int = 3, type_filter: str | None = None) -> list[Hit]:
        # 每个子检索器多取一些候选，融合才有的可选
        pool = max(k * 4, 10)
        fused: dict[str, float] = defaultdict(float)
        cards: dict[str, Card] = {}
        for r in self.retrievers:
            for rank, hit in enumerate(r.search(query, k=pool, type_filter=type_filter), start=1):
                fused[hit.id] += 1.0 / (self.rrf_k + rank)
                cards[hit.id] = hit.card
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(cards[cid], score, self.name) for cid, score in ranked]


# --------------------------------------------------------------------------- #
# Rerank：召回—精排两级链路
# --------------------------------------------------------------------------- #
@dataclass
class RerankRetriever:
    """第一级宽召回，第二级 cross-encoder 精排。

    ━━━ 为什么两级，而不是把第一级调得更准 ━━━

    前面三个检索器都是 **bi-encoder**（或词袋）：query 和 card 各自独立
    编码成一个向量，再比相似度。这让检索可以预先建索引、在整个语料上
    一次算完——代价是 query 和 card **从来没有互相看过**，所有交互被
    压缩进一个点积里。

    cross-encoder 把 (query, card) 拼成一对一起送进模型，两边逐 token
    互相注意，判别力明显更强。代价同样明显：它**不能预先建索引**，
    每个候选都要跑一次模型，复杂度 O(候选数) 而不是 O(1)。

    所以工业界的标准形状是两级：
        第一级要**召回**——宽一点，便宜，宁滥勿缺；
        第二级要**精度**——窄一点，昂贵，在小候选集上重排。
    这个类就是第二级，它包装任意一个满足 Retriever 协议的第一级。

    ━━━ candidate_k 为什么是 20，以及为什么没调 ━━━

    20 是这类流水线的通行默认值，**不是在本项目评测集上调出来的**。
    这是刻意的：留出集已经被用于选型、诊断和验证三次，
    见 docs/EVALUATION.md §4.4。再用它调一个超参就是第四次污染，
    而这个项目最值钱的东西恰恰是「我知道哪个数字能对外说」。

    在干净的第三份评测集出现之前，这里只报**不调参的默认配置**。
    """

    base: Retriever
    model_name: str = "BAAI/bge-reranker-base"
    # 第一级放宽到多少个候选交给精排
    candidate_k: int = 20
    name: str = "rerank"
    # 打分函数注入点：传入就不加载真模型。
    # 这让整个测试套件在没有 torch、没有模型权重、没有网络的情况下
    # 依然能验证**重排逻辑本身**——和 DenseRetriever 懒加载同一个思路。
    score_fn: Callable[[str, Sequence[Card]], Sequence[float]] | None = None
    _model: object | None = field(default=None, init=False)

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_name)

    def _score(self, query: str, cards: Sequence[Card]) -> Sequence[float]:
        if self.score_fn is not None:
            return self.score_fn(query, cards)
        self._ensure()
        pairs = [(query, c.index_text()) for c in cards]
        return [float(s) for s in self._model.predict(pairs)]  # type: ignore[union-attr]

    def search(self, query: str, k: int = 3, type_filter: str | None = None) -> list[Hit]:
        # type_filter 交给第一级执行。在这里再过滤一次会先花精排的算力、
        # 再把结果丢掉，而且会让实际参与重排的候选数小于 candidate_k。
        cands = self.base.search(query, k=self.candidate_k, type_filter=type_filter)
        if not cands:
            return []
        scores = self._score(query, [h.card for h in cands])
        if len(scores) != len(cands):
            raise ValueError(
                f"重排打分数量对不上：{len(scores)} 个分数 vs {len(cands)} 个候选"
            )
        ranked = sorted(zip(cands, scores), key=lambda p: p[1], reverse=True)
        return [Hit(h.card, float(s), self.name) for h, s in ranked[:k]]


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #
def build_retriever(
    cards: Iterable[Card],
    backend: str | None = None,
    cfg: Config | None = None,
) -> Retriever:
    cfg = cfg or CONFIG
    backend = (backend or cfg.retrieval_backend).lower()
    cards = list(cards)
    if backend == "bm25":
        return BM25Retriever(cards)
    if backend == "dense":
        return DenseRetriever(cards, cfg.embedding_model, index=cfg.dense_index)
    if backend == "hybrid":
        return HybridRetriever(
            [BM25Retriever(cards), DenseRetriever(cards, cfg.embedding_model, index=cfg.dense_index)],
            rrf_k=cfg.rrf_k,
        )
    if backend == "rerank":
        if cfg.rerank_base.lower() == "rerank":
            raise ValueError("HB_RERANK_BASE 不能是 rerank 自己——会无限递归")
        return RerankRetriever(
            base=build_retriever(cards, backend=cfg.rerank_base, cfg=cfg),
            model_name=cfg.reranker_model,
            candidate_k=cfg.rerank_candidate_k,
        )
    raise ValueError(f"未知检索后端: {backend}")
