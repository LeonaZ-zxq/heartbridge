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
_PUNCT = re.compile(r"[\s，。、！？；：''""（）()《》【】…—\-·,.!?;:'\"/\\|]+")
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
    _model: object | None = field(default=None, init=False)
    _emb: object | None = field(default=None, init=False)

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

    def search(self, query: str, k: int = 3, type_filter: str | None = None) -> list[Hit]:
        self._ensure()
        import numpy as np

        qv = self._model.encode([query], normalize_embeddings=True)[0]  # type: ignore[union-attr]
        sims = np.asarray(self._emb) @ qv
        order = np.argsort(-sims)
        out: list[Hit] = []
        for i in order:
            card = self.cards[int(i)]
            if type_filter and card.type != type_filter:
                continue
            out.append(Hit(card, float(sims[int(i)]), self.name))
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
        return DenseRetriever(cards, cfg.embedding_model)
    if backend == "hybrid":
        return HybridRetriever(
            [BM25Retriever(cards), DenseRetriever(cards, cfg.embedding_model)],
            rrf_k=cfg.rrf_k,
        )
    raise ValueError(f"未知检索后端: {backend}")
