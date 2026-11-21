"""向量索引后端：numpy / faiss 精确 / faiss 近似。

━━━ 这组测试在验证什么 ━━━

**单元测试验证逻辑，评测脚本验证质量。** 这里不测「HNSW 召回率掉多少」——
那是 scripts/eval_retrieval.py 在标注集上该回答的问题，一个单元测试
断言近似检索的召回率只会得到一个脆弱的、随机种子一换就红的断言。

这里只锁三件事：
1. faiss 精确检索（IndexFlatIP）必须和 numpy 全量点积**逐位一致**。
   不一致就是实现有 bug，而不是「faiss 的正常误差」——精确检索没有误差。
2. type_filter 在 faiss 路径下语义不变。faiss 只返回全局 top-n，
   过滤后可能一张不剩，所以带 filter 时必须多取。
3. numpy 仍然是默认，faiss 缺席不影响任何东西。

向量是合成的：这让整条 faiss 代码路径在**没有 sentence-transformers、
没有模型权重、没有网络**的情况下被真实执行。
"""
import numpy as np
import pytest

from core.config import CONFIG
from core.knowledge.retrieval import DenseRetriever
from core.knowledge.schema import load_cards

faiss = pytest.importorskip("faiss", reason="faiss 是可选依赖，缺了只跳过这组")

DIM = 16


@pytest.fixture(scope="module")
def cards():
    return load_cards(CONFIG.cards_dir)


@pytest.fixture(scope="module")
def emb(cards):
    rng = np.random.default_rng(20261121)
    m = rng.normal(size=(len(cards), DIM)).astype("float32")
    return m / np.linalg.norm(m, axis=1, keepdims=True)   # 归一化：内积即余弦


class _FakeEncoder:
    """只负责把查询变成一个确定的向量，不碰任何真模型。"""

    def __init__(self, vec):
        self.vec = vec

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        return np.asarray([self.vec], dtype="float32")


def _wired(cards, emb, index, qvec):
    r = DenseRetriever(cards, index=index)
    r._model = _FakeEncoder(qvec)      # 占住 _model，_ensure 就不会去加载真模型
    r._emb = emb
    if index != "numpy":
        r._build_faiss()
    return r


@pytest.fixture
def qvec():
    rng = np.random.default_rng(7)
    v = rng.normal(size=DIM).astype("float32")
    return v / np.linalg.norm(v)


def test_faiss_精确检索必须和_numpy_逐位一致(cards, emb, qvec):
    """精确检索没有「可接受的误差」。对不上就是 bug。"""
    a = _wired(cards, emb, "numpy", qvec).search("q", k=5)
    b = _wired(cards, emb, "flat", qvec).search("q", k=5)
    assert [h.id for h in a] == [h.id for h in b]
    for x, y in zip(a, b):
        assert x.score == pytest.approx(y.score, abs=1e-5)


def test_精确检索在_type_filter_下也一致(cards, emb, qvec):
    a = _wired(cards, emb, "numpy", qvec).search("q", k=3, type_filter="somatic")
    b = _wired(cards, emb, "flat", qvec).search("q", k=3, type_filter="somatic")
    assert [h.id for h in a] == [h.id for h in b]
    assert a and all(h.card.type == "somatic" for h in a)


def test_近似检索返回的仍是合法的卡片(cards, emb, qvec):
    """HNSW 的召回率由评测脚本回答，这里只保证它不返回垃圾。"""
    hits = _wired(cards, emb, "hnsw", qvec).search("q", k=3)
    ids = {c.id for c in cards}
    assert len(hits) == 3
    assert all(h.id in ids for h in hits)
    assert len({h.id for h in hits}) == 3          # 不重复


def test_近似检索尊重_type_filter(cards, emb, qvec):
    hits = _wired(cards, emb, "hnsw", qvec).search("q", k=3, type_filter="somatic")
    assert hits and all(h.card.type == "somatic" for h in hits)


def test_未知索引名字要报错(cards, emb, qvec):
    r = DenseRetriever(cards, index="annoy")
    r._model = _FakeEncoder(qvec)
    r._emb = emb
    with pytest.raises(ValueError, match="未知向量索引"):
        r._build_faiss()


def test_默认仍然是_numpy_不碰_faiss(cards):
    """faiss 是可选依赖。默认路径不能依赖它。"""
    assert DenseRetriever(cards).index == "numpy"
    assert CONFIG.dense_index == "numpy"
