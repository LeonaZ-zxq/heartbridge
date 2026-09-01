"""集中式配置。所有可调参数都在这里，代码里不出现硬编码的模型名/阈值/路径。

设计理由（面试点）：
- 免费 LLM 模型经常下线或改名，模型名必须是配置项，一行切换，不用改业务代码。
- 阈值（检索 top-k、危机检测置信度）是需要调参的实验变量，集中管理才能做消融实验。
- 路径集中管理，保证 CLI / 测试 / 未来的 Web UI 读写的是同一份数据。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv 是可选依赖，没装也不该崩
    pass

ROOT = Path(__file__).resolve().parent.parent


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Config:
    # ---------- 路径 ----------
    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    # 种子知识库：来自公开权威来源，进 git。
    cards_dir: Path = ROOT / "knowledge_base" / "cards"
    # 私密数据：恋人 profile、聊天记录、转写稿，永不进 git。
    private_dir: Path = ROOT / "data"
    chroma_dir: Path = ROOT / "data" / "chroma"
    db_path: Path = ROOT / "data" / "heartbridge.db"
    transcripts_dir: Path = ROOT / "data" / "transcripts"

    # ---------- LLM ----------
    # provider: mock | openrouter | gemini
    # mock 是刻意设计的：测试和 CI 不该依赖网络和 API key，也不该花钱。
    llm_provider: str = field(default_factory=lambda: _env("HB_LLM_PROVIDER", "mock"))
    openrouter_key: str = field(default_factory=lambda: _env("OPENROUTER_API_KEY", ""))
    openrouter_model: str = field(
        default_factory=lambda: _env("HB_OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    )
    gemini_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: _env("HB_GEMINI_MODEL", "gemini-2.5-flash"))
    llm_timeout_s: int = field(default_factory=lambda: _env_int("HB_LLM_TIMEOUT", 60))

    # ---------- 检索 ----------
    # backend: bm25 | dense | hybrid
    # 默认是 dense，不是 hybrid —— 这是被留出集数据推翻默认直觉的结果：
    #   留出集 Recall@3    bm25 36.7% | dense 76.7% | hybrid 70.0%
    # RRF 对两个检索器等权重融合，当其中一个明显更弱时，它会把结果拖下来。
    # 「混合检索总是更好」是个流行说法，在这个语料上不成立。
    retrieval_backend: str = field(default_factory=lambda: _env("HB_RETRIEVAL_BACKEND", "dense"))
    embedding_model: str = field(
        default_factory=lambda: _env("HB_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    )
    top_k: int = field(default_factory=lambda: _env_int("HB_TOP_K", 3))
    # RRF 融合的平滑常数，业界惯例 60，见 Cormack et al. 2009
    rrf_k: int = field(default_factory=lambda: _env_int("HB_RRF_K", 60))

    # ---------- 安全 ----------
    # 危机检测走「宁可误报，不可漏报」：召回率优先于准确率。
    crisis_llm_second_pass: bool = field(
        default_factory=lambda: _env("HB_CRISIS_LLM_PASS", "1") == "1"
    )
    crisis_rule_threshold: float = field(
        default_factory=lambda: _env_float("HB_CRISIS_RULE_THRESHOLD", 0.5)
    )

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.chroma_dir, self.transcripts_dir):
            d.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
