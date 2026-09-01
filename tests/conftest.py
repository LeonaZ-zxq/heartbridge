import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ━━━ 测试必须是封闭的（hermetic）━━━
# 这几行要在 core.config 被导入**之前**执行。
# 起因是一次真实的失败：开发者在 .env 里配了真实的 API key 之后，
# 界面测试不再走 demo 分支，转而去调真实模型，于是在没有网络的 CI 上必挂。
# 一个会因为「谁的机器上有什么配置」而变红的测试，是没有价值的。
# python-dotenv 默认不覆盖已存在的环境变量，所以这里设死就能盖住 .env。
os.environ["HB_LLM_PROVIDER"] = "mock"
os.environ["OPENROUTER_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["HB_RETRIEVAL_BACKEND"] = "bm25"   # 测试不依赖 embedding 模型下载

from core.config import CONFIG  # noqa: E402
from core.knowledge.schema import load_cards  # noqa: E402


@pytest.fixture(scope="session")
def cards():
    return load_cards(CONFIG.cards_dir)


@pytest.fixture(scope="session")
def eval_queries():
    from core.knowledge.evaluator import load_eval_set

    return load_eval_set(ROOT / "tests/fixtures/retrieval_eval.json")
