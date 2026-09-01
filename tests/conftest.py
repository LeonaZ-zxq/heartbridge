import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import CONFIG  # noqa: E402
from core.knowledge.schema import load_cards  # noqa: E402


@pytest.fixture(scope="session")
def cards():
    return load_cards(CONFIG.cards_dir)


@pytest.fixture(scope="session")
def eval_queries():
    from core.knowledge.evaluator import load_eval_set

    return load_eval_set(ROOT / "tests/fixtures/retrieval_eval.json")
