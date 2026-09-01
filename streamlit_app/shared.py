"""Streamlit 各页共用的加载与展示组件。

部署约束（这本身就是一个可以讲的工程取舍）：
Streamlit Community Cloud 免费实例内存有限，装不下 torch + embedding 模型，
所以**公开 demo 强制走 BM25 后端**。本地运行时才用 dense。
这不是降级凑合，而是把「哪个环境用哪个后端」变成一个显式配置，
并且在界面上如实告诉访问者他看到的是哪一个。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import CONFIG  # noqa: E402
from core.knowledge.retrieval import build_retriever  # noqa: E402
from core.knowledge.schema import load_cards  # noqa: E402
from core.profile.models import PartnerProfile  # noqa: E402
from core.safety.templates import DISCLAIMER, RESOURCES_AU, RESOURCES_CN  # noqa: E402
from core.utils.llm import MockProvider, get_llm  # noqa: E402

DEMO_OPTIONS = {
    "options": [
        {"text": "我听到了。你现在真的这么觉得，那一定很沉。",
         "why": "先验证情绪、不反驳。抑郁认知下直接反驳会被读成「你不理解我」，反而增加孤立感。",
         "card_id": "comm_001", "style": "共情"},
        {"text": "你不用现在就相信我说的。我就在这儿，不走。",
         "why": "把「相信」的压力拿掉，只承诺在场。稳定的在场比说服更能反驳「所有人都会走」。",
         "card_id": "comm_001", "style": "陪伴"},
    ]
}


def page_setup(title: str, icon: str = "🌉") -> None:
    st.set_page_config(page_title=f"HeartBridge · {title}", page_icon=icon, layout="wide")


@st.cache_resource(show_spinner=False)
def get_cards():
    return load_cards(CONFIG.cards_dir)


@st.cache_resource(show_spinner="正在建立检索索引…")
def get_retriever(backend: str):
    return build_retriever(load_cards(CONFIG.cards_dir), backend=backend)


def backend_name() -> str:
    """公开 demo 固定 BM25；本地可通过环境变量切 dense。"""
    return CONFIG.retrieval_backend if not is_demo() else "bm25"


def is_demo() -> bool:
    """没有配置真实 LLM 就是 demo 模式。"""
    return CONFIG.llm_provider.lower() == "mock" or not (
        CONFIG.openrouter_key or CONFIG.gemini_key
    )


def get_llm_for_ui():
    if not is_demo():
        return get_llm(CONFIG)
    llm = MockProvider()
    llm.register("硬性要求", lambda s, u: json.dumps(DEMO_OPTIONS, ensure_ascii=False))
    llm.register("危机信号识别器", lambda s, u: json.dumps({"level": "none", "reason": "demo"}))
    return llm


def demo_profile() -> PartnerProfile:
    """演示用的**假**档案。真实档案只存在本机 SQLite，永不上云。"""
    data = json.loads((ROOT / "examples/sample_profile.json").read_text(encoding="utf-8"))
    data.pop("_note", None)
    return PartnerProfile.model_validate(data)


def demo_banner() -> None:
    if is_demo():
        st.info(
            "**演示模式** — 回复由固定示例生成，不调用任何语言模型；档案是虚构数据。"
            "检索、危机检测、卡片库都是真实运行的。本地配置 API key 后即为完整功能。",
            icon="🧪",
        )


def crisis_sidebar() -> None:
    """危机资源固定在侧边栏，每一页都在。

    这不是装饰。产品定位决定了任何一页都可能是用户在最糟的时刻打开的那一页。
    """
    with st.sidebar:
        st.markdown("### 🆘 紧急求助")
        st.markdown("**澳洲**")
        for r in RESOURCES_AU:
            st.markdown(f"- {r.line()}")
        st.markdown("**中国大陆**")
        for r in RESOURCES_CN:
            st.markdown(f"- {r.line()}")
        st.divider()
        st.caption(DISCLAIMER)


def footer() -> None:
    st.divider()
    st.caption(
        f"{DISCLAIMER}  \n"
        "HeartBridge · 知识来源 Beyond Blue / Healthdirect Australia · "
        "[GitHub](https://github.com/LeonaZ-zxq/heartbridge)"
    )
