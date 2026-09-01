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
from core.utils.llm import (  # noqa: E402
    GeminiProvider, MockProvider, OpenRouterProvider, get_llm,
)

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


# --------------------------------------------------------------------------- #
# 自带 key（BYOK）
# --------------------------------------------------------------------------- #
# 第一版公开 demo 没有 key 就只能看硬编码示例。那是个错误的产品决策：
# 访问者打开看到的是一堆假回复，而**真正的功能一次都没被演示到**。
#
# 正确的做法是 BYOK（bring your own key）：让访问者填自己的 key。
#   - key 只存在 st.session_state（这一个浏览器会话的服务端内存里），
#     不落盘、不进日志、不写进 URL，关掉标签页就没了
#   - 不填也能用：检索、危机检测、卡片库、躯体化速查全都是真实运行的，
#     只有「生成回复」这一步需要模型
# 这既解决了「谁付钱」的问题，也让隐私叙事保持一致：
# 本来就不该由我来托管别人伴侣的对话。

_KEY_SS = "hb_user_api_key"
_PROVIDER_SS = "hb_user_provider"


def user_key() -> tuple[str, str]:
    """返回 (provider, key)。没填就是 ("", "")。"""
    return st.session_state.get(_PROVIDER_SS, ""), st.session_state.get(_KEY_SS, "")


def has_llm() -> bool:
    _, k = user_key()
    return bool(k) or (
        CONFIG.llm_provider.lower() != "mock"
        and bool(CONFIG.openrouter_key or CONFIG.gemini_key)
    )


def is_demo() -> bool:
    """没有任何可用的模型通道 = demo 模式。"""
    return not has_llm()


def api_key_sidebar() -> None:
    """侧边栏里的 key 输入。每一页都有。"""
    with st.sidebar:
        st.markdown("### 🔑 用你自己的 key")
        if has_llm() and user_key()[1]:
            st.success("已连接，本页的回复是实时生成的。", icon="✅")
            if st.button("断开并清除", use_container_width=True):
                st.session_state.pop(_KEY_SS, None)
                st.session_state.pop(_PROVIDER_SS, None)
                st.rerun()
        else:
            st.caption(
                "不填也能用：检索、危机检测、卡片库、躯体化速查都是真的。"
                "只有「生成回复」需要模型。"
            )
            prov = st.selectbox("供应商", ["openrouter", "gemini"], key="_prov_pick")
            key = st.text_input(
                "API key", type="password", key="_key_input",
                placeholder="sk-or-v1-…" if prov == "openrouter" else "AIza…",
                help="只存在这一次浏览器会话里，不落盘、不进日志。关掉标签页就没了。",
            )
            if st.button("连接", type="primary", use_container_width=True, disabled=not key):
                st.session_state[_KEY_SS] = key.strip()
                st.session_state[_PROVIDER_SS] = prov
                st.rerun()
            st.caption(
                "[OpenRouter 免费 key](https://openrouter.ai/keys) · "
                "[Gemini 免费 key](https://aistudio.google.com/apikey)"
            )
        st.divider()


def get_llm_for_ui():
    prov, key = user_key()
    if key:
        # 访问者自带的 key 优先于服务端配置
        if prov == "gemini":
            return GeminiProvider(key, CONFIG.gemini_model, CONFIG.llm_timeout_s)
        return OpenRouterProvider(key, CONFIG.openrouter_model, CONFIG.llm_timeout_s)
    if has_llm():
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
            "**你现在看到的回复是写死的示例，不是模型生成的。** "
            "左边侧边栏填一个自己的 API key（免费的就行）就会变成实时生成。\n\n"
            "不填也没关系：检索、危机检测、卡片库、躯体化速查全都是真实运行的，"
            "档案用的是虚构数据。",
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
