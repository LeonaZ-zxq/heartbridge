"""HeartBridge — 情境求助（主页）。

UI 层刻意做得很薄：它不含任何业务逻辑，只调用 core.engine.pipeline.advise。
这就是分层设计的回报——同一套逻辑，CLI 和网页共用，行为完全一致，
而所有测试都跑在 core 上，不需要为 UI 单独写一套。
"""
from __future__ import annotations

import streamlit as st
from shared import (
    api_key_sidebar, backend_name, crisis_sidebar, demo_banner, demo_profile,
    footer, get_llm_for_ui, get_retriever, is_demo, page_setup,
)

from core.engine.pipeline import advise
from core.safety.detector import RiskLevel

page_setup("情境求助")
api_key_sidebar()
crisis_sidebar()

st.title("🌉 HeartBridge")
st.markdown(
    "**把你卡住的那段对话粘进来。** 系统会判断风险等级、检索有出处的应对建议，"
    "给出 2–3 个回复选项——每个都附上「为什么这么说有效」。"
)
demo_banner()

EXAMPLES = {
    "自我否定": "小鱼 23:45\n我觉得我就是个废物\n你为什么还要跟我在一起",
    "躯体化（异地）": "小鱼 02:10\n我喘不上气\n手也开始麻了\n我是不是要死了",
    "推开你": "小鱼 21:02\n你别管我了\n你走吧",
    "危机信号": "小鱼 03:12\n谢谢你这些年\n你要照顾好自己",
}

col_l, col_r = st.columns([3, 2], gap="large")

with col_r:
    st.markdown("##### 试试这些例子")
    for label, text in EXAMPLES.items():
        if st.button(label, use_container_width=True, key=f"ex_{label}"):
            st.session_state["transcript"] = text
    st.caption("最后一个例子会触发危机分支——注意它**不会**走正常的检索和生成流程。")

with col_l:
    transcript = st.text_area(
        "聊天记录 / 情境描述",
        key="transcript",
        height=200,
        placeholder="小鱼 23:45\n我觉得我就是个废物",
    )
    situation = st.text_input(
        "补充说明（可选）", placeholder="例如：我在墨尔本，他在国内，现在只能打字"
    )
    use_profile = st.toggle("注入伴侣档案（演示数据）", value=True,
                            help="真实档案只存在本机 SQLite，永不上传。这里用的是虚构示例。")
    go = st.button("给我建议", type="primary", use_container_width=True, key="go")

if go and transcript.strip():
    with st.spinner("检索中…"):
        result = advise(
            transcript,
            get_retriever(backend_name()),
            get_llm_for_ui(),
            demo_profile() if use_profile else None,
            situation=situation,
        )

    st.divider()

    if result.is_crisis:
        st.error("检测到危机信号，已跳过正常流程。", icon="🚨")
        st.markdown(result.crisis_response)
        with st.expander("这里发生了什么（技术说明）"):
            st.markdown(
                "- 规则层命中，管道**短路**：没有做检索，也没有调用生成模型。\n"
                "- 你看到的内容是**人工审查过的固定模板**，不是生成的。相同输入永远得到相同输出。\n"
                f"- 触发依据：`{result.risk.rationale}`"
            )
    else:
        if result.risk.level == RiskLevel.ELEVATED:
            st.warning("这段话里有明显的痛苦信号。回复之后建议继续留意。", icon="⚠️")

        if not result.options:
            st.info("没有生成回复选项（可能是模型调用失败）。下面是检索到的依据卡片。")

        for i, opt in enumerate(result.options, 1):
            with st.container(border=True):
                head = f"**选项 {i}**" + (f" · {opt.style}" if opt.style else "")
                st.markdown(head)
                st.markdown(f"### {opt.text}")
                st.markdown(f"**为什么有效：** {opt.why}")
                if opt.grounded:
                    st.caption(f"依据卡片：`{opt.card_id}`")
                else:
                    st.caption(f"⚠️ 引用了未检索到的卡片 `{opt.card_id}`——已标记为不可信")

        if result.hits:
            with st.expander(f"检索到的 {len(result.hits)} 张卡片（后端：{backend_name()}）"):
                for h in result.hits:
                    c = h.card
                    title = c.technique_name if c.type == "communication" else f"躯体化：{c.symptom}"
                    st.markdown(f"**`{c.id}` · {title}**")
                    st.caption(f"来源：{c.source.authority or c.source.platform} · {c.source.url or ''}")

elif go:
    st.warning("先粘贴一段对话，或点右边的例子。")

footer()
