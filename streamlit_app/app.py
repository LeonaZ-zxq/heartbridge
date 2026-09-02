"""HeartBridge — 情境求助（主页）。

UI 层刻意做得很薄：它不含任何业务逻辑，只调用 core.engine.pipeline.advise。
这就是分层设计的回报——同一套逻辑，CLI 和网页共用，行为完全一致，
而所有测试都跑在 core 上，不需要为 UI 单独写一套。
"""
from __future__ import annotations

import streamlit as st
from shared import (
    active_profile, api_key_sidebar, backend_name, crisis_sidebar, demo_banner,
    footer, get_llm_for_ui, get_retriever, is_demo, page_setup, profile_is_custom,
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
    _custom = profile_is_custom()
    use_profile = st.toggle(
        "注入伴侣档案" + ("" if _custom else "（演示数据）"),
        value=True,
        help=("正在用你自己填的档案。" if _custom else
              "现在用的是虚构示例（小鱼）。去「伴侣档案」那一页填成你的情况，"
              "建议会明显更贴。档案只留在本次会话，不落盘。"),
    )
    go = st.button("给我建议", type="primary", use_container_width=True, key="go")

if go and transcript.strip():
    with st.spinner("检索中…"):
        result = advise(
            transcript,
            get_retriever(backend_name()),
            get_llm_for_ui(),
            active_profile() if use_profile else None,
            situation=situation,
        )

    st.divider()

    if result.is_crisis:
        st.error("检测到危机信号，已跳过正常流程。", icon="🚨")
        st.markdown(result.crisis_response)
        if result.hits:
            st.markdown("### 针对这个信号，具体可以怎么做")
            st.caption("以下内容来自人工撰写、标注临床来源的危机卡片——**不是模型生成的**。")
            for h in result.hits:
                card = h.card
                with st.container(border=True):
                    st.markdown(f"**{card.signal}**")
                    st.caption(card.what_it_means)
                    st.markdown("**现在就做**")
                    for x in card.do_now:
                        st.markdown(f"- {x}")
                    st.markdown("**可以这样说**")
                    for x in card.say:
                        st.markdown(f"- 「{x}」")
                    with st.expander("不要说 / 什么情况必须立刻求助"):
                        for x in card.avoid_saying:
                            st.markdown(f"- ✗ {x}")
                        st.markdown("**出现这些就立刻求助专业或急救：**")
                        for x in card.escalate_if:
                            st.markdown(f"- ⚠️ {x}")
                    st.caption(f"来源：{card.source.authority or card.source.url}"
                               + ("　·　⚠ 尚未人工复核" if card.needs_review else ""))

        with st.expander("这里发生了什么（技术说明）"):
            st.markdown(
                "- 规则层命中，管道**短路**：没有调用生成模型。\n"
                "- 固定模板 + 按信号检索到的危机卡片，两者都是人工审查过的文本。\n"
                "- 相同输入永远得到相同输出——这条路径上没有任何随机性。\n"
                f"- 触发依据：`{result.risk.rationale}`"
            )
    else:
        if result.risk.level == RiskLevel.ELEVATED:
            st.warning("这段话里有明显的痛苦信号。回复之后建议继续留意。", icon="⚠️")

        if not result.options:
            # 「可能」两个字是个信号：说明这里在猜。
            # generator 现在把失败原因随返回值带出来了，如实说。
            err = getattr(result.options, "error", "")
            kind = getattr(result.options, "kind", "")
            if kind == "validation":
                # 这里以前写的是「引用无效或过于笼统」——那是猜的，而且是错的：
                # 引用校验和套话校验都带 `or options` 兜底，永远不可能把列表清空。
                # 真正会清空的只有结构校验。照着错误的原因去修，只会越修越远。
                st.warning("模型答了，JSON 也解析出来了，但没有一条选项能用——"
                           "重问一次之后仍然不行。下面是它两次到底返回了什么。", icon="🧪")
                st.code(err)
                st.caption("常见原因：模型改了键名（把 text 写成 reply、把 options 写成中文键）、"
                           "返回了空的 options 数组、或者每条都没写 why。"
                           "换一个更会守格式的模型通常就好了。")
            elif err:
                st.error(f"模型调用失败，没有生成回复选项：\n\n`{err}`", icon="🔌")
                if "429" in err or "quota" in err.lower():
                    st.caption("免费额度用完了。等额度重置，或在侧边栏换一个 key。")
                elif "超时" in err or "timeout" in err.lower():
                    st.caption("请求超时。可以再试一次；长文本比短文本更容易撞上超时。")
            else:
                st.warning("没有生成回复选项。下面是检索到的依据卡片。", icon="🧪")

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
