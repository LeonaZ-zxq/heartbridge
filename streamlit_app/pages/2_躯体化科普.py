"""躯体化症状速查：在场 / 远程两栏并列。

两栏并列不是排版偏好，是产品判断：
异地关系里「抱抱他」这类建议是不可执行的，把可执行的那一栏
直接放在眼前，比让用户自己从一段话里筛出适用的部分更有用。
"""
from __future__ import annotations

import streamlit as st
from shared import crisis_sidebar, footer, get_cards, page_setup

page_setup("躯体化科普", "🩺")
crisis_sidebar()

st.title("🩺 躯体化症状：怎么办")
st.markdown(
    "抑郁与焦虑常伴随真实的身体反应。这些不是「想出来的」，"
    "但也**不能**因此跳过必要的医学检查。每张卡都标了「什么情况必须就医」。"
)
st.warning(
    "本页内容整理自 Beyond Blue 与 Healthdirect Australia 的公开材料，"
    "仅供参考，不能替代医生的判断。症状剧烈或首次出现时请先就医排除躯体疾病。",
    icon="⚕️",
)

soma = [c for c in get_cards() if c.type == "somatic"]
names = [c.symptom for c in soma]
pick = st.selectbox("选择症状", names, index=0)
card = next(c for c in soma if c.symptom == pick)

if card.aliases:
    st.caption("也可能这样描述：" + " · ".join(card.aliases))

st.info(f"**这是什么** — {card.what_it_is}")

left, right = st.columns(2, gap="large")
with left:
    st.markdown("### 🏠 你在他身边时")
    for x in card.in_person:
        st.markdown(f"- {x}")
with right:
    st.markdown("### 📱 异地 / 只能线上时")
    for x in card.remote:
        st.markdown(f"- {x}")

st.divider()
a, b = st.columns(2, gap="large")
with a:
    st.markdown("### ✅ 可以说")
    for x in card.say:
        st.markdown(f"> {x}")
with b:
    st.markdown("### ❌ 不要说")
    for x in card.avoid_saying:
        st.markdown(f"- {x}")

st.error("### 🚑 这些情况必须就医或走危机流程\n" + "\n".join(f"- {x}" for x in card.seek_help_if))

src = card.source
st.caption(f"来源：{src.authority or src.platform}" + (f" · [原文]({src.url})" if src.url else ""))
footer()
