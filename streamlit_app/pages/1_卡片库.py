"""沟通技巧卡片库：可筛选浏览。

这一页的作用不只是展示，它是**知识库的人工审查界面**——
每张卡的来源都可点开核对，这是「可溯源」从设计变成日常操作的地方。
"""
from __future__ import annotations

import streamlit as st
from shared import crisis_sidebar, footer, get_cards, page_setup

page_setup("卡片库", "📚")
crisis_sidebar()

st.title("📚 沟通技巧卡片库")
cards = [c for c in get_cards() if c.type == "communication"]
all_tags = sorted({t for c in cards for t in c.tags})

col1, col2 = st.columns([2, 3])
with col1:
    query = st.text_input("关键词筛选", placeholder="例如：自我否定 / 异地 / 就医")
with col2:
    tags = st.multiselect("按主题筛选", all_tags)

shown = [
    c for c in cards
    if (not query or query in c.scenario + c.technique_name + " ".join(c.user_phrasings))
    and (not tags or set(tags) & set(c.tags))
]
st.caption(f"共 {len(cards)} 张卡，当前显示 {len(shown)} 张。每张卡都有可追溯的来源。")

for c in shown:
    with st.expander(f"`{c.id}` · {c.technique_name} — {c.scenario}"):
        a, b = st.columns(2)
        with a:
            st.markdown("**✅ 要做**")
            for x in c.do:
                st.markdown(f"- {x}")
        with b:
            st.markdown("**❌ 不要做**")
            for x in c.dont:
                st.markdown(f"- {x}")
        st.markdown("**可以这样说**")
        for x in c.example_phrases:
            st.markdown(f"> {x}")
        st.markdown(f"**为什么有效：** {c.why_it_works}")
        if c.tags:
            st.caption("主题：" + " · ".join(c.tags))
        src = c.source
        label = src.authority or src.author or src.platform or "—"
        st.caption(f"来源：{label}" + (f" · [原文]({src.url})" if src.url else ""))

footer()
