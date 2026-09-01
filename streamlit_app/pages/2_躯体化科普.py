"""躯体化症状速查：搜索 → 详细解释 → 伴侣能做什么。

为什么是搜索而不是下拉框（第一版是下拉框，是错的）：
下拉框要求用户**先知道这个症状的学名**才能找到它。但真实的用法是
半夜两点、对方说「我胸口发闷心跳好快」，而使用者根本不知道该找
「惊恐发作」还是「心悸」。他会照着对方的原话去搜。

所以这里直接复用系统已有的检索器：输入口语描述 → 匹配症状卡。
`aliases` 字段（「起来就晕」「脑子像糊了」）本来就是为这个准备的——
和沟通卡的 user_phrasings 一样，是 doc2query 式的索引扩充。
"""
from __future__ import annotations

import streamlit as st
from shared import (
    api_key_sidebar, backend_name, crisis_sidebar, footer, get_cards,
    get_retriever, page_setup,
)

page_setup("躯体化科普", "🩺")
api_key_sidebar()
crisis_sidebar()

soma = [c for c in get_cards() if c.type == "somatic"]

st.title("🩺 身体不舒服：这是什么，我能做什么")
st.markdown(
    "抑郁与焦虑常伴随真实的身体反应。这些不是「想出来的」，"
    "但也**不能**因此跳过必要的医学检查。每张卡都标了「什么情况必须就医」。"
)
st.warning(
    "本页内容整理自 Beyond Blue、Healthdirect Australia 与 NPS MedicineWise 的公开材料，"
    "仅供参考，不能替代医生的判断。症状剧烈或首次出现时请先就医排除躯体疾病。",
    icon="⚕️",
)

# --------------------------------------------------------------------------- #
# 搜索
# --------------------------------------------------------------------------- #
q = st.text_input(
    "他现在哪里不舒服？用他的原话就行",
    placeholder="比如：胸口发闷心跳好快 / 起来就眼前发黑 / 一直坐不住 / 脑子像糊了",
    key="soma_q",
)

EXAMPLES = ["胸口发闷心跳好快", "一直坐不住像有电", "脑子转不动记不住事", "喉咙像堵着东西", "什么都感觉不到"]
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True, key=f"ex_{ex}"):
        st.session_state.soma_q = ex
        st.rerun()

card = None
if q:
    retriever = get_retriever(backend_name())
    hits = [h for h in retriever.search(q, k=8) if h.card.type == "somatic"]
    if hits:
        labels = [f"{h.card.symptom}" for h in hits[:5]]
        pick = st.radio("最接近的是哪个？", labels, horizontal=False, key="soma_pick")
        card = next(h.card for h in hits if h.card.symptom == pick)
    else:
        st.info(
            "没有匹配到症状卡。可以换他的原话再试一次；"
            "如果症状剧烈、突然出现或者你拿不准，**先看医生排除身体原因**，这永远是对的顺序。"
        )

if card is None:
    with st.expander(f"或者直接翻全部 {len(soma)} 张症状卡", expanded=not q):
        names = [c.symptom for c in soma]
        pick = st.selectbox("全部症状", names, key="soma_all")
        card = next(c for c in soma if c.symptom == pick)

# --------------------------------------------------------------------------- #
# 详情
# --------------------------------------------------------------------------- #
if card:
    st.divider()
    st.header(card.symptom)
    if card.aliases:
        st.caption("也可能这样描述：" + " · ".join(card.aliases))

    st.markdown("### 这是什么")
    st.info(card.what_it_is)

    st.markdown("### 你可以做什么")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### 🏠 你在他身边时")
        for x in card.in_person:
            st.markdown(f"- {x}")
    with right:
        st.markdown("#### 📱 异地 / 只能线上时")
        for x in card.remote:
            st.markdown(f"- {x}")

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### ✅ 可以说")
        for x in card.say:
            st.markdown(f"> {x}")
    with b:
        st.markdown("#### ❌ 不要说")
        for x in card.avoid_saying:
            st.markdown(f"- {x}")

    st.error(
        "#### 🚑 这些情况必须就医或走危机流程\n"
        + "\n".join(f"- {x}" for x in card.seek_help_if)
    )

    src = card.source
    tier = {"clinical_guideline": "临床指南／权威机构",
            "practitioner": "从业者经验",
            "lived_experience": "亲历经验"}.get(card.evidence_tier, card.evidence_tier)
    line = f"证据等级：{tier} · 来源：{src.authority or src.platform or src.author}"
    if src.url:
        line += f" · [原文]({src.url})"
    st.caption(line)
    if card.needs_review:
        st.caption("⚠️ 这张卡尚未逐条核对原始出处，仅供参考。")

footer()
