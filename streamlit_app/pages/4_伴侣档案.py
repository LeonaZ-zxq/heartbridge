"""伴侣档案编辑页。

为什么这一页必须存在：档案正是让回复**具体**而不是通用的那个机制。
在它之前，访问者打开 demo 看到的是「小鱼」的虚构数据，
没有任何办法用自己的情况试一次——等于这个卖点一次都没被演示到。
这和「没有 key 就只能看写死示例」是同一类产品错误。

隐私实现见 shared._PROFILE_SS 上方的注释：只在会话内存里，永不落盘。
"""
from __future__ import annotations

import json

import streamlit as st
from shared import (
    active_profile, api_key_sidebar, clear_profile, crisis_sidebar, custom_profile,
    demo_profile, footer, page_setup, save_profile,
)

from core.profile.models import PartnerProfile

page_setup("伴侣档案", "📇")
api_key_sidebar()
crisis_sidebar()

st.title("📇 伴侣档案")
st.caption(
    "档案会被压成一段文本注入 prompt，让建议贴着**你们的**情况，而不是泛泛而谈。"
)

st.info(
    "**这份档案不会离开这次会话。** 它只存在服务器内存里的当前浏览器会话中，"
    "不写数据库、不写磁盘、刷新页面即消失。想留着下次用，请用下面的「导出」存成文件。\n\n"
    "这不是省事的做法，是刻意的：这里记的是另一个人的精神健康信息，"
    "而这是一个公开实例——代码里根本没有写盘那条路径。",
    icon="🔒",
)

cur = custom_profile()
base = cur or demo_profile()

if cur is None:
    st.warning(
        "你还没有填过档案，下面预填的是**虚构的演示数据**（小鱼）。"
        "改成你自己的情况再保存，或者直接清空。",
        icon="🧪",
    )
else:
    st.success("正在使用你自己填的档案。", icon="✅")


def _lines(label: str, values: list[str], help_: str = "") -> list[str]:
    """多值字段统一用「一行一条」，比逗号分隔更不容易出错。"""
    txt = st.text_area(label, value="\n".join(values), height=100, help=help_)
    return [x.strip() for x in txt.splitlines() if x.strip()]


with st.form("profile_form"):
    c1, c2 = st.columns(2)
    with c1:
        nickname = st.text_input("怎么称呼他/她", value=base.nickname)
        years = st.number_input(
            "在一起多久（年）", min_value=0.0, max_value=80.0, step=0.5,
            value=float(base.relationship_years or 0.0),
        )
        long_distance = st.checkbox(
            "异地", value=base.long_distance,
            help="勾上之后，建议会优先给远程可做的方案——很多在场做法异地根本用不了",
        )
    with c2:
        diagnosis = st.text_input(
            "诊断（可选）", value=base.diagnosis or "",
            help="不填也能用。填了会让建议更贴合，但系统不会因此给医疗判断",
        )
        medication = st.text_input("用药（可选）", value=base.medication or "")
        treat_map = {"没填": None, "是": True, "否": False}
        treat_label = st.selectbox(
            "正在接受治疗", list(treat_map),
            index=list(treat_map.values()).index(base.in_treatment),
        )

    st.divider()
    st.markdown("##### 这几项对回复质量影响最大")

    c3, c4 = st.columns(2)
    with c3:
        triggers = _lines("已知触发点", base.triggers, "什么情境下他/她会明显变差。一行一条")
        core_fears = _lines("核心焦虑", base.core_fears, "反复出现的那几句自我否定")
        landmines = _lines("雷区（绝对不要提）", base.landmines,
                           "这一项会作为硬约束进 prompt")
    with c4:
        what_helps = _lines("以前有效的", base.what_helps)
        what_backfires = _lines("以前起反效果的", base.what_backfires)
        pet_names = _lines("你们的称呼习惯", base.pet_names)

    voice = _lines(
        "你平时说话的样子（几句就够）", base.my_voice_samples,
        "这是 few-shot 语气样本：让生成的回复像**你**，而不像客服。"
        "写你真的会发出去的那种句子，越口语越好",
    )

    saved = st.form_submit_button("保存到本次会话", type="primary",
                                  use_container_width=True)

if saved:
    try:
        save_profile(PartnerProfile(
            profile_id="session",
            nickname=nickname or "他",
            relationship_years=years or None,
            long_distance=long_distance,
            diagnosis=diagnosis or None,
            medication=medication or None,
            in_treatment=treat_map[treat_label],
            triggers=triggers, core_fears=core_fears, landmines=landmines,
            what_helps=what_helps, what_backfires=what_backfires,
            my_voice_samples=voice, pet_names=pet_names,
        ))
        st.success("已保存到本次会话。回「情境求助」那一页就会用这一份。", icon="✅")
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"校验没通过：{exc}")

st.divider()

c5, c6, c7 = st.columns(3)
with c5:
    st.download_button(
        "⬇️ 导出为 JSON", use_container_width=True,
        data=json.dumps(active_profile().model_dump(mode="json"),
                        ensure_ascii=False, indent=2),
        file_name="heartbridge_profile.json", mime="application/json",
        help="存成文件自己保管。下次进来导入即可——服务器这边什么都不留",
    )
with c6:
    up = st.file_uploader("⬆️ 导入 JSON", type="json", label_visibility="collapsed")
    if up is not None:
        try:
            save_profile(PartnerProfile.model_validate(json.load(up)))
            st.success("已导入。", icon="✅")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"这个文件读不了：{exc}")
with c7:
    if st.button("🗑️ 清除档案", use_container_width=True,
                 help="清除后回到虚构的演示数据"):
        clear_profile()
        st.rerun()

st.divider()

# 透明度：直接把注入 prompt 的那段原文给使用者看。
# 「我们会用你的档案」是一句承诺，把原文摊开才是可核验的。
with st.expander("看看这份档案实际会变成什么（注入 prompt 的原文）"):
    st.caption("只有非空字段会进去——空字段既浪费 token 又稀释注意力。")
    st.code(active_profile().to_prompt_block(), language="text")

footer()
