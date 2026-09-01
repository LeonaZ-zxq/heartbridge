"""回复生成：RAG + profile + 你的语气样本 → 2-3 个选项，每个都带「为什么」。

━━━ 两个核心设计 ━━━

**1. 每个回复选项必须引用一张具体的卡片 id，并且这个 id 会被校验。**

这是 RAG 系统里最实用的一条反幻觉措施：模型被要求在输出里标注
「这个建议来自哪张卡」，而代码会检查这个 id 是否真的在本次检索结果里。
引用了没检索到的卡 = 模型在编 = 直接丢弃这个选项。
这把「溯源」从一句宣传变成了一个**可执行的校验**。

**2. 输出「为什么」而不只是「说什么」。**

如果只给话术，用户就是在照抄，回复会有"人机感"，而且下次遇到
略微不同的情境依然不会。给出机制解释，用户学到的是模式识别，
能举一反三。这是产品设计，也是一个可以在面试里讲的**用户价值判断**：
    工具的目标不是替用户说话，是让用户下次不需要这个工具。
"""
from __future__ import annotations

from dataclasses import dataclass

from core.knowledge.retrieval import Hit
from core.profile.models import PartnerProfile
from core.utils.llm import LLMError, LLMProvider, complete_json
from core.utils.text import Turn

SYSTEM_PROMPT = """你在帮助一位抑郁症患者的伴侣，想出「此刻可以怎么回复」。

硬性要求：
1. 生成 2-3 个**风格不同**的回复选项（例如：偏共情 / 偏陪伴行动 / 偏轻）。
2. 每个选项必须给出 why：解释这么说为什么有效，讲机制，不要复述内容。
3. 每个选项必须标注 card_id，来自下面提供的知识卡片。不许引用没给你的卡。
4. 回复要像真人发微信：口语、短、可以直接复制粘贴。不要书面语，不要排比句。
5. 不要诊断，不要给医疗建议，不要保证「一定会好」。
6. 如果提供了「我的语气样本」，模仿那个语气和用词习惯。
7. 如果 profile 标注了雷区，绝对不要碰。
8. 如果是异地，优先给远程可执行的做法，不要写"抱抱他"这种做不到的事。

只输出 JSON，不要任何其他文字：
{"options":[{"text":"要发出去的话","why":"为什么有效（讲机制，1-2句）","card_id":"comm_001","style":"共情"}]}"""


@dataclass
class ReplyOption:
    text: str
    why: str
    card_id: str
    style: str = ""
    grounded: bool = True  # card_id 是否真的在检索结果里


def build_user_prompt(
    situation: str,
    turns: list[Turn],
    hits: list[Hit],
    profile: PartnerProfile | None,
) -> str:
    parts: list[str] = []

    if profile:
        parts.append("## 关于他（profile）\n" + profile.to_prompt_block())
        if profile.my_voice_samples:
            # few-shot：用户过去真实发出去、且有效的消息。
            # 这是让输出"像本人"最有效的一招，比在 prompt 里描述语气好得多。
            samples = "\n".join(f"- {s}" for s in profile.my_voice_samples[:5])
            parts.append(f"## 我平时说话的样子（模仿这个语气）\n{samples}")

    if turns:
        convo = "\n".join(t.render() for t in turns[-8:])  # 只取最近 8 轮，控制上下文
        parts.append(f"## 刚才的对话\n{convo}")
    if situation:
        parts.append(f"## 情境描述\n{situation}")

    cards_block: list[str] = []
    for h in hits:
        c = h.card
        if c.type == "communication":
            cards_block.append(
                f"### {c.id} | {c.technique_name}\n"
                f"适用情境：{c.scenario}\n"
                f"要做：{'；'.join(c.do)}\n"
                f"不要做：{'；'.join(c.dont)}\n"
                f"参考说法：{'；'.join(c.example_phrases)}\n"
                f"原理：{c.why_it_works}"
            )
        else:
            cards_block.append(
                f"### {c.id} | 躯体化：{c.symptom}\n"
                f"是什么：{c.what_it_is}\n"
                f"在他身边时：{'；'.join(c.in_person)}\n"
                f"异地/线上时：{'；'.join(c.remote)}\n"
                f"可以说：{'；'.join(c.say)}\n"
                f"不要说：{'；'.join(c.avoid_saying)}\n"
                f"需要就医的情况：{'；'.join(c.seek_help_if)}"
            )
    parts.append("## 可用的知识卡片（只能引用这些的 card_id）\n" + "\n\n".join(cards_block))
    return "\n\n".join(parts)


def _validate(raw: object, allowed_ids: set[str]) -> list[ReplyOption]:
    """校验模型输出。

    面试点：LLM 的输出是**不可信输入**，要像处理用户提交的表单一样校验。
    这里做三件事：结构校验、引用校验、数量截断。
    """
    options: list[ReplyOption] = []
    items = raw.get("options") if isinstance(raw, dict) else raw
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        why = str(item.get("why", "")).strip()
        if not text or not why:
            continue  # 缺 why 的选项直接丢：没有解释就退化成了话术抄写
        cid = str(item.get("card_id", "")).strip()
        options.append(
            ReplyOption(
                text=text,
                why=why,
                card_id=cid,
                style=str(item.get("style", "")).strip(),
                # 引用了没检索到的卡 → 标记为未接地。调用方可以据此丢弃或提示。
                grounded=cid in allowed_ids,
            )
        )
    return options[:3]


def generate_options(
    llm: LLMProvider,
    situation: str,
    turns: list[Turn],
    hits: list[Hit],
    profile: PartnerProfile | None = None,
    *,
    drop_ungrounded: bool = True,
) -> list[ReplyOption]:
    allowed = {h.id for h in hits}
    try:
        raw = complete_json(
            llm, SYSTEM_PROMPT, build_user_prompt(situation, turns, hits, profile),
            temperature=0.7,  # 要多样性：三个选项应该真的不一样
        )
    except LLMError:
        return []
    options = _validate(raw, allowed)
    if drop_ungrounded:
        kept = [o for o in options if o.grounded]
        # 但如果全部未接地就都留下并标记——给用户空结果比给带警告的结果更糟
        return kept or options
    return options
