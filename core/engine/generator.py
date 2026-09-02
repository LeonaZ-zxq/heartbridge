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

import re
from dataclasses import dataclass

from core.knowledge.retrieval import Hit
from core.profile.models import PartnerProfile
from core.utils.llm import LLMError, LLMProvider, complete_json
from core.utils.text import Turn

class _Failed(list):
    """空的选项列表 + 失败原因。

    刻意做成 list 的子类：所有 `if not options` / `for o in options` 的调用点
    都不需要改，但想知道原因的地方可以读 `.error`。
    在「不破坏现有契约」和「不丢失信息」之间，这是成本最低的一条路。
    """

    def __init__(self, items, error: str = ""):
        super().__init__(items)
        self.error = error


SYSTEM_PROMPT = """你在帮助一位抑郁症患者的伴侣，想出「此刻可以怎么回复」。

硬性要求：
1. 生成 2-3 个**风格不同**的回复选项（例如：偏共情 / 偏陪伴行动 / 偏轻）。
2. 每个选项必须给出 why：解释这么说为什么有效，讲机制，不要复述内容。
3. 每个选项必须标注 card_id，来自下面提供的知识卡片。不许引用没给你的卡。
4. **每个选项必须给出 anchor：从他刚才说的话里原样摘一个词或短句**，
   表示这条回复是在回应这个具体的东西。anchor 必须是他消息里的**原文**，不能改写。
5. **禁止写放到任何情境都成立的句子。**「我在」「我陪着你」「别难过」「会好起来的」
   这类话单独成句一律不合格——它们没有回应任何具体的东西，读起来像模板。
   每条回复里必须出现他提到的**具体的人、事、时间或身体感受**。
   反例（不合格）：「我在呢，有什么都跟我说」
   正例（合格）：「面试官只见了你三十分钟，我见了你两年半」
6. 回复要像真人发微信：口语、短、可以直接复制粘贴。不要书面语，不要排比句。
7. 不要诊断，不要给医疗建议，不要保证「一定会好」。
8. 如果提供了「我的语气样本」，模仿那个语气和用词习惯。
9. 如果 profile 标注了雷区，绝对不要碰。
10. 如果是异地，优先给远程可执行的做法，不要写"抱抱他"这种做不到的事。

只输出 JSON，不要任何其他文字：
{"options":[{"text":"要发出去的话","why":"为什么有效（讲机制，1-2句）","anchor":"从他消息里原样摘的词","card_id":"comm_001","style":"共情"}]}"""



_PUNCT = re.compile(r"[\s，。！？、,.!?；;：:~…\-—「」『』\"\'（）()【】\[\]]+")


def _norm(t: str) -> str:
    """去掉空白和标点再比对。

    模型摘原文时经常带上或漏掉标点（「今天面试又挂了」vs「今天面试又挂了，」），
    逐字节比对会把这些全判成不合格。归一化之后比的是**内容**，不是排版。
    """
    return _PUNCT.sub("", t)


def _check_anchor(anchor: str, haystack: str) -> bool:
    """anchor 必须是他原话里的一段。

    ━━━ 为什么这一条要由代码来判，而不是写在 prompt 里 ━━━

    「回复要具体、不要说套话」这种要求写进 prompt 是**没有约束力**的：
    模型会真诚地答应，然后继续生成「我在，有什么都跟我说」。
    因为「具体」对模型来说不是一个可验证的目标。

    所以把它换成一个**可验证**的代理目标：
        你必须指出这条回复在回应他说的哪一句原话，而且那句话我会去他的
        消息里查。查不到，这条就不算数。

    这不能保证回复一定好，但它能**结构性地挡掉**那一类
    「放到任何情境都成立」的模板句——因为一条通用回复根本找不到
    可以锚定的具体原文。这和 card_id 校验是同一个思路：
    **把一个语义要求，翻译成一个代码能执行的检查。**
    """
    a = _norm(anchor)
    if len(a) < 2:          # 太短（比如摘了个「我」）等于没锚定
        return False
    return a in _norm(haystack)


@dataclass
class ReplyOption:
    text: str
    why: str
    card_id: str
    style: str = ""
    grounded: bool = True   # card_id 是否真的在检索结果里
    anchor: str = ""        # 这条回复在回应他说的哪一句原话
    specific: bool = True   # anchor 是否真的出现在他的消息里


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
        elif c.type == "somatic":
            cards_block.append(
                f"### {c.id} | 躯体化：{c.symptom}\n"
                f"是什么：{c.what_it_is}\n"
                f"在他身边时：{'；'.join(c.in_person)}\n"
                f"异地/线上时：{'；'.join(c.remote)}\n"
                f"可以说：{'；'.join(c.say)}\n"
                f"不要说：{'；'.join(c.avoid_saying)}\n"
                f"需要就医的情况：{'；'.join(c.seek_help_if)}"
            )
        else:
            # 原来这里是 `else:` 直接当躯体化卡处理——**开放的枚举配封闭的分支**。
            # 加进第三种卡片类型时，这个假设就悄悄不成立了，
            # 表现为 `'CrisisCard' object has no attribute 'symptom'`。
            #
            # 这里选择抛错而不是跳过：跳过会让这张卡仍然留在「允许引用」的白名单里，
            # 而 prompt 里却没有它的内容——模型于是可以引用一张自己没看过的卡，
            # 引用校验也拦不住。宁可响亮地失败。
            # 危机卡由 pipeline._retrieve_for_generation 挡在门外，
            # 走到这里说明是编程错误，不是用户输入问题。
            raise ValueError(
                f"生成路径收到了不该出现的卡片类型 {c.type!r}（{c.id}）。"
                "危机卡不得进入生成 prompt；新增卡片类型时必须同时更新这里。"
            )
    parts.append("## 可用的知识卡片（只能引用这些的 card_id）\n" + "\n\n".join(cards_block))
    return "\n\n".join(parts)


def _validate(raw: object, allowed_ids: set[str], said: str = "") -> list[ReplyOption]:
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
        anchor = str(item.get("anchor", "")).strip()
        options.append(
            ReplyOption(
                text=text,
                why=why,
                card_id=cid,
                style=str(item.get("style", "")).strip(),
                anchor=anchor,
                # anchor 查不到 = 这条回复没有真的在回应他说的话
                specific=_check_anchor(anchor, said) if said else True,
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
    drop_generic: bool = True,
) -> list[ReplyOption]:
    allowed = {h.id for h in hits}
    said = " ".join([situation, *(t.text for t in turns)])
    try:
        raw = complete_json(
            llm, SYSTEM_PROMPT, build_user_prompt(situation, turns, hits, profile),
            temperature=0.7,  # 要多样性：三个选项应该真的不一样
        )
    except LLMError as exc:
        # 以前这里是 `return []`，于是「模型挂了」和「模型答了但全被校验刷掉」
        # 产出一模一样的空列表，UI 只能写「**可能**是模型调用失败」。
        # 这和摄取管道里那个「转写成功却零产出」是同一类失败：
        # 静默降级把两种完全不同的原因压成了同一个观测结果。
        # 把异常挂在返回值上，让上层能如实说出发生了什么。
        return _Failed([], str(exc))
    options = _validate(raw, allowed, said)
    if drop_generic:
        # 同样的兜底哲学：全部不合格时宁可带标记给出，也不给空结果。
        options = [o for o in options if o.specific] or options
    if drop_ungrounded:
        kept = [o for o in options if o.grounded]
        # 但如果全部未接地就都留下并标记——给用户空结果比给带警告的结果更糟
        return kept or options
    return options
