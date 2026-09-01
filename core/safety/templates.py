"""危机应对模板与求助资源。

为什么危机回复是**硬编码模板**而不是 LLM 生成（面试必问）：

在最高风险的那一刻，你需要的是**可预测、经过审查、每次都一样**的输出。
生成模型在这里的每一个优点都变成缺点：
    · 多样性 → 意味着某一次可能说出有害的话
    · 流畅 → 意味着有害内容看起来也很可信
    · 不可复现 → 意味着出事后无法追责、无法回归测试
模板是可以被人类逐字审查、被单元测试逐条断言的。
**在安全关键路径上，可预测性的价值高于个性化。**

LLM 在危机分支里只保留一个很小的作用：把用户的具体情境复述一句，
让消息不显得像自动回复。核心内容全部来自模板。
"""
from __future__ import annotations

from dataclasses import dataclass

DISCLAIMER = (
    "HeartBridge 是沟通技巧助手，不是心理治疗，也不做诊断。"
    "遇到危险请立即联系专业服务或急救。"
)


@dataclass(frozen=True)
class Resource:
    region: str
    name: str
    contact: str
    note: str = ""

    def line(self) -> str:
        tail = f"（{self.note}）" if self.note else ""
        return f"{self.name}：{self.contact}{tail}"


# 来源：Healthdirect — Mental health crisis support
# https://www.healthdirect.gov.au/mental-health-crisis-support
RESOURCES_AU: list[Resource] = [
    Resource("AU", "紧急救助 Emergency", "000", "有立即危险时"),
    Resource("AU", "Lifeline", "13 11 14", "24/7 危机支持"),
    Resource("AU", "Beyond Blue", "1300 22 4636", "24/7 抑郁与焦虑支持"),
    Resource("AU", "Suicide Call Back Service", "1300 659 467", "自杀相关咨询"),
]

RESOURCES_CN: list[Resource] = [
    Resource("CN", "紧急救助", "120", "有立即危险时"),
    Resource("CN", "全国心理援助热线", "12356", "24 小时"),
]

# 危机当下该做什么。参考 Healthdirect 的 crisis support 指引：
# 保持冷静、多听少说、不要做会让对方更激动的事、必要时移除危险物品。
CRISIS_STEPS: list[str] = [
    "**先留在这段对话里。** 不要挂断、不要下线。你的在场本身就是保护因素。",
    "**直接问。** 「你是在想结束自己的生命吗？」直接询问不会增加风险，回避才会。",
    "**多听少说。** 语速放慢、句子变短、不评判、不辩论、不讲道理。",
    "**问他现在安不安全。** 身边有没有危险物品，能不能先离开那个环境。",
    "**不要独自承担。** 联系他身边能立刻到场的人，或拨打下面的热线。",
    "**不要承诺保密。** 你可以承诺陪着他，但不能承诺不告诉任何人。",
]

CRISIS_SAY: list[str] = [
    "我在这儿，我不会挂电话。",
    "我想直接问你一句：你有在想伤害自己吗？",
    "不管你的答案是什么，我都不走。",
    "我们一起打个电话好不好，我陪着你。",
]

CRISIS_AVOID: list[str] = [
    "别说傻话 / 你想太多了（会关掉他唯一一次开口的窗）",
    "你要为我想想 / 你对得起我吗（施加内疚，会增加风险）",
    "长篇道理和劝说（此刻他处理不了长句）",
    "「我保证不告诉任何人」（做不到的承诺）",
]


def render_crisis_response(
    context: str = "",
    regions: tuple[str, ...] = ("AU", "CN"),
) -> str:
    """渲染危机回复。纯字符串拼装，无模型参与，输出完全可预测。"""
    blocks: list[str] = ["🚨 **这条消息里有需要立刻认真对待的信号。**\n"]
    if context:
        blocks.append(f"> 触发原因：{context}\n")

    blocks.append("### 现在就做这几件事")
    blocks += [f"{i}. {s}" for i, s in enumerate(CRISIS_STEPS, 1)]

    blocks.append("\n### 可以这样说")
    blocks += [f"- 「{s}」" for s in CRISIS_SAY]

    blocks.append("\n### 千万不要说")
    blocks += [f"- {s}" for s in CRISIS_AVOID]

    blocks.append("\n### 求助资源")
    pool = ([*RESOURCES_AU] if "AU" in regions else []) + ([*RESOURCES_CN] if "CN" in regions else [])
    for res in pool:
        blocks.append(f"- [{res.region}] {res.line()}")

    blocks.append(
        "\n⚠️ AI 不能替代专业干预。如果他有立即的危险，"
        "请现在就拨打急救电话，不要等。\n"
    )
    blocks.append(f"_{DISCLAIMER}_")
    return "\n".join(blocks)
