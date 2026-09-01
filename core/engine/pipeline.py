"""端到端编排：一段聊天记录进来，一组带解释的回复选项出去。

流程（安全分支优先，这个顺序本身就是设计）：

    原始文本
      └─ 解析成对话轮次
          └─ 危机评估 ──── 命中 ──→ 危机模板（短路，不检索、不生成）
              └─ 未命中
                  └─ 选择卡片类型（躯体化 vs 沟通）
                      └─ RAG 检索 top-k
                          └─ 注入 profile + 语气样本 → 生成 2-3 个选项 + why

**为什么危机分支必须在检索和生成之前**：
把危机判断放在生成之后（比如让模型自己决定要不要提热线），
等于把安全保证托付给一个非确定性组件。短路设计保证了
「有危机信号时用户看到的一定是审查过的模板」，与模型状态无关。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.config import CONFIG, Config
from core.knowledge.retrieval import Hit, Retriever
from core.profile.models import PartnerProfile
from core.safety.detector import RiskAssessment, RiskLevel, assess
from core.safety.templates import DISCLAIMER, render_crisis_response
from core.engine.generator import ReplyOption, generate_options
from core.utils.llm import LLMProvider
from core.utils.text import Turn, parse_transcript

# 提示躯体症状的词。命中则把检索限定在 somatic 卡上——
# 一个廉价的路由器：不值得为它调一次 LLM。
_SOMATIC_HINTS = (
    "胸闷", "胸口", "心跳", "心悸", "喘", "呼吸", "手麻", "发麻", "头晕", "抖", "颤",
    "冷汗", "出汗", "恶心", "想吐", "吃不下", "没胃口", "胃", "累", "起不来", "没力气",
    "头疼", "头痛", "背疼", "浑身疼", "不真实", "发呆", "麻木", "失眠", "睡不着", "早醒",
)


@dataclass
class Advice:
    risk: RiskAssessment
    turns: list[Turn] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    options: list[ReplyOption] = field(default_factory=list)
    crisis_response: str | None = None
    disclaimer: str = DISCLAIMER

    @property
    def is_crisis(self) -> bool:
        return self.crisis_response is not None

    def render(self) -> str:
        if self.crisis_response:
            return self.crisis_response
        lines = [f"风险等级：{self.risk.level.label}"]
        if self.risk.level == RiskLevel.ELEVATED:
            lines.append("⚠️ 这段话里有明显的痛苦信号，回复之后建议继续留意。")
        lines.append("")
        for i, o in enumerate(self.options, 1):
            tag = f"（{o.style}）" if o.style else ""
            lines.append(f"**选项 {i}{tag}**")
            lines.append(f"> {o.text}")
            lines.append(f"_为什么：{o.why}_")
            mark = "" if o.grounded else "  ⚠️ 引用了未检索到的卡片"
            lines.append(f"_依据：{o.card_id}{mark}_\n")
        if self.hits:
            lines.append("参考卡片：" + "、".join(h.id for h in self.hits))
        lines.append(f"\n_{self.disclaimer}_")
        return "\n".join(lines)


def _looks_somatic(text: str) -> bool:
    return any(h in text for h in _SOMATIC_HINTS)


def advise(
    raw: str,
    retriever: Retriever,
    llm: LLMProvider,
    profile: PartnerProfile | None = None,
    cfg: Config | None = None,
    *,
    situation: str = "",
) -> Advice:
    cfg = cfg or CONFIG
    turns = parse_transcript(raw, llm=None)  # 解析阶段不需要 LLM，走确定性路径
    body = raw if not turns else "\n".join(t.text for t in turns)
    query_text = (situation + "\n" + body).strip()

    # ---- 1. 安全优先 ----
    risk = assess(query_text, llm=llm, cfg=cfg)
    if risk.is_crisis:
        return Advice(
            risk=risk,
            turns=turns,
            crisis_response=render_crisis_response(risk.rationale),
        )

    # ---- 2. 路由 + 检索 ----
    type_filter = "somatic" if _looks_somatic(query_text) else None
    hits = retriever.search(query_text, k=cfg.top_k, type_filter=type_filter)
    # 躯体化检索兜底：过滤后没结果就放开限制，宁可给相关的沟通卡也别给空
    if not hits and type_filter:
        hits = retriever.search(query_text, k=cfg.top_k)

    # ---- 3. 生成 ----
    options = generate_options(llm, situation, turns, hits, profile)
    return Advice(risk=risk, turns=turns, hits=hits, options=options)
