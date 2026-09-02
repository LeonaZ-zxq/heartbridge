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


def _retrieve_for_generation(retriever, query: str, k: int, type_filter):
    """给生成器用的检索——**排除危机卡**。

    为什么必须排除，而不只是让 prompt 组装能处理它：
    整个安全架构的立足点是「危机内容确定性交付、永不生成」。
    危机卡一旦进了 prompt，模型就能改写它——包括「什么时候必须立刻叫救护车」
    这类升级条件。那正是这个架构明令禁止的事。

    这个泄漏是真实发生过的：输入「他说他撑不下去了」判为 ELEVATED（不是 CRISIS），
    走正常路径，而正常路径的检索不带类型过滤，于是 crisis_001 被检索到并喂进了生成器
    （「撑不下去了」正好在它的 aliases 里）。是测试先撞上了 prompt 组装的崩溃，
    才让人看见底下这个设计泄漏——崩溃只是表象。

    多取几条再过滤：否则混进来的危机卡会白白占掉名额，让生成器少看几张能用的卡。
    """
    raw = retriever.search(query, k=k + 3, type_filter=type_filter)
    return [h for h in raw if h.card.type != "crisis"][:k]


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
        # 短路仍然成立：**不调用 LLM**。确定性是这条路径的核心性质，
        # 已经是 CI 的发布门禁，不能为了「更贴切」把它换掉。
        #
        # 但「不生成」不等于「不具体」。在此之前这里连检索都没做，
        # 于是所有危机情境共用同一段模板——最需要具体话术的时刻，
        # 反而是唯一没有知识支撑的分支。
        # 现在检索人工撰写、有临床来源的危机卡：**话术来自卡片，不来自模型**。
        crisis_hits = retriever.search(query_text, k=2, type_filter="crisis")
        return Advice(
            risk=risk,
            turns=turns,
            hits=crisis_hits,
            crisis_response=render_crisis_response(risk.rationale),
        )

    # ---- 2. 路由 + 检索 ----
    type_filter = "somatic" if _looks_somatic(query_text) else None
    hits = _retrieve_for_generation(retriever, query_text, cfg.top_k, type_filter)
    # 躯体化检索兜底：过滤后没结果就放开限制，宁可给相关的沟通卡也别给空
    if not hits and type_filter:
        hits = _retrieve_for_generation(retriever, query_text, cfg.top_k, None)

    # ---- 3. 生成 ----
    options = generate_options(llm, situation, turns, hits, profile)
    return Advice(risk=risk, turns=turns, hits=hits, options=options)
