"""ReAct 循环 + 工具调用，外面套一圈代码强制的安全边界。

━━━ 这个模块真正的论点 ━━━

会写 ReAct 循环的人很多。这个实现想说明的是另一件事：

    **安全约束不能靠「在 prompt 里告诉模型别跳过」来保证。**

现有管道的危机短路是确定性的：assess() 命中就返回审查过的模板，
不检索、不生成。把这条管道换成 Agent，最容易丢掉的就是这个保证——
因为 Agent 的全部卖点就是「让模型自己决定下一步做什么」，
而一旦下一步由模型决定，「先查危机」就变成了一句建议。

所以这里的 check_crisis **不是一个工具**，是**循环不变量**：
它在模型拿到第一个回合之前就已经跑完了，命中危机直接返回，
模型根本不会被调用。Agent 无权跳过它，因为 Agent 还没开始。

这和 generator 里的 card_id 校验、anchor 校验、invented_facts 校验
是同一条原则，在这个项目里已经重复三次了：
**把一个语义要求，翻译成一个代码能执行的检查。**

三条硬边界，每条都有测试锁死：
1. 危机检查前置，模型无权跳过（crisis 输入下模型一次都不会被调用）
2. 最大步数 + 单工具调用上限，超了优雅终止而不是无限转圈
3. 工具抛异常时降级回退到现有的 RAG 直通路径，而不是整个崩掉
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Sequence

from core.config import CONFIG, Config
from core.engine.generator import generate_options
from core.engine.pipeline import Advice, _retrieve_for_generation, advise
from core.knowledge.retrieval import Hit, Retriever
from core.profile.models import PartnerProfile
from core.safety.detector import assess
from core.safety.templates import render_crisis_response
from core.utils.llm import LLMError, LLMProvider, complete_json
from core.utils.text import Turn, parse_transcript


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
@dataclass
class Tool:
    """一个可被模型调用的工具。

    `max_calls` 是**每次会话**的调用预算，不是全局限流。
    没有它，一个把 search_knowledge 当成搜索引擎反复试词的模型
    能在一次请求里打穿整天的额度——而且每次都「看起来在努力」。
    """

    name: str
    description: str
    run: Callable[[dict], str]
    max_calls: int = 3


def build_tools(
    retriever: Retriever,
    profile: PartnerProfile | None,
    collected: list[Hit],
    cfg: Config | None = None,
) -> dict[str, Tool]:
    """造出这次会话可用的工具集。

    `collected` 是检索结果的累加器：Agent 可能分几次用不同措辞检索，
    最后交给生成器的是这些轮次的**并集**（按 card_id 去重）。
    这正是 Agent 相对直通管道的实际价值——一次检索解决不了的复合问题，
    可以拆成几次。
    """
    cfg = cfg or CONFIG
    seen = {h.id for h in collected}

    def _search(args: dict) -> str:
        q = str(args.get("query", "")).strip()
        if not q:
            return "错误：query 不能为空"
        # 必须走 _retrieve_for_generation，**不能**直接调 retriever.search。
        #
        # 这不是风格问题，是这个 Agent 实打实引入过的一个安全回归：
        # 第一版直接调了 retriever.search，于是危机卡可以被检索进生成 prompt，
        # 而整个安全架构的立足点是「危机内容确定性交付、永不生成」。
        # pipeline 里早就修过同一个泄漏（见那个函数的 docstring），
        # Agent 绕过上层直接调底层 API，就把它原样放了回来。
        #
        # 教训：**安全过滤必须长在所有调用方都绕不开的那一层**。
        # 这里宁可跨模块引一个带下划线的私有函数，也不复制一份过滤逻辑——
        # 复制出来的那份迟早和本体漂移，而漂移的方向永远是漏掉。
        hits = _retrieve_for_generation(retriever, q, cfg.top_k, None)
        if not hits:
            return "没有检索到任何卡片。换个说法再试，或者直接作答。"
        lines = []
        for h in hits:
            if h.id not in seen:
                seen.add(h.id)
                collected.append(h)
            c = h.card
            label = getattr(c, "technique_name", None) or getattr(c, "symptom", "")
            lines.append(f"- {c.id}｜{label}｜适用：{getattr(c, 'scenario', '')[:40]}")
        return "检索到：\n" + "\n".join(lines)

    def _check_crisis(args: dict) -> str:
        """模型可以对**新发现的**文本片段再查一次危机。

        循环开始前整段输入已经查过了（见 run_agent 的 gate）。
        这个工具是给「profile 里读到一句以前没查过的话」这类情况用的，
        不是前置检查的替代品。
        """
        text = str(args.get("text", "")).strip()
        if not text:
            return "错误：text 不能为空"
        r = assess(text, llm=None)      # 工具内只走规则层：确定性、免费、微秒级
        return f"风险等级={r.level.name}；依据={r.rationale or '无'}"

    def _get_profile(args: dict) -> str:
        if profile is None:
            return "没有提供伴侣档案。"
        return profile.to_prompt_block()

    return {
        t.name: t
        for t in (
            Tool("search_knowledge",
                 "按一句话检索知识卡片。参数：{\"query\": \"...\"}", _search, max_calls=4),
            Tool("check_crisis",
                 "对一段文本做规则层危机评估。参数：{\"text\": \"...\"}", _check_crisis, max_calls=3),
            Tool("get_partner_profile",
                 "读取伴侣档案（偏好、雷区、是否异地）。参数：{}", _get_profile, max_calls=1),
        )
    }


# --------------------------------------------------------------------------- #
# 轨迹
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AgentStep:
    thought: str
    action: str
    action_input: dict
    observation: str
    ok: bool = True


@dataclass
class AgentTrace:
    """Agent 这一轮到底做了什么，以及**为什么停下**。

    `stopped` 不是日志，是契约的一部分：
    调用方（和测试）据此判断这次回答是正常产出，还是撞了某条边界。
    静默地在两者之间切换，正是这个项目一路在修的那类 bug。
    """

    steps: list[AgentStep] = field(default_factory=list)
    stopped: str = "answered"   # answered | crisis | max_steps | tool_budget | degraded
    note: str = ""
    gate_ran: bool = False      # 危机前置检查是否真的执行过

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.action not in ("finish", ""))


_SYSTEM = """你在帮助一位抑郁症患者的伴侣。你可以调用工具收集信息，然后作答。

每一步只输出一个 JSON，不要任何其他文字：
{"thought":"你在想什么（一句话）","action":"工具名或 finish","action_input":{...}}

可用工具：
{tools}

规则：
- 想清楚还缺什么信息再调工具；信息够了就 action="finish"，action_input={{}}。
- 同一个工具不要用几乎一样的参数反复调，那不会得到新信息。
- 最多 {max_steps} 步。步数用完会强制结束，所以别把步数浪费在试探上。"""


def _render_scratchpad(steps: Sequence[AgentStep]) -> str:
    if not steps:
        return "（还没有任何观察）"
    out = []
    for i, s in enumerate(steps, 1):
        out.append(f"[{i}] 想法：{s.thought}\n    调用：{s.action}({json.dumps(s.action_input, ensure_ascii=False)})\n    观察：{s.observation}")
    return "\n".join(out)


def run_agent(
    llm: LLMProvider,
    raw: str,
    retriever: Retriever,
    profile: PartnerProfile | None = None,
    cfg: Config | None = None,
    *,
    situation: str = "",
    max_steps: int = 6,
) -> Advice:
    """跑一轮 ReAct，返回和直通管道**同一个** Advice 类型。

    返回类型刻意不变：UI、CLI、测试全都不需要知道这次走的是 Agent
    还是直通路径。Agent 是一条可替换的实现，不是一个新产品。
    轨迹挂在 `advice.trace` 上，想看的人才看。
    """
    cfg = cfg or CONFIG
    turns = parse_transcript(raw, llm=None)
    body = raw if not turns else "\n".join(t.text for t in turns)
    query_text = (situation + "\n" + body).strip()

    trace = AgentTrace()

    # ---- 不变量：危机检查在模型拿到任何一个回合之前跑完 ----
    #
    # 注意这不是「第一个工具」。它在循环外面，Agent 没有任何办法
    # 影响它是否执行。命中危机时模型**一次都不会被调用**——
    # 这是一条可以写进测试的断言，而「我们在 prompt 里要求它先查」不是。
    risk = assess(query_text, llm=llm, cfg=cfg)
    trace.gate_ran = True
    if risk.is_crisis:
        trace.stopped = "crisis"
        trace.note = "危机命中，短路返回审查过的模板，未进入 Agent 循环"
        adv = Advice(risk=risk, turns=turns,
                     crisis_response=render_crisis_response(risk.rationale))
        adv.trace = trace               # type: ignore[attr-defined]
        return adv

    collected: list[Hit] = []
    tools = build_tools(retriever, profile, collected, cfg)
    used: dict[str, int] = {}

    tool_desc = "\n".join(f"- {t.name}：{t.description}" for t in tools.values())
    system = _SYSTEM.replace("{tools}", tool_desc).replace("{max_steps}", str(max_steps))

    def _degrade(note: str) -> Advice:
        """回退到现有的直通 RAG 路径。

        降级要**产出一个可用的回答**，不是一个错误页。
        Agent 是优化，不是依赖；它坏了，用户该拿到的还是原来那套结果。

        ━━━ 为什么这里要再包一层 try ━━━

        降级路径和 Agent **共用同一个 retriever**。所以「检索器坏了」
        这一类故障会让回退本身也抛出同一个异常——
        第一版就是这么写的，测试立刻抓到：降级到一条同样会失败的路径，
        不叫降级，只是把崩溃推迟了一个栈帧。

        真正坏到没有回答可给的时候，诚实的输出是一个
        **没有选项、但说明了原因的 Advice**，不是一个异常。
        """
        trace.stopped = "degraded"
        trace.note = note
        try:
            adv = advise(raw, retriever, llm, profile=profile, cfg=cfg, situation=situation)
        except Exception as exc:                      # noqa: BLE001
            trace.note = f"{note}；直通路径同样失败：{type(exc).__name__}: {exc}"
            adv = Advice(risk=risk, turns=turns)
        adv.trace = trace               # type: ignore[attr-defined]
        return adv

    for _ in range(max_steps):
        prompt = (f"## 用户给的情况\n{query_text}\n\n"
                  f"## 你已经做过的事\n{_render_scratchpad(trace.steps)}\n\n"
                  f"下一步？")
        try:
            raw_step = complete_json(llm, system, prompt, temperature=0.2)
        except LLMError as exc:
            return _degrade(f"模型调用失败：{exc}")

        if not isinstance(raw_step, dict):
            trace.steps.append(AgentStep("", "", {}, f"返回不是对象：{type(raw_step).__name__}", ok=False))
            continue

        action = str(raw_step.get("action", "")).strip()
        thought = str(raw_step.get("thought", "")).strip()
        args = raw_step.get("action_input") or {}
        if not isinstance(args, dict):
            args = {}

        if action in ("finish", "final", ""):
            trace.steps.append(AgentStep(thought, "finish", {}, "准备作答"))
            break

        tool = tools.get(action)
        if tool is None:
            trace.steps.append(AgentStep(
                thought, action, args,
                f"没有叫 {action} 的工具。可用：{'、'.join(tools)}", ok=False))
            continue

        used[action] = used.get(action, 0) + 1
        if used[action] > tool.max_calls:
            # 预算用完不是错误，是边界。告诉模型，让它自己收敛到作答。
            trace.steps.append(AgentStep(
                thought, action, args,
                f"{action} 的调用次数已用完（上限 {tool.max_calls}）。请直接 finish 作答。",
                ok=False))
            if trace.tool_calls >= sum(t.max_calls for t in tools.values()):
                trace.stopped = "tool_budget"
                break
            continue

        try:
            obs = tool.run(args)
            trace.steps.append(AgentStep(thought, action, args, obs))
        except Exception as exc:                      # noqa: BLE001
            # 单个工具炸了先让模型看到错误、自己换路；
            # 只有在什么都没收集到的情况下才整体降级。
            trace.steps.append(AgentStep(
                thought, action, args, f"工具执行失败：{type(exc).__name__}: {exc}", ok=False))
            if not collected:
                return _degrade(f"工具 {action} 失败且没有任何检索结果：{exc}")
    else:
        trace.stopped = "max_steps"

    if not collected:
        return _degrade("Agent 没有收集到任何卡片")

    options = generate_options(llm, situation, turns, collected, profile)
    if not options:
        return _degrade("Agent 收集到了卡片但生成阶段没有产出可用选项")

    adv = Advice(risk=risk, turns=turns, hits=collected, options=options)
    adv.trace = trace                   # type: ignore[attr-defined]
    return adv
