"""危机检测：两级架构（规则 → LLM 二次确认）。

━━━ 设计原则：这是一个召回优先、且不对称的系统 ━━━

普通分类任务追求 F1 平衡。这里不是。
漏报一次自杀信号的代价，和误报一次「让用户多看到一段求助资源」的代价，
相差好几个数量级。所以：
    · 阈值偏向召回（宁可误报）
    · 危机分支**优先于一切**，命中即短路，跳过 RAG 和正常生成
    · LLM 只被允许**升级**风险，永远不能降级

为什么规则在前、LLM 在后（面试必答）：
1. **确定性**：安全关键路径不该依赖一个可能限速、超时、幻觉的远程服务。
   规则层离线可跑、毫秒级、行为完全可预测、可被单元测试逐条覆盖。
2. **可审计**：出问题时能指着具体哪条规则命中/没命中，而不是「模型觉得」。
3. **成本与延迟**：绝大多数消息不需要调用 LLM。
4. **LLM 补的是规则的短板**：不含关键词的隐晦表达（"我把猫托付给我妈了"）。

为什么 LLM 只能升级不能降级（这是最关键的一条）：
    一个非确定性组件**不允许有权关掉一个已经亮起的安全信号**。
    如果允许 LLM 说「这条规则误报了」，那么模型的任何一次幻觉、
    任何一次 prompt 注入，都可能让真实的危机信号被静默掉。
    误报的代价是用户多看到一段求助热线；漏报的代价不可逆。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum

from core.config import CONFIG, Config
from core.utils.llm import LLMError, LLMProvider, complete_json


class RiskLevel(IntEnum):
    """用 IntEnum 是为了能直接比大小做「取更严重者」的合并。"""

    NONE = 0
    ELEVATED = 1  # 明显痛苦/绝望，但无自伤指向 → 正常流程 + 提高关注
    CRISIS = 2    # 自伤/自杀信号 → 短路，直接走危机模板

    @property
    def label(self) -> str:
        return {0: "普通", 1: "需要关注", 2: "危机"}[int(self)]


# --------------------------------------------------------------------------- #
# 0. 惯用语屏蔽：中文里「死」大量用作程度副词
# --------------------------------------------------------------------------- #
# 这是这个模块最实际的一段工程。中文口语里「累死了」「笑死」「饿死了」
# 全是程度表达，跟自杀毫无关系。如果不先处理，误报率会高到系统不可用，
# 而用户学会忽略警告之后，真正的危机提示也会被一起忽略——
# **过度告警本身就是一种安全失效（alert fatigue）**。
_IDIOM_GUARDS = re.compile(
    r"(累|困|笑|饿|渴|热|冷|疼|痛|吓|气|急|愁|烦|无聊|难受|美|香|帅|可爱|好玩|尴尬)死"
    r"|死机|死线|deadline|该死|死活|死党|拼死|死磕|要死不活|半死"
)


def _mask_idioms(text: str) -> str:
    """把惯用语里的「死」替换掉，再交给危机模式匹配。"""
    return _IDIOM_GUARDS.sub("〇", text)


# --------------------------------------------------------------------------- #
# 1. 规则层
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    level: RiskLevel
    negation_sensitive: bool = False  # 前面出现否定词时是否降级


def _r(name: str, regex: str, level: RiskLevel, neg: bool = False) -> Rule:
    return Rule(name, re.compile(regex), level, neg)


CRISIS_RULES: list[Rule] = [
    # ---- 明确的自杀意念表达 ----
    _r("自杀_直述", r"自杀|自尽|轻生|了结自己|结束(我的)?生命|结束这一切", RiskLevel.CRISIS, neg=True),
    _r("不想活", r"不想活|活不下去|不想再活|没法活|活着(太|好)?(累|没意思|没意义)", RiskLevel.CRISIS),
    _r("想死", r"想死|去死|死了算了|一了百了|解脱|不如死", RiskLevel.CRISIS, neg=True),
    _r("消失_永久", r"永远消失|从这个世界消失|人间蒸发|不存在了(比较)?好", RiskLevel.CRISIS),
    # ---- 方法/计划（风险显著更高：有计划意味着从意念走向行动）----
    _r("方法_提及", r"割腕|割手|吞药|攒(了)?药|安眠药|农药|上吊|绳子|跳楼|跳下去|天台|楼顶|煤气|烧炭",
       RiskLevel.CRISIS),
    # 自伤既包括「意图」也包括「痕迹」。第一版只写了动作词，
    # 评测集立刻抓到漏报：「手臂上有新的划痕」不含任何动作词。
    # 教训：规则要覆盖用户**实际会怎么描述**，而不是覆盖症状学词汇。
    _r("自伤_行为", r"伤害自己|弄伤自己|划(自己|手臂|胳膊)|烫自己|打自己|自残|自伤"
                   r"|又割了|割了|割腕|割手", RiskLevel.CRISIS),
    _r("自伤_痕迹", r"划痕|新的(伤口|伤痕|疤)|(手臂|胳膊|手腕|腿)上(有|全是).{0,4}(伤|痕|疤|口子)"
                   r"|烟头烫|自己弄的(伤|口子)", RiskLevel.CRISIS),
    # ---- 告别式语言：常出现在行动前，且往往不含任何「死」字 ----
    _r("告别语", r"遗书|后事|谢谢你(一直|这些年|这么久)|对不起.{0,6}先走|照顾好(自己|我妈|我爸)"
                 r"|把.{0,10}(留给|托付给|送给)(你|我)|(用不上|不需要)了.{0,6}(留给|给你|给我)"
                 r"|最后(跟|和)(你|我)说|再也(见不到|不会打扰)", RiskLevel.CRISIS),
]


# 准备性行为的**良性解释守卫**。
# 和 _IDIOM_GUARDS 同一个思路：不是把规则写得更聪明，而是承认
# 「同一个行为在不同语境里意思完全不同」，然后把语境显式建模出来。
# 命中准备性行为、但附近存在一个平凡的解释（搬家/出差/生日）时降一级。
# 只对 PREPARATORY 生效——绝不对真正的危机规则做任何降级。
_BENIGN_CONTEXT = re.compile(
    r"搬家|出差|旅行|旅游|留学|装修|过敏|房东|不让养|工作调动|回国|出国"
    r"|生日|纪念日|结婚|领证|情人节|过节|毕业"
    r"|太浪费时间|想戒|戒掉|专心|备考|捐掉|捐给|二手|闲鱼"
)

# ---- 准备性行为（安排后事）----
# 这一类是**后来补的**，补的原因写在 docs/ROADMAP.md 里，值得记住：
# 第一版安全层在 41 条评测集上拿到 100% recall，而这个数字是错的——
# 评测集和检测器出自同一个人、同一套心智模型，于是**共享同一个盲区**：
# 两边都只想到了「怎么用语言表达想死」，都没想到「不说死，但在做准备」。
# 送走宠物、把东西托付给人、注销账号、留一封信——这些是自杀风险评估里
# 标准的 warning signs，而且往往**一个死字都不含**。
#
# 定级在 ELEVATED 而不是 CRISIS，是刻意的：
#   「我把猫送去我妈那儿养了」完全可能只是因为他这个月出差。
# 这类信号的**歧义是本质的**，不是规则写得不够好。所以正确的处理不是
# 猜，而是**保证它进入 LLM 二次确认**（`assess()` 在 level < CRISIS 时
# 才调 LLM）。规则层负责「不要漏看」，模型层负责「结合上下文判断」。
# 这正是两级架构存在的理由——第一次有一类信号真的用上了它。
PREPARATORY_RULES: list[Rule] = [
    _r("托付_宠物", r"(把|将)?(猫|狗|它|宠物|兔子|仓鼠).{0,8}(送(去|到|给|走)|托付|交给|拜托).{0,6}"
                    r"(养|照顾|管)|不要(它|猫|狗)了|养不了(它|猫|狗)了", RiskLevel.ELEVATED),
    _r("交代_财物", r"(东西|遗物|衣服|书|收藏|房子|存款).{0,8}(你们|你|他们).{0,6}(看着)?(分|处理|拿走|收着)"
                    r"|(密码|账号|银行卡|保单|保险).{0,10}(告诉(你|我)|发给(你|我)|记一下|存好)", RiskLevel.ELEVATED),
    _r("留信", r"(写|留)了.{0,4}(封|张)?(信|纸条|字条|东西).{0,10}(抽屉|桌上|枕头|给你|你看)"
               r"|等我不在了.{0,6}(再|你)?(看|打开)", RiskLevel.ELEVATED),
    _r("告别_弱式", r"谢谢你(这|那)?(两|三|四|五|几|\d+)年|谢谢你(一路|一直)?(陪|的照顾)"
                    r"|你(要|得)(好好的|过得好|开心)|(以后|将来)(要|你)?好好(照顾自己|生活|的)"
                    r"|别(记恨|怪)我", RiskLevel.ELEVATED),
    _r("清理_痕迹", r"(注销|删掉|删了|清空)(了)?(微博|朋友圈|账号|所有(照片|聊天)|社交)"
                    r"|(退了|退出)所有(群|群聊)", RiskLevel.ELEVATED),
]

ELEVATED_RULES: list[Rule] = [
    _r("绝望", r"永远(好不了|不会好)|(没有|看不到)(任何)?希望|看不到(头|尽头|未来)"
                r"|这辈子(就这样|完了|都?(好不了|不会好))|(一辈子|再也)(好不了|不会好)", RiskLevel.ELEVATED),
    _r("撑不住", r"撑不(下去|住)|坚持不(下去|住)|扛不住|受不了了|到极限了", RiskLevel.ELEVATED),
    _r("无价值", r"不配(被爱|活着|存在)|是个(废物|累赘|垃圾)|一无是处|没有价值|多余的", RiskLevel.ELEVATED),
    _r("拖累", r"拖累|负担|没有我(会|更)?(好|轻松)|离开我(会|更)?好", RiskLevel.ELEVATED),
    _r("无意义", r"活着没(意思|意义)|没有意义|无所谓了|随便吧.{0,4}反正", RiskLevel.ELEVATED),
]

# 否定词。命中「否定 + 危机词」时**降一级**，不清零——
# "他说他不会自杀" 值得放松警惕，但不值得完全不看。
_NEGATORS = re.compile(r"(不会|没有|不是|不想|并不|绝不|别|不至于)$")


@dataclass
class RiskAssessment:
    level: RiskLevel
    signals: list[str] = field(default_factory=list)
    stage: str = "rule"          # rule | llm
    rationale: str = ""

    @property
    def is_crisis(self) -> bool:
        return self.level >= RiskLevel.CRISIS


def rule_scan(text: str) -> RiskAssessment:
    """第一级：纯规则，确定性，无网络，微秒级。"""
    masked = _mask_idioms(text)
    level = RiskLevel.NONE
    signals: list[str] = []

    prep_names = {r.name for r in PREPARATORY_RULES}
    benign = bool(_BENIGN_CONTEXT.search(masked))

    for rule in CRISIS_RULES + PREPARATORY_RULES + ELEVATED_RULES:
        for m in rule.pattern.finditer(masked):
            hit_level = rule.level
            if benign and rule.name in prep_names:
                # 有平凡解释 → 降级。仍然记一条信号：降级不等于当没看见。
                hit_level = RiskLevel(max(RiskLevel.NONE, hit_level - 1))
                signals.append(f"{rule.name}(良性语境降级):{m.group()}")
                continue
            if rule.negation_sensitive:
                prefix = masked[max(0, m.start() - 4) : m.start()]
                if _NEGATORS.search(prefix):
                    # 降级而非清零：非确定性/模糊的情况一律偏保守
                    hit_level = RiskLevel(max(RiskLevel.NONE, hit_level - 1))
            if hit_level > RiskLevel.NONE:
                signals.append(f"{rule.name}:{m.group()}")
            level = max(level, hit_level)

    return RiskAssessment(level=level, signals=signals, stage="rule",
                          rationale="规则命中: " + ", ".join(signals) if signals else "规则层未命中")


# --------------------------------------------------------------------------- #
# 2. LLM 二次确认（只升不降）
# --------------------------------------------------------------------------- #
_LLM_SYSTEM = """你是一个心理危机信号识别器，服务对象是抑郁症患者的伴侣。
你的唯一任务：判断这段话里，患者本人是否表达了自伤或自杀的意图、计划或告别。

判定 crisis 的情形（宁可多判，不可漏判）：
- 直接或隐晦地表达不想活、想结束生命
- 提到具体方法、地点、时间或已经在准备
- 告别式表达：交代后事、把重要的东西送人、说最后的话
- 已经发生的自伤行为

判定 elevated 的情形：强烈痛苦、绝望、自我否定，但没有自伤指向。
判定 none 的情形：日常低落、抱怨、疲惫。

只输出 JSON，不要任何其他文字：
{"level": "none|elevated|crisis", "reason": "一句话依据"}"""


def llm_scan(text: str, llm: LLMProvider) -> RiskAssessment:
    """第二级：让模型抓规则抓不到的隐晦表达。"""
    try:
        data = complete_json(llm, _LLM_SYSTEM, text, temperature=0.0)
    except LLMError as exc:
        # 关键的失败模式设计：LLM 挂了不能让整个安全层失效。
        # 降级为「规则层结论」而不是抛异常——安全层必须永远可用。
        return RiskAssessment(RiskLevel.NONE, stage="llm", rationale=f"LLM 不可用，跳过二级: {exc}")
    mapping = {"none": RiskLevel.NONE, "elevated": RiskLevel.ELEVATED, "crisis": RiskLevel.CRISIS}
    level = mapping.get(str(data.get("level", "none")).lower(), RiskLevel.NONE)
    return RiskAssessment(level=level, stage="llm", rationale=str(data.get("reason", "")))


# --------------------------------------------------------------------------- #
# 3. 对外接口
# --------------------------------------------------------------------------- #
def assess(text: str, llm: LLMProvider | None = None, cfg: Config | None = None) -> RiskAssessment:
    """完整评估。返回两级中**更严重**的那个。

    注意这里的 max()：LLM 只能把风险往上抬。
    规则说 CRISIS 而 LLM 说 none 时，结果依然是 CRISIS。
    """
    cfg = cfg or CONFIG
    rule_result = rule_scan(text)

    # 规则已判危机 → 没必要再问 LLM（它也无权降级），省一次调用和延迟
    if rule_result.is_crisis or llm is None or not cfg.crisis_llm_second_pass:
        return rule_result

    llm_result = llm_scan(text, llm)
    if llm_result.level > rule_result.level:
        return RiskAssessment(
            level=llm_result.level,
            signals=rule_result.signals + [f"llm:{llm_result.rationale}"],
            stage="llm",
            rationale=f"LLM 升级风险等级: {llm_result.rationale}",
        )
    return rule_result
