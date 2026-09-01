"""把微信/QQ 的「合并转发」文本还原成结构化对话。

为什么值得单独一个模块（面试点：真实系统的输入永远是脏的）：

用户不会填表单，他们会**整段粘贴聊天记录**。这段文本的格式取决于
客户端版本、系统语言、有没有时间戳、名字里有没有冒号……
把它可靠地还原成 (说话人, 内容) 序列，是这个产品能不能用的前提。

策略：**规则优先，LLM 兜底**——和安全层同一个哲学。
    · 90% 的情况正则能精确解析：确定性、零成本、零延迟、可测试
    · 剩下的边缘格式再交给 LLM
反过来（先问 LLM）就是把一个可以确定性求解的问题变贵、变慢、变不可靠。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from core.utils.llm import LLMError, LLMProvider, complete_json


@dataclass
class Turn:
    speaker: str
    text: str

    def render(self) -> str:
        return f"{self.speaker}: {self.text}"


# 「名字 + 时间戳」单独成行，正文在下一行 —— 微信合并转发最常见的形态
_HEADER = re.compile(
    r"^\s*(?P<name>[^\s:：]{1,20}?)\s+"
    r"(?P<ts>(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+)?"
    r"(?:上午|下午|凌晨|早上|中午|晚上)?\s*\d{1,2}:\d{2}(?::\d{2})?)\s*$"
)
# 「名字：正文」同一行
_INLINE = re.compile(r"^\s*(?P<name>[^\s:：]{1,20})[:：]\s*(?P<text>.+?)\s*$")
# 系统提示行，直接丢弃
_NOISE = re.compile(
    r"^\s*(?:以下是.*(?:聊天|消息)记录.*|—+|-{3,}|={3,}"
    r"|\[(?:图片|表情|语音|视频|文件|动画表情|位置|链接)\]"
    r"|.*撤回了一条消息|.*加入了群聊|.*邀请.*加入了群聊)\s*$"
)


def parse_transcript(raw: str, llm: LLMProvider | None = None) -> list[Turn]:
    """解析聊天文本。规则解析不到 2 轮时，才考虑用 LLM 兜底。"""
    turns = _rule_parse(raw)
    if len(turns) >= 2 or llm is None:
        return turns
    return _llm_parse(raw, llm) or turns


def _rule_parse(raw: str) -> list[Turn]:
    turns: list[Turn] = []
    pending: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal pending, buffer
        if pending is not None and buffer:
            body = "\n".join(b for b in buffer if b.strip())
            if body.strip():
                turns.append(Turn(pending, body.strip()))
        buffer = []

    for line in raw.splitlines():
        if _NOISE.match(line):
            continue
        header = _HEADER.match(line)
        if header:
            flush()
            pending = header.group("name")
            continue
        inline = _INLINE.match(line)
        # 只有在「不处于某个说话人的正文块中」时才把冒号行当作新的一轮，
        # 否则正文里出现「他说：xxx」会被误判成换人
        if inline and pending is None:
            turns.append(Turn(inline.group("name"), inline.group("text")))
            continue
        if pending is not None:
            buffer.append(line)
        elif line.strip():
            turns.append(Turn("未知", line.strip()))
    flush()
    return _merge_consecutive(turns)


def _merge_consecutive(turns: list[Turn]) -> list[Turn]:
    """同一个人连发多条合并成一轮——这才符合「一轮对话」的语义。"""
    merged: list[Turn] = []
    for t in turns:
        if merged and merged[-1].speaker == t.speaker:
            merged[-1] = Turn(t.speaker, merged[-1].text + "\n" + t.text)
        else:
            merged.append(t)
    return merged


_LLM_SYSTEM = """把下面的聊天记录还原成结构化对话。
只输出 JSON 数组，每项 {"speaker": "说话人", "text": "内容"}，不要任何其他文字。"""


def _llm_parse(raw: str, llm: LLMProvider) -> list[Turn]:
    try:
        data = complete_json(llm, _LLM_SYSTEM, raw, temperature=0.0)
    except LLMError:
        return []
    out: list[Turn] = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and item.get("text"):
            out.append(Turn(str(item.get("speaker", "未知")), str(item["text"])))
    return out


def last_message_from(turns: list[Turn], speaker: str | None = None) -> str:
    """取最后一条（可指定说话人）——这通常就是「需要回什么」的那句话。"""
    for t in reversed(turns):
        if speaker is None or t.speaker == speaker:
            return t.text
    return ""
