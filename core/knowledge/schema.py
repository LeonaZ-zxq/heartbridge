"""知识卡片的数据模型与校验。

核心工程原则：**在边界处校验（validate at the edge）**。
LLM 蒸馏出来的东西是不可信的半结构化文本，它进入系统的第一道关口就是这里。
一旦通过校验，下游（检索、生成、UI）就可以假设数据一定合法，不用到处写 if 判空。

面试点：
- 为什么用 Pydantic 而不是 dataclass / 裸 dict？
  → 需要**运行时**校验 + 清晰的错误信息 + JSON 序列化，dataclass 只有类型注解，
    不在运行时拦截脏数据。
- 为什么躯体化卡强制要求 source？
  → 这是医疗相邻内容。LLM 幻觉一条医学建议的代价远高于沟通技巧说错。
    用类型系统把「不许无来源编造」变成一条编译期/加载期就会失败的硬约束，
    而不是写在 prompt 里祈祷模型遵守。
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
CardType = Literal["communication", "somatic", "crisis"]

ID_PATTERN = re.compile(r"^(comm|soma|crisis)_[0-9]{3,}$")


class Source(BaseModel):
    """来源可追溯。任何一张卡都必须能回答「这条建议是哪来的」。"""

    platform: str | None = Field(None, description="douyin / xiaohongshu / bilibili / web")
    authority: str | None = Field(None, description="权威机构名，如 Beyond Blue")
    author: str | None = None
    title: str | None = None
    url: str | None = None
    date_ingested: date = Field(default_factory=date.today)

    @model_validator(mode="after")
    def _need_some_provenance(self) -> "Source":
        if not any([self.platform, self.authority, self.url, self.author]):
            raise ValueError("Source 至少要有 platform / authority / url / author 其中之一")
        return self


class BaseCard(BaseModel):
    id: NonEmptyStr
    type: CardType
    source: Source
    tags: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not ID_PATTERN.match(v):
            raise ValueError(f"id 必须形如 comm_001 / soma_001 / crisis_001，收到 {v!r}")
        return v

    def index_text(self) -> str:
        """喂给检索器做索引的文本。

        这是 RAG 里最容易被忽视、但对命中率影响最大的一个设计决策。
        我们**不**索引整张卡的 JSON，而是拼接「用户可能会怎么描述这个情境」的字段：
        场景描述、症状名、反例（dont/avoid_saying）。
        因为用户查询长得像「他说自己不配被爱」，而不像「先接住情绪不急着反驳」。
        索引文本要贴近 query 的语言分布，而不是贴近答案的语言分布。

        这不是拍脑袋的设计，是 scripts/eval_index_ablation.py 测出来的结论：
        在 32 条查询的评测集上（BM25 后端），
            只索引技巧名                 Recall@3 = 31.2%
            索引场景 + 别名 + 标签       Recall@3 = 87.5%   ← 当前配置
            额外加入 example_phrases     Recall@3 = 84.4%   （变差）
            索引全部字段                 Recall@3 = 87.5%，但 Recall@1 从 78.1% 掉到 71.9%
        结论：加入「答案侧」的语言（例句、技巧名）会稀释索引，
        而把整张卡全塞进去虽然不降低召回，却明显损害排序质量。
        """
        raise NotImplementedError


class CommunicationCard(BaseCard):
    type: Literal["communication"] = "communication"
    scenario: NonEmptyStr = Field(description="什么情境下用，用用户会说的话来写")
    technique_name: NonEmptyStr
    do: list[NonEmptyStr] = Field(min_length=1)
    dont: list[NonEmptyStr] = Field(min_length=1)
    example_phrases: list[NonEmptyStr] = Field(min_length=1)
    why_it_works: NonEmptyStr
    user_phrasings: list[str] = Field(
        default_factory=list,
        description="用户可能会怎么描述这个处境（口语、俚语、缩写）。"
                    "这是**文档扩展 / doc2query**：用预期的查询语言扩充索引，"
                    "补上词法检索最大的短板——文档和查询用词不重合（vocabulary mismatch）。",
    )

    def index_text(self) -> str:
        # 只索引「用户会怎么描述这个处境」：场景 + 用户口语说法 + 主题标签。
        # 刻意不含 technique_name / example_phrases —— 见消融实验结论。
        return " ".join([self.scenario, *self.user_phrasings, *self.tags])


class SomaticCard(BaseCard):
    type: Literal["somatic"] = "somatic"
    symptom: NonEmptyStr
    aliases: list[str] = Field(default_factory=list, description="用户可能的口语说法")
    what_it_is: NonEmptyStr
    in_person: list[NonEmptyStr] = Field(min_length=1, description="你在他身边时怎么做")
    remote: list[NonEmptyStr] = Field(min_length=1, description="异地/线上时怎么做")
    say: list[NonEmptyStr] = Field(min_length=1)
    avoid_saying: list[NonEmptyStr] = Field(min_length=1)
    seek_help_if: list[NonEmptyStr] = Field(min_length=1, description="什么情况必须转专业/就医")

    @model_validator(mode="after")
    def _require_authority(self) -> "SomaticCard":
        # 硬约束：医疗相邻内容必须有权威来源，不接受「某博主说」单独成立。
        if not (self.source.authority or self.source.url):
            raise ValueError(
                f"躯体化卡 {self.id} 缺少权威来源：source.authority 或 source.url 必须有一个"
            )
        return self

    def index_text(self) -> str:
        # aliases 是用户的口语说法（"喘不上气" vs 学名"过度换气"），
        # 对躯体化卡来说这是命中率的主要来源。
        return " ".join([self.symptom, *self.aliases, *self.tags])


Card = Union[CommunicationCard, SomaticCard]


def parse_card(raw: dict) -> Card:
    """按 type 字段分派到对应模型。校验失败会抛 pydantic.ValidationError。"""
    t = raw.get("type")
    if t == "communication":
        return CommunicationCard.model_validate(raw)
    if t == "somatic":
        return SomaticCard.model_validate(raw)
    raise ValueError(f"未知卡片 type: {t!r}")


def load_cards(path: Path) -> list[Card]:
    """从一个 JSON 文件或一个装着 JSON 文件的目录加载卡片。"""
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    cards: list[Card] = []
    seen: set[str] = set()
    for f in files:
        payload = json.loads(f.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else [payload]
        for raw in items:
            card = parse_card(raw)
            if card.id in seen:
                raise ValueError(f"卡片 id 重复: {card.id}（在 {f.name}）")
            seen.add(card.id)
            cards.append(card)
    return cards
