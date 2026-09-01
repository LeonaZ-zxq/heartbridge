"""恋人 profile 的数据模型。

隐私设计（面试里的 responsible AI 部分，要主动讲）：
- 这里存的是**最高敏感级**的个人健康信息（心理诊断、用药、崩溃史）。
- 因此整个 profile 只落在本机 SQLite，永不上传，`data/` 全目录 gitignore。
- 云端 demo 版本只加载 examples/ 里的假数据。
- 字段全部可选：不强迫用户为了用工具而交出更多信息（数据最小化原则）。
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class PartnerProfile(BaseModel):
    profile_id: str = "default"
    nickname: str = "他"
    relationship_years: float | None = None
    long_distance: bool = False

    diagnosis: str | None = Field(None, description="可选。用户不填也能用。")
    medication: str | None = None
    in_treatment: bool | None = None

    triggers: list[str] = Field(default_factory=list, description="已知触发点")
    core_fears: list[str] = Field(default_factory=list, description="反复出现的核心焦虑")
    landmines: list[str] = Field(default_factory=list, description="雷区话题")
    what_helps: list[str] = Field(default_factory=list, description="以前有效的安慰")
    what_backfires: list[str] = Field(default_factory=list, description="以前起反效果的")

    # few-shot 语气样本：让生成的回复像"你"，而不像客服
    my_voice_samples: list[str] = Field(default_factory=list)
    pet_names: list[str] = Field(default_factory=list)

    updated_at: date = Field(default_factory=date.today)

    def to_prompt_block(self) -> str:
        """把 profile 压成注入 prompt 的一段文本。

        只注入**非空**字段：空字段写进 prompt 只会浪费 token 并稀释注意力。
        """
        lines: list[str] = [f"称呼：{self.nickname}"]
        if self.relationship_years:
            lines.append(f"在一起：{self.relationship_years} 年")
        if self.long_distance:
            lines.append("异地：是（很多在场做法不适用，优先给远程方案）")
        if self.diagnosis:
            lines.append(f"诊断：{self.diagnosis}")
        if self.in_treatment is not None:
            lines.append(f"正在接受治疗：{'是' if self.in_treatment else '否'}")
        for label, values in (
            ("已知触发点", self.triggers),
            ("核心焦虑", self.core_fears),
            ("雷区（绝对不要提）", self.landmines),
            ("以前有效的安慰", self.what_helps),
            ("以前起反效果的", self.what_backfires),
            ("你们的称呼习惯", self.pet_names),
        ):
            if values:
                lines.append(f"{label}：{'；'.join(values)}")
        return "\n".join(lines)
