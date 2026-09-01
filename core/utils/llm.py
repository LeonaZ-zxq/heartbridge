"""LLM 调用抽象层：一个接口，三个后端（OpenRouter / Gemini / Mock）。

为什么要这一层（面试高频追问「你为什么不直接调 openai SDK」）：
1. **供应商锁定风险**：免费模型经常限速、改名、下线。业务代码只依赖 `LLMProvider`
   这个接口，换供应商只改一个工厂函数。
2. **可测试性**：`MockProvider` 让整个测试套件不需要网络、不需要 API key、不花钱，
   且结果确定（deterministic），CI 才能跑。这是把「LLM 应用」变成「可工程化的软件」
   的关键一步。
3. **可靠性**：LLM 返回的 JSON 经常带 markdown 围栏或尾随解释文字。JSON 修复 + 重试
   逻辑写在这一层，业务代码拿到的永远是干净的 dict。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from core.config import CONFIG, Config


class LLMError(RuntimeError):
    """LLM 调用失败（网络、鉴权、限速、返回无法解析）。"""


# --------------------------------------------------------------------------- #
# 接口
# --------------------------------------------------------------------------- #
class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str, *, temperature: float = 0.3) -> str:
        """给定 system + user prompt，返回模型的纯文本回复。"""
        ...


# --------------------------------------------------------------------------- #
# 后端实现
# --------------------------------------------------------------------------- #
@dataclass
class OpenRouterProvider:
    """OpenRouter：一个网关聚合了几十个模型，含免费额度模型。兼容 OpenAI 协议。"""

    api_key: str
    model: str
    timeout_s: int = 60
    name: str = "openrouter"

    def complete(self, system: str, user: str, *, temperature: float = 0.3) -> str:
        import httpx

        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY 未设置")
        try:
            resp = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com/heartbridge",
                    "X-Title": "HeartBridge",
                },
                json={
                    "model": self.model,
                    "temperature": temperature,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=self.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - 网络异常统一包成 LLMError
            raise LLMError(f"OpenRouter 请求失败: {exc}") from exc

        if resp.status_code != 200:
            raise LLMError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"OpenRouter 返回结构异常: {resp.text[:300]}") from exc


@dataclass
class GeminiProvider:
    """Gemini：备用通道。长上下文（整篇视频文字稿蒸馏）时比免费小模型更稳。"""

    api_key: str
    model: str
    timeout_s: int = 60
    name: str = "gemini"

    def complete(self, system: str, user: str, *, temperature: float = 0.3) -> str:
        import httpx

        if not self.api_key:
            raise LLMError("GEMINI_API_KEY 未设置")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        try:
            resp = httpx.post(
                url,
                headers={"x-goog-api-key": self.api_key},
                json={
                    "system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"parts": [{"text": user}]}],
                    "generationConfig": {"temperature": temperature},
                },
                timeout=self.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Gemini 请求失败: {exc}") from exc

        if resp.status_code != 200:
            raise LLMError(f"Gemini HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Gemini 返回结构异常: {resp.text[:300]}") from exc


class MockProvider:
    """确定性假 LLM。

    不是「为了偷懒」，而是一个明确的工程决策：
    - 单元测试要断言业务逻辑（分类分支、检索融合、模板渲染），不该断言模型输出。
    - 用 handler 按 prompt 内容路由，让不同测试拿到不同的可控回复。
    """

    name = "mock"

    def __init__(self) -> None:
        self._handlers: list[tuple[str, Callable[[str, str], str]]] = []
        self.calls: list[dict[str, str]] = []  # 便于测试断言「有没有调用 LLM」

    def register(self, marker: str, handler: Callable[[str, str], str]) -> None:
        """当 prompt 里出现 marker 时，用 handler 生成回复。后注册的优先。"""
        self._handlers.insert(0, (marker, handler))

    def complete(self, system: str, user: str, *, temperature: float = 0.3) -> str:
        self.calls.append({"system": system, "user": user})
        blob = f"{system}\n{user}"
        for marker, handler in self._handlers:
            if marker in blob:
                return handler(system, user)
        return json.dumps({"mock": True, "note": "no handler matched"}, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# 工厂 + 便捷函数
# --------------------------------------------------------------------------- #
def get_llm(cfg: Config | None = None) -> LLMProvider:
    cfg = cfg or CONFIG
    provider = cfg.llm_provider.lower()
    if provider == "openrouter":
        return OpenRouterProvider(cfg.openrouter_key, cfg.openrouter_model, cfg.llm_timeout_s)
    if provider == "gemini":
        return GeminiProvider(cfg.gemini_key, cfg.gemini_model, cfg.llm_timeout_s)
    if provider == "mock":
        return MockProvider()
    raise LLMError(f"未知 LLM provider: {cfg.llm_provider}")


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def extract_json(text: str) -> Any:
    """从模型回复里挖出 JSON。

    真实世界的模型输出经常是：
        这是你要的结果：
        ```json
        {"a": 1}
        ```
        希望有帮助！
    所以要：去围栏 → 直接 parse → 失败则截取第一个 {...} 或 [...] 再 parse。
    """
    cleaned = _FENCE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"无法从模型输出中解析 JSON: {text[:300]}")


def complete_json(
    llm: LLMProvider,
    system: str,
    user: str,
    *,
    temperature: float = 0.2,
    retries: int = 2,
) -> Any:
    """要求结构化输出的调用：带重试 + JSON 修复。

    重试用指数退避（1s, 2s），因为免费模型最常见的失败是限速（429），
    立刻重试只会再撞一次。
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            raw = llm.complete(system, user, temperature=temperature)
            return extract_json(raw)
        except LLMError as exc:
            last = exc
            if attempt < retries:
                time.sleep(2**attempt)
    raise LLMError(f"complete_json 在 {retries + 1} 次尝试后失败: {last}")
