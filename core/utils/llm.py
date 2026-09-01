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

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
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
        gen: dict = {"temperature": temperature, "maxOutputTokens": 8192}
        if "2.5" in self.model or "3." in self.model:
            # 2.5 之后的 flash 默认开思考。不关掉的话，思考会先吃掉输出预算，
            # 于是 candidate 里根本没有 text 部分——表现出来是一句
            # "返回结构异常"，看不出真实原因。蒸馏是结构化抽取，
            # 不需要思考预算。
            gen["thinkingConfig"] = {"thinkingBudget": 0}
        try:
            resp = httpx.post(
                url,
                headers={"x-goog-api-key": self.api_key},
                json={
                    "system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"parts": [{"text": user}]}],
                    "generationConfig": gen,
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
            # 200 但取不到文本，最常见的原因是 finishReason 而不是格式问题
            # （MAX_TOKENS：预算被思考吃完；SAFETY：被安全过滤拦下——
            # 本项目的素材天然涉及自伤话题，这条会真的发生）。
            # 把 finishReason 提到最前面，否则排查要靠猜。
            reason = ""
            try:
                cand = resp.json().get("candidates") or [{}]
                reason = cand[0].get("finishReason") or ""
            except Exception:  # noqa: BLE001
                pass
            hint = f"（finishReason={reason}）" if reason else ""
            raise LLMError(
                f"Gemini 没有返回文本{hint}: {resp.text[:300]}"
            ) from exc


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


class CassetteProvider:
    """录制 / 回放真实 LLM 调用（cassette 模式）。

    ━━━ 为什么评测需要这一层 ━━━

    一份发布出去的评测报告，别人应该能复现。但生成评测依赖一次真实的
    LLM 调用，而真实调用是：要 key、要额度、会限速、**而且不确定**——
    同样的 prompt 明天跑出来就是另一批回复。于是「我复现不出你的数字」
    既可能是我配错了，也可能只是模型飘了，谁也说不清。

    解决办法是软件工程里已经很成熟的一招（HTTP 测试里叫 VCR / cassette）：
    **把那一次真实调用的输入输出录下来，之后按输入的哈希回放。**

    - 录制一次 → 评测结果永久可复现，不需要 key，不花钱，CI 里能跑
    - prompt 一改，哈希就对不上 → 强制你重新录制，不会拿旧回复冒充新系统
      （这条很重要：它让 cassette 没法变成自欺欺人的工具）
    - 回放缺条目时是**硬失败**，不是静默降级——评测工具必须能区分
      「系统坏了」和「系统表现差」

    `strict=False` 时缺条目会转交给 `fallback`（真实 provider），
    用于增量补录：改了两个情境，只重打那两次。
    """

    name = "cassette"

    def __init__(self, path: "Path", *, mode: str = "replay",
                 fallback: "LLMProvider | None" = None, strict: bool = True) -> None:
        self.path = Path(path)
        self.mode = mode          # "replay" | "record"
        self.fallback = fallback
        self.strict = strict
        self.hits = 0
        self.misses = 0
        self._data: dict[str, dict] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def key_for(system: str, user: str) -> str:
        """输入的指纹。prompt 变了指纹就变了——这正是我们要的。"""
        h = hashlib.sha256()
        h.update(system.encode("utf-8"))
        h.update(b"\x00")
        h.update(user.encode("utf-8"))
        return h.hexdigest()[:16]

    def complete(self, system: str, user: str, *, temperature: float = 0.3) -> str:
        k = self.key_for(system, user)
        entry = self._data.get(k)
        if entry is not None:
            self.hits += 1
            return entry["response"]

        self.misses += 1
        if self.fallback is None or self.strict:
            raise LLMError(
                f"cassette 里没有这条记录（key={k}）。"
                "prompt 变过了，或者这次的情境没录过 —— 需要重新录制，"
                "不能拿旧回复冒充新系统。"
            )
        resp = self.fallback.complete(system, user, temperature=temperature)
        self._data[k] = {"system": system, "user": user, "response": resp}
        self.save()
        return resp

    def put(self, system: str, user: str, response: str) -> str:
        """手工写入一条（用于离线补录）。返回 key。"""
        k = self.key_for(system, user)
        self._data[k] = {"system": system, "user": user, "response": response}
        return k

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._data)


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
    retries: int = 3,
) -> Any:
    """要求结构化输出的调用：带重试 + JSON 修复。

    重试用指数退避（4s, 8s, 16s），因为免费模型最常见的失败是限速（429）。
    退避从 1s 改成 4s 起步，是因为免费档的配额单位是**每分钟**
    （Gemini 免费档 10 RPM）：撞上限速后 1 秒回来必然再撞一次，
    等于把三次重试机会一口气浪费掉。退避必须和配额的时间尺度同量级。
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            raw = llm.complete(system, user, temperature=temperature)
            return extract_json(raw)
        except LLMError as exc:
            last = exc
            if attempt < retries:
                time.sleep(min(30, 4 * 2**attempt))
    raise LLMError(f"complete_json 在 {retries + 1} 次尝试后失败: {last}")
