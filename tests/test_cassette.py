"""cassette（录制/回放）的行为契约。

它存在的意义是让一份发布出去的评测可复现。所以最关键的一条不是「能回放」，
而是「**prompt 改了就必须硬失败**」——否则 cassette 会变成拿旧回复
给新系统贴金的工具，比没有评测更糟。
"""
import pytest

from core.utils.llm import CassetteProvider, LLMError


def test_录制后能按内容回放(tmp_path):
    c = CassetteProvider(tmp_path / "c.json", mode="record")
    c.put("SYS", "USER", '{"ok": true}')
    c.save()

    replay = CassetteProvider(tmp_path / "c.json")
    assert replay.complete("SYS", "USER") == '{"ok": true}'
    assert replay.hits == 1


def test_prompt_变了就硬失败_而不是静默回放旧的(tmp_path):
    c = CassetteProvider(tmp_path / "c.json", mode="record")
    c.put("SYS", "USER", '{"ok": true}')
    c.save()

    replay = CassetteProvider(tmp_path / "c.json")
    with pytest.raises(LLMError, match="没有这条记录"):
        replay.complete("SYS", "USER 改了一个字")


def test_key_只由输入决定(tmp_path):
    a = CassetteProvider.key_for("s", "u")
    assert a == CassetteProvider.key_for("s", "u")
    assert a != CassetteProvider.key_for("s", "u2")
    # 分隔符：确保 ("ab","c") 和 ("a","bc") 不会撞在一起
    assert CassetteProvider.key_for("ab", "c") != CassetteProvider.key_for("a", "bc")
