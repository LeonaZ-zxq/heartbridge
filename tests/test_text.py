"""聊天记录解析测试。真实输入是脏的，这里覆盖实际会遇到的格式。"""
from core.utils.text import Turn, last_message_from, parse_transcript

WECHAT = """小鱼 2026-08-30 23:41
今天又没去上班
躺了一天

Leona 23:42
没事的，累了就休息

小鱼 23:45
我觉得我就是个废物"""

INLINE = """小鱼：我今天很难受
Leona：怎么了
小鱼：说不上来"""


def test_parses_wechat_merged_forward():
    turns = parse_transcript(WECHAT)
    assert [t.speaker for t in turns] == ["小鱼", "Leona", "小鱼"]
    assert "躺了一天" in turns[0].text


def test_multi_line_message_stays_in_one_turn():
    """同一个人连发的多行必须合成一轮，否则轮次结构就错了。"""
    turns = parse_transcript(WECHAT)
    assert turns[0].text.count("\n") == 1


def test_parses_inline_colon_format():
    turns = parse_transcript(INLINE)
    assert len(turns) == 3
    assert turns[2].text == "说不上来"


def test_noise_lines_are_dropped():
    turns = parse_transcript("以下是聊天记录\n————\n小鱼 10:00\n在吗\n[图片]")
    assert len(turns) == 1 and turns[0].text == "在吗"


def test_consecutive_same_speaker_merged():
    turns = parse_transcript("A 10:00\n一\nA 10:01\n二")
    assert len(turns) == 1 and turns[0].text == "一\n二"


def test_plain_text_without_speakers_still_works():
    """用户直接口述情境而不粘贴记录时，不能崩，也不能瞎猜说话人。"""
    turns = parse_transcript("他今天说他很累")
    assert len(turns) == 1 and turns[0].speaker == "未知"


def test_last_message_from():
    turns = [Turn("A", "1"), Turn("B", "2"), Turn("A", "3")]
    assert last_message_from(turns) == "3"
    assert last_message_from(turns, "B") == "2"
