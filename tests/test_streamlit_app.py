"""Streamlit 界面的冒烟测试，用官方的 AppTest 真正渲染每一页。

为什么值得测 UI 层：
业务逻辑已经在 core 上测过了，这里要防的是另一类问题——
页面里写错一个字段名、改了 core 的接口忘了同步 UI、
或者**危机资源/免责声明在某一页漏掉了**。最后这一条是安全要求，
不能靠人记得。

这些测试同样不需要网络和 API key：demo 模式下 UI 用的是 MockProvider。
"""
import pytest

streamlit = pytest.importorskip("streamlit", reason="UI 是可选依赖")
from streamlit.testing.v1 import AppTest  # noqa: E402

from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# AppTest 的相对路径是相对**调用它的文件**解析的，所以这里一律用绝对路径
APP = str(ROOT / "streamlit_app/app.py")
PAGES = [
    str(ROOT / "streamlit_app/pages/1_卡片库.py"),
    str(ROOT / "streamlit_app/pages/2_躯体化科普.py"),
    str(ROOT / "streamlit_app/pages/3_设计与隐私.py"),
]
TIMEOUT = 60


def run(path: str) -> AppTest:
    at = AppTest.from_file(path, default_timeout=TIMEOUT)
    at.run()
    return at


@pytest.mark.parametrize("path", [APP, *PAGES], ids=lambda p: Path(p).stem)
def test_every_page_renders_without_exception(path):
    at = run(path)
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("path", [APP, *PAGES], ids=lambda p: Path(p).stem)
def test_every_page_shows_crisis_resources_in_sidebar(path):
    """安全要求：任何一页都可能是用户在最糟的时刻打开的那一页。"""
    at = run(path)
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown)
    assert "13 11 14" in sidebar_text, "Lifeline 号码缺失"
    assert "000" in sidebar_text
    assert "12356" in sidebar_text, "中国大陆通道缺失"


@pytest.mark.parametrize("path", [APP, *PAGES], ids=lambda p: Path(p).stem)
def test_every_page_carries_the_disclaimer(path):
    at = run(path)
    all_text = " ".join(
        [c.value for c in at.caption] + [m.value for m in at.sidebar.markdown]
        + [c.value for c in at.sidebar.caption]
    )
    assert "不做诊断" in all_text


def test_main_page_normal_flow_produces_options_with_reasons():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.text_area(key="transcript").set_value("小鱼 23:45\n我觉得我就是个废物")
    at.button(key="go").click().run()
    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "为什么有效" in body


def test_main_page_crisis_input_shows_crisis_branch_not_options():
    """端到端复验安全短路：网页端和 CLI 必须表现一致。"""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.text_area(key="transcript").set_value("小鱼 03:12\n我不想活了")
    at.button(key="go").click().run()
    assert not at.exception
    assert at.error, "危机输入必须显示 error 级别的提示"
    body = " ".join(m.value for m in at.markdown)
    assert "13 11 14" in body
    assert "为什么有效" not in body, "危机分支不该出现正常回复选项"


def test_somatic_page_shows_both_in_person_and_remote():
    at = run(PAGES[1])
    body = " ".join(m.value for m in at.markdown)
    assert "你在他身边时" in body and "异地" in body
    assert "必须就医" in " ".join(e.value for e in at.error)
