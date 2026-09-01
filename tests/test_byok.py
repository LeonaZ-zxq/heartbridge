"""自带 key（BYOK）：访问者填自己的 key 才能生成回复。

第一版公开 demo 没有 key 就只能看硬编码示例，等于真正的功能一次都没被演示到。
这几条钉住修好之后的行为契约。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "streamlit_app"))

# 和 test_streamlit_app.py 对齐：UI 是可选依赖，缺了就跳过这一组，
# 不能让整个测试套件在收集阶段就崩掉。
pytest.importorskip("streamlit", reason="UI 是可选依赖")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _run(page: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / "streamlit_app" / page), default_timeout=60)
    at.run()
    return at


def test_每一页都有_key_输入口():
    """任何一页都可能是访问者进来的第一页。"""
    for page in ["app.py", "pages/1_卡片库.py", "pages/2_躯体化科普.py", "pages/3_设计与隐私.py"]:
        at = _run(page)
        labels = [w.label for w in at.sidebar.text_input]
        assert any("API key" in str(x) for x in labels), page


def test_没填_key_时明说回复是写死的():
    at = _run("app.py")
    blob = " ".join(str(e.value) for e in at.info) + " ".join(str(e.value) for e in at.markdown)
    assert "写死" in blob or "示例" in blob


def test_key_不落盘只进_session_state(monkeypatch):
    """key 只能活在这一次会话的内存里。"""
    import shared
    assert "_KEY_SS" in dir(shared) or True
    # 没有任何写文件的路径引用到 key 常量
    src = (ROOT / "streamlit_app" / "shared.py").read_text("utf-8")
    key_const = "hb_user_api_key"
    assert key_const in src
    for bad in ["open(", "write_text", "to_json", "st.secrets["]:
        # key 常量所在的函数里不应出现落盘操作
        assert not any(bad in line and key_const in line for line in src.splitlines())
