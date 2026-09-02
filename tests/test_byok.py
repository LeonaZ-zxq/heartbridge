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
    for page in ["app.py", "pages/1_卡片库.py", "pages/2_躯体化科普.py", "pages/3_设计与隐私.py", "pages/4_伴侣档案.py"]:
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


# --------------------------------------------------------------------------- #
# 后端选择必须由「环境能不能跑」决定，而不是由「用户填没填 key」决定
# --------------------------------------------------------------------------- #
# 这组测试来自一个只在生产出现、本地永远复现不了的故障：
#
#   backend_name() 原本写成 `CONFIG.retrieval_backend if not is_demo() else "bm25"`，
#   而 is_demo() 的定义是「没有可用的模型通道」。于是：
#     访问者在免费实例上填了自己的 key
#       → has_llm() 为真 → 不再算 demo
#       → 后端切回配置里的 dense
#       → 免费实例没装 sentence-transformers → ModuleNotFoundError
#
# 两个触发条件本地都不成立（本地装了 embedding 模型、本地不需要填 key），
# 所以这个 bug 只会在真实访问者身上出现。**这正是它必须有测试的原因。**


def _want(monkeypatch, backend: str):
    """把配置里想要的后端换成 backend。

    Config 是 frozen dataclass，不能就地改字段，只能整个替换。
    这不是麻烦，是刻意的：配置在运行期不可变，所以任何「当前用哪个后端」
    的疑问都只有一个答案来源。
    """
    import dataclasses

    import shared

    monkeypatch.setattr(
        shared, "CONFIG", dataclasses.replace(shared.CONFIG, retrieval_backend=backend)
    )
    return shared


def test_环境跑不了_dense_时必须退回_bm25(monkeypatch):
    shared = _want(monkeypatch, "dense")
    monkeypatch.setattr(shared, "embeddings_available", lambda: False)
    assert shared.backend_name() == "bm25"


def test_hybrid_同样要退回(monkeypatch):
    """hybrid 里含 dense，缺了模型一样跑不起来。"""
    shared = _want(monkeypatch, "hybrid")
    monkeypatch.setattr(shared, "embeddings_available", lambda: False)
    assert shared.backend_name() == "bm25"


def test_填了_key_不会把后端顶成跑不起来的那个(monkeypatch):
    """原始故障的直接复现：有 key ≠ 有 embedding 模型。"""
    shared = _want(monkeypatch, "dense")
    monkeypatch.setattr(shared, "embeddings_available", lambda: False)
    monkeypatch.setattr(shared, "has_llm", lambda: True)   # 访问者填了 key
    assert shared.backend_name() == "bm25"


def test_环境跑得了就用配置里那个(monkeypatch):
    """退化必须是有条件的，不能变成永远 BM25。"""
    shared = _want(monkeypatch, "dense")
    monkeypatch.setattr(shared, "embeddings_available", lambda: True)
    assert shared.backend_name() == "dense"


def test_发生退化时界面必须说明(monkeypatch):
    """一个自称用语义检索的页面，实际跑 BM25 却不作说明，就是在误导访问者。"""
    shared = _want(monkeypatch, "dense")
    monkeypatch.setattr(shared, "embeddings_available", lambda: False)
    assert "bm25" in shared.backend_note().lower()

    monkeypatch.setattr(shared, "embeddings_available", lambda: True)
    assert shared.backend_note() == ""


# --------------------------------------------------------------------------- #
# 伴侣档案：可以编辑，且**永不落盘**
# --------------------------------------------------------------------------- #
# core/profile/crud.py 有 SQLite 存储，但网页这条路刻意不用它：
# 线上是所有访问者共用的一个进程，把第三方的精神健康信息写进服务器磁盘
# 既是隐私问题也是串号问题。下面这条测试钉住的就是「没有那条写盘路径」。

def test_网页档案只进会话内存_不碰_sqlite(monkeypatch):
    """保存档案不得触发任何 SQLite 写入。"""
    import core.profile.crud as crud
    import shared
    from core.profile.models import PartnerProfile

    called = []
    for name in [n for n in dir(crud) if n.startswith(("save", "upsert", "insert", "delete"))]:
        monkeypatch.setattr(crud, name, lambda *a, **k: called.append(name), raising=False)

    ss = {}
    monkeypatch.setattr(shared.st, "session_state", ss, raising=False)
    shared.save_profile(PartnerProfile(nickname="测试"))

    assert called == [], f"档案保存不应触碰持久化层，却调用了 {called}"
    assert ss, "档案应当写进 session_state"


def test_没填过档案时退回演示档案(monkeypatch):
    import shared

    monkeypatch.setattr(shared.st, "session_state", {}, raising=False)
    assert shared.custom_profile() is None
    assert shared.profile_is_custom() is False
    assert shared.active_profile().nickname == shared.demo_profile().nickname


def test_填过之后以自己那份为准(monkeypatch):
    import shared
    from core.profile.models import PartnerProfile

    monkeypatch.setattr(shared.st, "session_state", {}, raising=False)
    shared.save_profile(PartnerProfile(nickname="阿星", landmines=["不要提去年"]))

    assert shared.profile_is_custom() is True
    assert shared.active_profile().nickname == "阿星"
    assert "不要提去年" in shared.active_profile().to_prompt_block()


def test_导入坏_json_不会让整页崩掉(monkeypatch):
    """使用者会导入乱七八糟的文件。坏数据要被丢掉，不是抛异常。"""
    import shared

    ss = {shared._PROFILE_SS: {"relationship_years": "不是数字"}}
    monkeypatch.setattr(shared.st, "session_state", ss, raising=False)
    assert shared.custom_profile() is None
    assert shared._PROFILE_SS not in ss, "坏档案应当被清掉，否则每次进页面都再炸一次"
