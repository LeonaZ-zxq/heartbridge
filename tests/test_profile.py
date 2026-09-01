"""Profile 存储测试。包含隐私相关的行为断言。"""
from core.profile.crud import delete_profile, load_profile, save_profile
from core.profile.models import PartnerProfile


def test_roundtrip(tmp_path):
    db = tmp_path / "t.db"
    p = PartnerProfile(nickname="小鱼", triggers=["工作压力"], long_distance=True)
    save_profile(p, db)
    got = load_profile("default", db)
    assert got and got.nickname == "小鱼" and got.triggers == ["工作压力"]


def test_upsert_overwrites(tmp_path):
    db = tmp_path / "t.db"
    save_profile(PartnerProfile(nickname="A"), db)
    save_profile(PartnerProfile(nickname="B"), db)
    assert load_profile("default", db).nickname == "B"


def test_missing_profile_returns_none(tmp_path):
    assert load_profile("nope", tmp_path / "t.db") is None


def test_delete_actually_removes(tmp_path):
    """数据主体有权随时删除自己的数据。这不是可选功能，所以有测试。"""
    db = tmp_path / "t.db"
    save_profile(PartnerProfile(nickname="小鱼"), db)
    delete_profile("default", db)
    assert load_profile("default", db) is None


def test_prompt_block_omits_unset_fields():
    p = PartnerProfile(nickname="小鱼")
    block = p.to_prompt_block()
    assert "小鱼" in block
    assert "诊断" not in block and "雷区" not in block
