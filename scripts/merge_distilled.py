#!/usr/bin/env python3
"""把复核过的蒸馏卡合并进知识库。

    # 看看有什么待合并的，不写任何文件
    python scripts/merge_distilled.py --dry-run

    # 合并全部
    python scripts/merge_distilled.py

    # 只合并某一个来源
    python scripts/merge_distilled.py --only 测试博主_如何陪伴

━━━ 为什么需要这个脚本 ━━━

摄取管线（core/ingestion/pipeline.py）刻意**不会**自动把产出写进知识库：
它写到 data/distilled/，等人工复核。这是对的——LLM 蒸馏出来的卡
直接进检索库，等于把幻觉变成「有出处的知识」。

但这个设计有一个没被补上的缺口：**复核完之后没有合并的路径**。
结果是用户辛苦摄取的视频素材躺在 data/distilled/ 里，
而 app 只读 knowledge_base/cards/，检索永远看不到它们——
表现出来就是「我明明喂了素材，回复里却一点都没用上」。

这个脚本就是那条缺失的路径。它做三件事：
1. id 去重：跟现有知识库撞 id 的直接拒绝（load_cards 本身也会查，但那时已经晚了）
2. schema 校验：走 parse_card，坏卡在进库前就被挡住
3. 可回滚：合并进的是**单独的文件**，不动原有卡片文件，删掉即可撤销
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import CONFIG  # noqa: E402
from core.knowledge.schema import load_cards, parse_card  # noqa: E402

MERGED_PREFIX = "merged_"


def _existing_ids() -> set[str]:
    """知识库里已经占用的 id。"""
    if not CONFIG.cards_dir.exists():
        return set()
    return {c.id for c in load_cards(CONFIG.cards_dir)}


def main() -> int:
    ap = argparse.ArgumentParser(description="把 data/distilled/ 的卡合并进知识库")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写文件")
    ap.add_argument("--only", help="只合并这个来源（data/distilled 下的文件名，不含 .json）")
    args = ap.parse_args()

    src_dir = CONFIG.private_dir / "distilled"
    if not src_dir.exists():
        print(f"没有 {src_dir}/ —— 先跑 scripts/ingest.py 摄取素材。")
        return 1

    files = sorted(src_dir.glob("*.json"))
    if args.only:
        files = [f for f in files if f.stem == args.only]
    if not files:
        print(f"{src_dir}/ 里没有待合并的卡。")
        return 1

    taken = _existing_ids()
    print(f"知识库现有 {len(taken)} 张卡。待合并文件 {len(files)} 个。\n")

    ok_files: list[tuple[Path, list[dict]]] = []
    total_new = 0
    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"  ✗ {f.name}：JSON 坏了（{e}）——跳过")
            continue
        items = payload if isinstance(payload, list) else [payload]

        good: list[dict] = []
        for raw in items:
            cid = raw.get("id", "<无 id>")
            try:
                parse_card(raw)          # schema 校验，坏卡在这里被挡住
            except Exception as e:
                print(f"  ✗ {f.name} / {cid}：schema 不合格（{e}）——跳过这张")
                continue
            if cid in taken:
                print(f"  ✗ {f.name} / {cid}：id 已被占用——跳过这张")
                continue
            taken.add(cid)               # 同一批里也不许自撞
            good.append(raw)

        if good:
            ids = [c["id"] for c in good]
            print(f"  ✓ {f.name}：{len(good)} 张可合并 [{ids[0]} … {ids[-1]}]")
            ok_files.append((f, good))
            total_new += len(good)

    if not total_new:
        print("\n没有任何可合并的卡。")
        return 1

    print(f"\n合计 {total_new} 张新卡。")
    if args.dry_run:
        print("（--dry-run，没有写任何文件）")
        return 0

    CONFIG.cards_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for f, good in ok_files:
        dest = CONFIG.cards_dir / f"{MERGED_PREFIX}{f.name}"
        dest.write_text(
            json.dumps(good, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written.append(dest)
        # 归档，避免下次重复合并
        archive = f.with_suffix(".json.merged")
        shutil.move(str(f), str(archive))

    # 最后整体加载一次：确认合并后的知识库仍然是可加载的
    try:
        final = load_cards(CONFIG.cards_dir)
    except Exception as e:
        for p in written:
            p.unlink(missing_ok=True)
        print(f"\n✗ 合并后知识库加载失败，已回滚：{e}")
        return 1

    print(f"✓ 已写入 {len(written)} 个文件，知识库现在共 {len(final)} 张卡。")
    print("\n下一步：")
    print("  1. git add knowledge_base/cards/ && git commit && git push")
    print("     —— 不提交的话，Streamlit Cloud 上部署的还是旧知识库")
    print("  2. 重启 app（本地 Ctrl+C 重跑 / Cloud 上 Reboot app）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
