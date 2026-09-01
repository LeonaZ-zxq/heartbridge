# 文字稿放这里

两种来源：

1. `scripts/ingest.py` 转写视频后自动写到这里（`*.txt`）
2. 你手动整理的文字稿（自己听打、或从别处复制的口播文案）也放这里

手动放的文字稿可以直接蒸馏，跳过转写这一步（调 prompt 时最快）：

    python scripts/ingest.py --transcript-file data/transcripts/xxx.txt

**已被 .gitignore。** 别人的口播内容是别人的作品，不进公开仓库。
