# 蒸馏出来的候选卡片放这里

`scripts/ingest.py` 的产出。**不会自动进知识库**，这是刻意的：
模型从口播里提取的东西必须人看过才算数。

复核流程：

1. 打开 `cards_*.json`，逐张读
2. 检查：scenario 写的是不是「求助者会怎么描述处境」；do/dont 是不是
   文字稿里真的说过；有没有编造医学说法
3. `review_*.json` 里是**疑似重复**的卡（检索相似度超阈值），
   决定是合并还是丢弃
4. 确认要留的，手工合并进 `knowledge_base/cards/`
5. 重跑 `pytest` 和 `python scripts/eval_retrieval.py --backends bm25 --set both`

**已被 .gitignore。**
