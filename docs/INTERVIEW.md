# HeartBridge · 面试讲解手册

> 这份文档是给项目作者自己看的：每个模块在做什么、为什么这么做、面试官会怎么追问、怎么答。
> **用法**：英文粗体句子是可以直接说出口的话术，建议记熟；中文是理解和展开用的。

---

## 0. 60 秒电梯陈述（先背熟这个）

> **"HeartBridge is a retrieval-grounded assistant for the partner of someone with depression. You paste the conversation you're stuck in, and it returns two or three reply options — each one carrying the mechanism behind it, not just a script.**
>
> **Three things make it an engineering project rather than a prompt wrapper. First, safety is structural: crisis detection is rule-based and short-circuits before retrieval or generation, and the LLM is only allowed to escalate risk, never to lower it. Second, every reply must cite a retrieved knowledge card, and I validate that citation — a citation to something that wasn't retrieved is a hallucination and gets dropped. Third, I built two evaluation sets, discovered my own test-set leakage, and report the lower, honest number."**

如果对方只想听一句：
> **"It's a RAG system where the interesting part isn't the retrieval — it's the safety architecture and the evaluation methodology."**

---

## 1. 项目动机（HR / hiring manager 会问）

**Q: 为什么做这个项目？**

真实回答，不要编：我的伴侣有抑郁症，我们异地。凌晨两点「我该说什么」是一个真实且反复出现的问题，而网上的建议零散、无出处、太笼统。同时我在找 AI 应用方向的实习，需要一个能把 RAG、agent、评测、responsible AI 串起来的项目。

> **"I'm the primary user. That's not a nice story — it's why the evaluation is honest. When you're the one who has to send the message at 2am, you can't fool yourself about whether the output is good."**

**Q: 这不是很敏感吗？你怎么处理伦理问题？**

主动讲三件事（见第 6 节）：定位声明、危机升级路径、本地优先的隐私架构。

---

## 2. 架构（几乎必问）

**Q: 讲讲你的系统架构。**

分三句说：

1. **分层**：所有逻辑在 `core/` 这个纯 Python 包里，没有框架依赖；CLI、以后的 Discord bot、Streamlit 都只是薄调用层。
2. **数据流**：粘贴文本 → 确定性解析 → 危机评估（命中即短路）→ 症状路由 → 检索 → prompt 组装 → 生成 → 引用校验 → 输出。
3. **一个关键顺序**：安全分支在检索和生成**之前**。

> **"The core is a plain Python package with no framework dependency. Every front end is a thin caller. That means business logic is unit-testable without spinning up a bot, and swapping front ends costs nothing."**

**Q（高频追问）: 你为什么不用 LangChain / LlamaIndex？**

不要说"我不会"。说取舍：

> **"For a system this size, a framework would have added an abstraction layer between me and the two things I actually needed to control: what exactly goes into the prompt, and what exactly happens to the model's output. My retrieval is 200 lines and my prompt assembly is explicit. I'd reach for a framework when I need connector breadth or a team convention, not for control."**

**Q: 你原计划用 Hermes agent 框架，为什么放弃了？**

> **"I dropped it because the framework was never load-bearing. All the logic was in `core/` by design, so the bot layer was maybe 20% of the work and swapping it cost nothing. That's the payoff of the layering decision, and it's why I made it up front."**

---

## 3. RAG：最密集的追问区

### 3.1 为什么用 RAG，不把知识全塞进 prompt？

四个理由，按份量排序：

1. **可溯源**：每个建议能指回具体一张卡、卡能指回 Beyond Blue 的具体页面。全塞 prompt 就没有「这条建议来自哪里」这个概念了。
2. **可增长**：知识库要持续从博主视频蒸馏新卡，塞 prompt 的做法每加一张卡都要重新调 prompt。
3. **上下文预算**：30 张卡已经上万 token，塞满会稀释注意力（而且我有消融数据证明**无关文本会伤害排序**，见 3.5）。
4. **成本与延迟**：免费模型上下文有限、限速严格。

> **"Retrieval isn't only about fitting in the context window. It's about provenance. In a mental-health-adjacent product, 'which source does this advice come from' has to be a property of the system, not a claim in the prompt."**

### 3.2 chunking 策略？

> **"I don't chunk. My knowledge unit is a hand-designed card with a fixed schema — scenario, do, don't, example phrases, mechanism — so the retrieval unit and the reasoning unit are the same object. Chunking is what you do when your knowledge arrives as unstructured documents; mine arrives as distilled structure, and that's a deliberate step in the ingestion pipeline."**

这是个加分回答，因为它说明你理解 chunking 是**手段不是目的**。

### 3.3 索引什么文本？（这是我最强的一段，主动往这儿引）

我做了消融实验（`scripts/eval_index_ablation.py`），BM25 后端，32 条查询：

| 索引字段 | Recall@1 | Recall@3 | MRR |
|---|---|---|---|
| 只有技巧名 | 21.9% | 31.2% | 0.266 |
| **场景 + 别名 + 标签** | **78.1%** | **87.5%** | **0.828** |
| 场景 + 例句 | 75.0% | 84.4% | 0.792 |
| 我原本的配置 | 78.1% | 84.4% | 0.802 |
| 全部字段 | 71.9% | 87.5% | 0.786 |

两个改变了代码的发现：

> **"Adding the model's suggested wording — the example phrases — made retrieval worse. That text belongs to the answer distribution. Queries look like the situation, not like the reply. Index text should match the distribution of queries, not the distribution of answers."**

> **"Indexing every field kept recall but cost six points of Recall@1. Irrelevant text dilutes the signal: the right card is still retrieved, just ranked worse. That's exactly why I report Recall and MRR together — Recall@3 alone would have hidden that regression."**

### 3.4 embedding 模型怎么选？

> **"`bge-small-zh-v1.5`. Chinese-optimised, 95MB, runs on CPU. The driver was privacy, not benchmark scores: the user's text is their partner's mental-health information, so it must not leave the machine. That rules out hosted embedding APIs, which rules out anything that doesn't run locally. The constraint chose the model."**

这个回答的价值在于：**约束推导选型**，而不是"我用了大家都用的那个"。

已知局限（主动说，显得诚实且了解自己的系统）：
> **"It's weak on metaphorical symptom descriptions. My held-out set has 'he says the world feels unreal, like there's glass in between' — dense retrieval misses that dissociation card. Small Chinese embedding models don't handle figurative language well. That's a concrete limitation I can point at, not a guess."**

### 3.5 为什么是 hybrid？RRF 是什么？

> **"BM25 and dense retrieval fail differently. BM25 nails rare and exact terms — a specific symptom name — but collapses when the query and the document don't share vocabulary. Dense handles paraphrase but blurs near-neighbours. The failure modes are complementary, so I fuse them."**

**RRF 为什么优于加权求和（这题会筛掉很多人）：**

> **"BM25 scores are unbounded — mine go up to 9 — and cosine similarity lives in [-1, 1]. You can't add them without normalising, and normalising needs tuning and is sensitive to the score distribution. Reciprocal Rank Fusion uses only the ranks: score(d) = Σ 1/(k + rank). It sidesteps the normalisation problem entirely. k=60 is the standard value from Cormack et al. 2009."**

### 3.6 ⭐ 「混合检索总是更好」——我实测它不成立

这是项目里第二强的故事，主动讲。留出集（30 条改写式查询）：

| 后端 | Recall@1 | Recall@3 | MRR |
|---|---|---|---|
| BM25 | 33.3% | 36.7% | 0.350 |
| **Dense** | **60.0%** | **76.7%** | **0.672** |
| Hybrid (RRF) | 40.0% | 70.0% | 0.522 |

> **"On my held-out set, dense alone beat the hybrid — 76.7% versus 70.0%. RRF weights every retriever equally, so when one component is much weaker — BM25 at 36.7% against dense at 76.7% — its rankings still get a full vote and they push correct documents down the fused list. You can see it in Recall@1: hybrid drops to 40% while dense alone gets 60%. Fusion is only free when the components are comparably strong. So I changed the default backend to dense."**

追问「那你为什么还留着 hybrid 的代码？」：
> **"Because the corpus will change. Once the ingestion pipeline adds creator-sourced cards with much more colloquial vocabulary, BM25 gets stronger and fusion may win again. The retriever is a config switch, and the evaluation harness re-runs in seconds — so it's a measurement I can redo, not a decision I have to relitigate."**

追问「怎么修好 hybrid？」（这题答好了很加分）：
> **"Not by tuning fusion weights on the held-out set — that would burn the only clean measurement I have. Weighted RRF with weights fit on a dev split, or query-adaptive routing: short keyword-like queries to BM25, long conversational ones to dense."**

⚠️ **一个要主动承认的方法论瑕疵**（承认它会显得你更强，不是更弱）：
> **"Choosing the backend using held-out results is itself selection on the test set. A rigorous setup needs three splits — tune on dev, select on validation, report on test. With 62 queries total that's not worth splitting three ways yet, but I know the next evaluation set has to be one no decision has touched."**

---

## 4. 评测方法论：这是整个项目的高光

### 4.1 「你怎么知道你的 RAG 好不好？」

> **"I have two evaluation sets and I report the lower one."**

然后讲完整的故事：

1. 我加了 `user_phrasings` 字段——**手工做的文档扩展（doc2query）**，给每张卡补上用户可能怎么口语描述这个处境。目的是补词法检索最大的短板：**词汇不匹配（vocabulary mismatch）**。
2. 开发集 Recall@3 从 87.5% 涨到 **100%**。
3. **然后我停下来了。** 因为我写这些扩展词的时候是看过开发集失败案例的——这是**测试集泄漏**，等于把答案写进了索引。
4. 我另写了一个 30 条查询的**留出集**，独立撰写、不参照任何已有措辞、全部是改写式查询。BM25 的诚实数字是 **36.7%**（对比开发集 100%）。接入 dense 后是 **76.7%**——40 个百分点的提升，是整个项目里最大的一次可测量收益。
5. 我还试了字符 bigram 分词想补救，只提升 3.3 个点，**记录为负面结果，没有为了数字好看而合入**。

> **"My dev-set number was 100%. My held-out number was 36.7%. The 100% was leakage. And that gap is also the quantified justification for semantic retrieval: the failures were semantic, not lexical, so no amount of tokenisation tweaking fixed them. Switching to dense embeddings took held-out Recall@3 from 36.7% to 76.7% — the largest measured gain in the project, and I only knew where to spend that effort because the held-out set told me where the failures actually were."**

**为什么这个故事这么有用**：它同时证明了三件事——你懂评测方法论、你会怀疑自己的结果、你诚实。绝大多数候选人的项目只有一个「效果不错」。

### 4.2 「为什么用 Recall@k 和 MRR？」

> **"Recall@k because the generator sees all k cards — 'is the right card anywhere in the context' is the business-relevant question. MRR because Recall@3 can't tell rank 1 from rank 3, so a ranking regression can hide behind a flat recall number. I've actually hit that: indexing every field kept Recall@3 at 87.5% while Recall@1 dropped six points."**

### 4.3 「LLM 应用怎么做回归测试？」

> **"You don't assert model output. You assert two things: metric floors, and structural invariants."**

- **指标下限**：`test_bm25_meets_recall_floor` 断言 `Recall@3 >= 0.80`，而不是等于 87.5%——知识库会一直加卡，硬编码数值会天天变红，但掉破下限一定是真退化。
- **结构不变量**：引用必须接地、每个选项必须有 why、危机时必须短路、LLM 不能降级风险。这些跟模型说什么无关。
- **确定性 mock**：整套 83 个测试离线、免费、可复现。

---

## 5. 安全设计：面试价值最高的模块

### 5.1 「你怎么处理自杀风险这类高危场景？」

> **"Rules first, LLM second, and the merge is asymmetric — the LLM can only escalate."**

四个理由讲清为什么规则在前：
1. **确定性**：安全关键路径不能依赖会限速、会超时、会幻觉的远程服务。
2. **可审计**：出事能指着具体哪条规则，而不是"模型觉得"。
3. **成本与延迟**：绝大多数消息不需要调 LLM。
4. **LLM 补短板**：不含关键词的隐晦表达。

### 5.2 「为什么 LLM 只能升级不能降级？」（最能加分的一题）

> **"A non-deterministic component is never allowed to silence a safety signal that has already fired. If the LLM could veto a rule hit, then any hallucination — or any prompt injection — becomes a path to suppressing a real crisis signal. The cost asymmetry is the whole argument: a false positive means the user sees a helpline they didn't need; a false negative is irreversible. So the merge is `max()`, not 'trust the smarter component'."**

对应测试：`test_llm_cannot_downgrade_a_rule_crisis`。

### 5.3 「你怎么保证危机检测的召回率？」

数字：41 条评测集（14 危机 / 8 需关注 / 19 普通，其中 10 条对抗样本），**危机召回 100%，误报 0%**。

主动讲第一次跑的失败：

> **"First run was 78.6%. Three real misses. My rules covered self-harm *actions* — 'cut', 'hurt myself' — but not *evidence*: 'there are new marks on his arm'. And my farewell-language rule was written in second person, 'leaving it to you', so it missed the user reporting in third person. The lesson generalises: safety rules have to cover how users actually describe things, not the clinical vocabulary for those things."**

### 5.4 「误报怎么办？」

> **"Ten of my nineteen normal cases are adversarial, because Chinese uses 死 — 'die' — as a degree adverb. 累死了 is 'dead tired'; 笑死 is 'dying of laughter'; 死机 is 'the computer crashed'. Without idiom masking those all fire as crisis signals. And that matters beyond tidiness: alert fatigue is itself a safety failure. A user who's learned to dismiss the crisis banner will dismiss the real one."**

### 5.5 「危机回复为什么是硬编码模板？」

> **"On the highest-risk path, predictability and auditability beat personalisation. Every strength of a generative model becomes a liability there: diversity means some sample might say something harmful, fluency means the harmful thing looks credible, and irreproducibility means you can't audit it after the fact or write a regression test for it. A template can be reviewed word by word by a human and asserted line by line in tests."**

---

## 6. Responsible AI / 隐私（AI 产品方向必问）

**Q: 用户数据怎么处理？**

> **"A partner's mental-health information is the most sensitive category of personal data there is, so the architecture is local-first. Profile and transcripts live in SQLite on one machine. The whole `data/` directory is gitignored. Embeddings run on local CPU, so the user's text never reaches a third party — that requirement is what chose the embedding model. Every profile field is optional, empty fields aren't even injected into prompts, and `delete_profile()` exists with a test covering it. The public demo uses fabricated data only."**

四个关键词，都能落到代码上：**local-first / data minimisation / local inference / erasure**。

**Q: 怎么防止模型在医疗话题上编造？**

> **"I moved the constraint out of the prompt and into the type system. A somatic-symptom card without an authoritative source raises a validation error at load time — it structurally cannot enter the knowledge base. Prompt instructions are best-effort; a model validator is not. And at generation time, every option must cite a card that was actually retrieved; ungrounded citations are dropped before the user sees them."**

**Q: 这个工具会不会让人产生依赖？**

> **"That's why every option ships with a 'why' rather than just a script. If you only hand someone words to copy, they learn nothing and they're back tomorrow. The product goal is that the user needs the tool less over time — so options without an explanation are rejected in code, not just discouraged in the prompt."**

---

## 7. LLM 工程细节

**Q: 你怎么处理模型返回的 JSON 不合法？**

> **"Three-level fallback in `extract_json`: parse directly, strip markdown fences and reparse, then slice from the first brace to the last and reparse. On top of that, `complete_json` retries with exponential backoff — 1s, 2s — because the most common failure with free models is a 429, and retrying immediately just hits it again."**

**Q: 你怎么测一个会调 LLM 的系统？**

> **"I treat the LLM as an I/O boundary, not as business logic. Everything deterministic lives on my side of that boundary and is unit-tested; the model is behind a `MockProvider` that routes canned responses by prompt marker. The whole suite runs offline, free, and reproducibly — which is what makes it usable in CI."**

**Q: 换模型/换供应商要改多少代码？**

> **"One factory function. Model names and thresholds are all in `core/config.py` — nothing is hardcoded, because free models get renamed and deprecated constantly, and thresholds are experimental variables I need to sweep."**

**Q: 温度怎么设的？**

> **"Zero for the crisis classifier — I want the same input to give the same verdict. 0.7 for reply generation, because the three options are supposed to be genuinely different in style; at low temperature you get three paraphrases of the same sentence."**

---

## 8. 局限与下一步（诚实回答，非常重要）

主动说局限，比被问出来强得多。

> **"Three honest limitations. Zero, and I'll name it first because it's methodological: I selected my retrieval backend using held-out results, which is mild selection on the test set. The next evaluation set needs to be one no decision has touched. One: generation quality isn't measured yet — my guarantees there are structural, not qualitative. The plan is a blind rating over 20 held-out situations scoring 'would I actually send this?', which is the only acceptance criterion that matters for this product. Two: held-out retrieval is still weak with lexical-only search; hybrid closes most of it but I want the held-out hybrid number before I claim anything. Three: the knowledge base is 30 cards from English-language Australian clinical sources. The ingestion pipeline for Chinese creator content — audio to Whisper to LLM distillation — is designed but not built."**

**Q: 要支持多用户需要改什么？**

> **"Profile store moves from SQLite to Postgres with row-level tenant isolation — `core/profile/crud.py` is the only file that changes. Embeddings get precomputed and cached instead of recomputed per process. The retriever protocol already abstracts the index, so a real vector store drops in. The safety layer doesn't change, and that's on purpose — it's the part that must stay deterministic under load. The genuinely hard part isn't scale, it's that health data across tenants needs encryption at rest, retention limits and a real erasure path — which is the reason the project is single-user by design right now."**

---

## 9. 行为面试题的项目化答案

**「讲一次你犯的技术错误」**
→ 测试集泄漏那件事。发现过程、修正方式（写留出集）、学到什么（评测集的价值取决于它的独立性）。

**「讲一次你被数据推翻的判断」**
→ 索引文本消融：我以为把整张卡塞进索引更好，实测 Recall@1 掉 6 个点。

**「讲一次你为了正确性放弃了更好看的数字」**
→ 字符 bigram 只提升 3.3 个点，记录为负面结果没有合入；以及坚持对外报 36.7% 而不是 100%。

**「你怎么做技术权衡」**
→ hybrid 用 9 点 Recall@1 换 6 点 Recall@3，因为生成器消费的是 top-k 全部，precision@1 不是业务指标。

---

## 10. 简历项目描述（等 holdout 数字回来后定稿）

```
HeartBridge — Retrieval-grounded communication assistant  ·  Python, RAG, pytest
· Built a RAG system over a 30-card knowledge base distilled from Beyond Blue and
  Healthdirect clinical guidance, with BM25, dense (bge-small-zh) and RRF-hybrid
  retrieval backends selected by measurement rather than default.
· Designed a two-stage crisis detector (deterministic rules + escalation-only LLM
  second pass): 100% crisis recall and 0% false-positive rate on a 41-case set
  including 10 adversarial Chinese idiom samples.
· Built the evaluation harness that surfaced test-set leakage in my own results:
  dev-set Recall@3 of 100% fell to 36.7% on an independently authored held-out set.
  Diagnosing the gap as semantic drove the switch to dense retrieval (36.7% -> 76.7%)
  and showed RRF hybrid underperforming dense alone (70.0%), against expectation.
· Enforced anti-hallucination structurally: generated replies must cite a retrieved
  card and citations are validated; medical-adjacent cards without an authoritative
  source fail schema validation at load time.
· 83 deterministic tests via a mock LLM provider — full suite runs offline and free.
```
