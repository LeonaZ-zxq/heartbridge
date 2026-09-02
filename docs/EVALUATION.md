# Evaluation methodology

Everything in this document is reproducible:

```bash
python scripts/eval_retrieval.py --set both --show-failures
python scripts/eval_index_ablation.py
python scripts/eval_safety.py
```

---

## 1. Retrieval

### 1.1 Two evaluation sets, and why the lower number is the real one

| Set | File | n | Written when |
|---|---|---|---|
| Dev | `tests/fixtures/retrieval_eval.json` | 32 | Before document expansion |
| Held-out | `tests/fixtures/retrieval_holdout.json` | 30 | After, independently, without consulting any existing wording |

| Backend | Dev R@1 | Dev R@3 | Dev MRR | Held-out R@1 | Held-out R@3 | Held-out MRR |
|---|---|---|---|---|---|---|
| BM25 | 90.6% | 100.0% | 0.948 | 33.3% | 36.7% | 0.350 |
| **Dense** (bge-small-zh) | 87.5% | 96.9% | 0.917 | **60.0%** | **76.7%** | **0.672** |
| Hybrid (RRF, k=60) | 90.6% | 100.0% | 0.948 | 40.0% | 70.0% | 0.522 |

**The dev result is contaminated and must not be reported as a system capability.**

The contamination is specific and worth naming. Each card carries a `user_phrasings` field — manual document expansion (doc2query), added to close the vocabulary-mismatch gap that lexical retrieval suffers from. Those phrasings were authored *after* inspecting which dev-set queries were failing. That is test-set leakage: the answers were, in effect, written into the index. Dev-set Recall@3 rose from 87.5% to 100%, and roughly all of that rise is leakage.

The held-out set was written afterwards without reference to the phrasings, and consists entirely of paraphrased queries with minimal lexical overlap. It is a deliberately hard set, and it is the honest measurement.

### 1.2 What the gap actually shows

BM25 at 36.7% Recall@3 on paraphrased queries is not a bug in the implementation — it is the known ceiling of lexical matching when documents and queries do not share vocabulary. Character-bigram tokenisation was tried as a mitigation:

| Tokenisation | Dev R@3 | Held-out R@3 | Held-out MRR |
|---|---|---|---|
| Word-level (jieba) | 100.0% | 36.7% | 0.350 |
| Word + character bigrams | 100.0% | 40.0% | 0.367 |

+3.3pp. Recorded as a negative result and **not shipped**, because it adds index size and tokenisation complexity for a gain that does not change the conclusion.

This is the quantified argument for **semantic** retrieval: the failures are semantic, not lexical, so the fix has to be semantic. Dense retrieval lifts held-out Recall@3 from 36.7% to 76.7% — a 40-point improvement, and the single largest measured gain in the project.

`HybridRetriever` remains implemented, and fuses BM25 and dense ranks with Reciprocal Rank Fusion (Cormack et al., 2009), which uses ranks rather than scores and therefore needs no normalisation between two incomparable score scales. It is not the default — see below for why.

### 1.2b Hybrid lost. That was not the expected result.

On the held-out set, dense retrieval alone (76.7% Recall@3) beat the BM25+dense hybrid (70.0%). The dev set had shown the opposite ordering, because on the dev set BM25 is artificially perfect.

The mechanism is straightforward once you look at it. Reciprocal Rank Fusion weights every retriever equally: `score(d) = Σ 1/(k + rank_r(d))`. When one retriever is substantially weaker than the other — BM25 at 36.7% versus dense at 76.7% — the weak retriever's rankings still get a full vote, and they pull correct documents down the fused list. Fusion is only free when the components are comparably strong.

The visible symptom is the Recall@1 column: hybrid sits at 40.0% while dense alone reaches 60.0%. Fusion is actively displacing correct top-1 results with BM25's noise.

**The default backend was therefore changed from `hybrid` to `dense`.** "Hybrid retrieval is always better" is a widely repeated claim that did not survive measurement on this corpus.

One methodological caveat, stated rather than hidden: **this backend choice was made using held-out results**, which is itself a mild form of selection on the test set. A rigorous setup needs three splits — tune on dev, select on validation, report on test. With 30 cards and 62 total queries, splitting three ways is not yet worth it, but the next evaluation set should be a true test set that no decision has touched.

### 1.2c What would actually fix hybrid

Not fusion tuning on the held-out set — that would burn the only clean measurement available. The principled options, in order:
1. **Weighted RRF**, with weights fit on a dev split, so a weak retriever contributes proportionally.
2. **Query-adaptive routing**: send short keyword-like queries to BM25 and long conversational ones to dense.
3. **Strengthen BM25 itself** so fusion has two comparable components. The document expansion already attempted this; the honest reading of the held-out number is that it did not generalise.

### 1.3 Index-text ablation

What text gets embedded/indexed matters more than most RAG tutorials suggest. Measured with BM25 on the dev set, before document expansion:

| Indexed fields | Recall@1 | Recall@3 | MRR |
|---|---|---|---|
| Technique name only | 21.9% | 31.2% | 0.266 |
| **Scenario + aliases + tags** | **78.1%** | **87.5%** | **0.828** |
| Scenario + example phrases | 75.0% | 84.4% | 0.792 |
| Original production config | 78.1% | 84.4% | 0.802 |
| Every field | 71.9% | 87.5% | 0.786 |

Two findings, both of which changed the code:

1. **Adding `example_phrases` — the model's suggested wording — made retrieval worse.** That text belongs to the *answer* distribution. Queries look like the situation, not like the reply. Index text should match the distribution of queries.
2. **Indexing every field kept recall but cost 6 points of Recall@1.** Irrelevant text dilutes the signal: the right card is still retrieved, but ranked worse. This is exactly why Recall and MRR are reported together — Recall@3 alone would have hidden this regression.

### 1.4 Metrics

- **Recall@k** — the downstream generator sees all k retrieved cards, so "is the right card anywhere in the context" is the business-relevant question.
- **MRR** — distinguishes rank 1 from rank 3, catching ranking regressions that leave recall flat.

---

## 2. Crisis detection

41 cases: 14 crisis, 8 elevated, 19 normal (10 of which are adversarial).

| Metric | Result |
|---|---|
| Crisis recall | 100% (14/14) |
| False-positive rate (normal → crisis) | 0% (0/19) |
| Severity downgrades (crisis → lower) | 0 |

### 2.1 Why three metrics rather than accuracy

The three error types differ in cost by orders of magnitude. A missed crisis signal is irreversible. A false positive means the user sees a helpline they did not need. Reporting a single accuracy figure would average across costs that are not comparable, which in a safety context is a reporting failure, not a simplification.

### 2.2 The adversarial subset

Ten of the nineteen normal cases exploit the fact that Chinese uses 死 ("die") as a degree adverb: 累死了 ("dead tired"), 笑死 ("dying of laughter"), 困死了, 饿死了, 死机 ("crashed"). Without idiom masking these all fire as crisis signals, and the false-positive rate makes the system unusable.

This matters beyond tidiness: **alert fatigue is itself a safety failure.** A user who has learned to dismiss the crisis banner will dismiss the real one.

### 2.3 What the first run caught

Crisis recall on the first run was 78.6%. Three real misses:

| Missed case | Why the rule failed |
|---|---|
| 「我发现他手臂上有新的划痕」 | Rules covered self-harm *actions* (割、划自己) but not *evidence* (划痕) |
| 「他说他今天又割了」 | Rules covered 割腕/割手, not the bare verb |
| 「他说要把他的吉他留给我」 | Farewell rule was written as 留给**你** — it did not anticipate the user reporting in third person |

The lesson generalises: safety rules must cover **how users actually describe things**, not the clinical vocabulary for those things. All three are now regression tests.

### 2.4 Invariants under test

These are asserted directly, because they are the design:

- `test_llm_cannot_downgrade_a_rule_crisis` — the LLM's verdict is merged with `max()`, never trusted as final.
- `test_negation_downgrades_but_never_to_zero` — "he said he won't" lowers severity one level, not to none.
- `test_llm_failure_does_not_break_safety_layer` — an LLM outage degrades to the rule layer rather than raising.
- `test_rule_crisis_skips_llm_call` — no wasted latency on a decision the LLM cannot change.
- `test_crisis_template_is_deterministic` — identical input, identical output, always.

Crisis cases are parameterised individually rather than aggregated: in a safety module, "93% passing" conveys nothing useful. You need to know *which* case failed.

---

## 3. Reply generation

### 3.1 Structural guarantees (enforced in tests)

- Every option cites a `card_id` that was actually retrieved; ungrounded citations are dropped.
- Every option carries a `why`; options without one are rejected.
- At most three options are returned.
- An LLM failure returns an empty list rather than raising.

### 3.2 The blind A/B protocol

Structural guarantees say the output is well-formed. They say nothing about whether it is any good. For that:

20 situations, written independently of both retrieval evaluation sets (asserted by a test). For each, replies are generated twice:

| Arm | Setup |
|---|---|
| `grounded` | normal RAG path — retrieved cards injected |
| `ungrounded` | identical model, identical prompt, **no cards at all** |

Both arms are shuffled together into one rating sheet with the arm and the card id stripped out, and rated by hand.

**Primary metric:** *would I actually send this?* — Y/N per option, then the share of situations with at least one usable option. Five diagnostic dimensions (validation, sendability, fidelity, harmlessness, explanation quality) are scored 1–5 to locate *where* a bad option went wrong.

The control arm is the point. "My system scores 4.1 out of 5" is unanchored. "The knowledge base moved the usable-reply rate from X to Y" is a claim about whether the retrieval layer earns its place.

Two details that make the blinding real, both covered by tests:

- The sheet contains neither the arm nor the card id. One rubric dimension was renamed from `groundedness` to `fidelity` purely because the substring `grounded` leaked the arm label into the sheet — the test caught it.
- Shuffling is seeded, so a rating session is reproducible, but ordering does not correlate with arm.

### 3.3 Why LLM-as-judge is a sentinel, not the verdict

A judge model runs the same rubric, because it is cheap enough to re-run after every prompt change. It is deliberately not the acceptance criterion:

1. **Self-preference bias** — models score outputs from their own family higher.
2. **Position bias** — identical content scores higher earlier in a list.
3. **Unverified correlation with human judgement in this domain.** "Is this the right thing to send to this person right now" depends on the relationship and the moment. There is no evidence a judge model proxies that, and this is not a domain in which to assume it.

The judge answers "did this change make things obviously worse". The human blind rating answers "is this good".

### 3.4 Crisis situations are excluded, on purpose

Situations that trigger the crisis branch never reach generation, so scoring them as generation quality would be meaningless. The evaluation script detects and skips them, and reports that it did. One deliberate boundary case (`g20`, giving a pet away — plausibly farewell language, plausibly not) is included so this branch is actually exercised.

---

## 4. Second snapshot: 93 cards (2026-09-01)

Everything above was measured on a 49-card knowledge base. Ingesting Chinese
creator content added 44 communication cards (`lived_experience`), taking the
base to 93. The numbers moved enough to be worth recording as a separate
snapshot rather than overwriting the first — the analyses above are about the
decisions made against *that* base, and rewriting their numbers would erase
the fact that the base changed.

| Backend | Dev R@1 | Dev R@3 | Dev MRR | Held-out R@1 | Held-out R@3 | Held-out MRR |
|---|---|---|---|---|---|---|
| BM25 | 81.2% | 84.4% | 0.828 | 30.0% | 33.3% | 0.311 |
| **Dense** (bge-small-zh) | **87.5%** | **96.9%** | **0.917** | **60.0%** | **70.0%** | **0.644** |
| Hybrid (RRF) | 84.4% | 96.9% | 0.891 | 33.3% | 56.7% | 0.428 |

### 4.1 The dev-set leakage became measurable

Section 2 asserted that BM25's 100% dev-set Recall@3 was "roughly all
leakage" — `user_phrasings` had been authored after inspecting dev-set
failures. That was an argument. Diluting the index turned it into evidence:

| | 49 cards | 93 cards | Δ |
|---|---|---|---|
| BM25 dev R@3 | 100.0% | 84.4% | **−15.6pp** |
| Dense dev R@3 | 96.9% | 96.9% | **0.0pp** |

The 44 new cards' `user_phrasings` were written without looking at any
evaluation set. Adding untuned candidates to the pool cost BM25 15.6 points
on the dev set and cost dense nothing.

> A score obtained through leakage decays when you add samples that were not
> tuned for that set. A score obtained through capability does not.
> **Dilution is a cheap, retrospective test for contamination** — it needs no
> new annotation, only new documents.

This also sharpens *why* BM25 was the leaked one: the leakage was lexical.
`user_phrasings` put the dev queries' exact vocabulary into the index, which
is precisely what a term-matching retriever rewards and what a semantic
retriever is largely indifferent to.

### 4.2 Fusion degraded further, for the predicted reason

Held-out hybrid fell from 70.0% to 56.7%, and hybrid Recall@1 (33.3%) is now
close to BM25 alone (30.0%) rather than to dense (60.0%). Section 2.3 argued
that RRF's equal weighting lets a weak retriever drag the fused ranking down.
As BM25 weakened in absolute terms, the drag got worse — the prediction held
without any change to the fusion code.

### 4.3 The held-out drop was a latent defect, not a regression

Dense held-out Recall@3 fell 76.7% → 60.0% on adding the cards, then
recovered to 70.0% after one change. `--show-failures` showed something
unexpected: of the 12 failures, **only one involved a new card**. The rest
were old-card-versus-old-card confusion.

The real signal was that four somatic queries were being answered with
communication cards. Comparing what actually gets embedded:

```
communication  comm_030   "他早上特别难受，说不出话，到了晚上又好一些" + 5 colloquial phrasings
somatic        soma_001   "胸闷、胸痛、心悸 胸口闷 喘不上气 心跳得好快 … 惊恐发作 急性"
```

The query `他说心口像被石头压着，喘气费劲` shares meaning with the alias
`喘不上气` — **the vocabulary was already right**. What differed was form:
somatic index text is a list of bare noun phrases; communication index text
is a full sentence with a subject, and so is the query.

> This is not vocabulary mismatch. It is **register mismatch**: the same words,
> packed into a different syntactic shape, land in a different region of the
> embedding space.

Fix: `SomaticCard.index_text()` now emits an additional `他说{alias}` variant
for each alias. No data change, no model change, no backend change.

- Held-out Recall@3 60.0% → 70.0%, failures 12 → 9
- The three recovered queries are exactly three of the four somatic ones

Of the 9 remaining failures: 8 are confusions among the original
`comm_001`–`comm_030` (with `comm_029` and `comm_030` each appearing 3 times —
textbook **hubness**, where a few vectors become nearest neighbours to many
queries); 1 is a stale label, where the newly retrieved `comm_063` answers the
query better than the annotated gold `comm_013`.

> Growing the index did not introduce these errors. It enlarged the candidate
> pool until pre-existing weaknesses stopped being survivable.
> **Scaling one part of a system is a stress test for the rest of it.**

### 4.4 The held-out set is now contaminated

The register fix was designed *after* reading held-out failures, then
validated *on* held-out. That is the same error as §2.1, for the third time:

1. `user_phrasings` written from dev-set failures
2. backend selected on the held-out set
3. index design diagnosed and validated on the held-out set

Consequences, stated so they are not quietly forgotten:

- **70.0% is not a publishable retrieval score.** It is evidence that the
  register-mismatch hypothesis was correct, and nothing more.
- Two obvious repairs are **deliberately not being made yet**: re-labelling the
  `comm_013`/`comm_063` query, and adding a dissociation alias so `soma_007`
  is reachable. Both would be authored directly from held-out failures.
- A third evaluation set — annotated without looking at any card, any failure
  list, or any retrieval output — has moved from "nice to have" to blocking.
  There is currently no clean set left to report a number on.

> Once a set has been used to make a design decision, it is no longer an
> evaluation set. It has become part of the training data.
