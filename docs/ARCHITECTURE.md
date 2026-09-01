# Architecture

## Layering principle

All logic lives in `core/`, a plain Python package with no framework dependency. The CLI — and any future Discord bot or Streamlit UI — is a thin caller.

Three consequences, and they are the reason for the decision:

1. **Testable.** Business logic can be unit-tested without spinning up a bot or a web server.
2. **Front-end agnostic.** Swapping or adding a front end touches no core code. The project originally planned to build on the Hermes agent framework; dropping it in favour of a plain CLI (and later `discord.py`) cost nothing, because the framework was never load-bearing.
3. **Explainable in an interview.** "Where does the business logic live?" has a one-sentence answer.

## Module map

| Module | Responsibility | Key decision |
|---|---|---|
| `core/config.py` | All model names, thresholds, paths | No hardcoded model names anywhere. Free models get renamed and deprecated constantly; and thresholds are experimental variables that need to be swept. |
| `core/utils/llm.py` | One interface, three backends (OpenRouter / Gemini / Mock) | The LLM is treated as an I/O boundary. `MockProvider` makes the whole suite deterministic, offline and free. JSON repair and exponential-backoff retry live here so callers always receive a clean dict. |
| `core/utils/text.py` | Merged-forward transcript → turns | Regex first, LLM only as fallback. 90% of inputs are deterministically parseable; asking a model first would make a solvable problem slow, costly and unreliable. |
| `core/knowledge/schema.py` | Card models + validation | Validate at the edge. Somatic cards structurally require an authoritative source. `index_text()` decides what gets indexed — measured, not guessed. |
| `core/knowledge/retrieval.py` | BM25 / dense / RRF hybrid | Three backends so the choice can be measured. RRF fuses ranks, not scores, so no cross-scale normalisation is needed. |
| `core/knowledge/evaluator.py` | Recall@k, MRR, failure dumps | Without an evaluation set, every "seems accurate" is self-congratulation. |
| `core/safety/` | Two-stage crisis detection + templates | Rules first (deterministic, auditable, offline), LLM second and escalation-only. Crisis responses are templates, never generated. |
| `core/engine/` | Prompt assembly, generation, citation validation, orchestration | Model output is untrusted input: structure-checked, citation-checked, truncated. |
| `core/profile/` | Partner profile + local SQLite | SQLite because the constraints are single-user, local-first, zero-ops. Swap for Postgres if it ever goes multi-tenant; the CRUD interface does not change. |

## Request flow

```
raw transcript
  │
  ├─ parse_transcript()            deterministic; LLM fallback unused in practice
  │
  ├─ assess()                      RULES ──hit──► render_crisis_response()  ◄── SHORT CIRCUIT
  │                                  │                (no retrieval, no generation)
  │                                  └─ LLM second pass, escalate-only, max() merge
  │
  ├─ _looks_somatic()              cheap keyword router → type_filter
  ├─ retriever.search()            top-k, with a fallback if the filter empties results
  ├─ build_user_prompt()           cards + profile (non-empty fields only) + voice few-shot
  ├─ complete_json()               retry + JSON repair
  └─ _validate()                   drop options with no 'why'; flag/drop ungrounded citations
        │
        └─ 2–3 options, each with text + why + card_id
```

## Why the crisis branch comes first

Placing crisis handling *after* generation — letting the model decide whether to surface helplines — would put the safety guarantee inside a non-deterministic component. Short-circuiting first guarantees that when a crisis signal fires, the user sees reviewed content, regardless of model state, model availability, or prompt injection. `test_crisis_short_circuits_before_retrieval_and_generation` asserts that `hits` and `options` are both empty on the crisis path.

## Scaling notes (what would change for multi-user)

| Concern | Today | If it went multi-tenant |
|---|---|---|
| Profile store | Local SQLite (WAL) | Postgres with row-level tenant isolation; `core/profile/crud.py` is the only file that changes |
| Vector index | In-process embeddings over 30 cards | A real vector store once the corpus outgrows memory; the `Retriever` protocol already abstracts this |
| Embeddings | Local CPU, per-process | Precompute once, cache; keep local inference for the privacy property |
| Safety | Same rules, per-request | Unchanged — and the rules layer is the part that must stay deterministic under load |
| Privacy | Local-only, gitignored | The hard part. Health data across tenants needs encryption at rest, retention limits, and a real erasure path. This is the reason the project is single-user by design. |
