# Running and deploying HeartBridge

## Local (full functionality)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install sentence-transformers      # dense retrieval
cp .env.example .env                   # then add an API key
```

`.env`:

```
HB_LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=...
HB_RETRIEVAL_BACKEND=dense
```

```bash
streamlit run streamlit_app/app.py     # web UI
python scripts/advise.py --demo --text "..."   # CLI
```

## Public demo (Streamlit Community Cloud)

1. Push to GitHub.
2. share.streamlit.io → sign in with GitHub → **New app**.
3. Repository `LeonaZ-zxq/heartbridge`, branch `main`, main file `streamlit_app/app.py`.
4. Deploy. Do **not** add any secrets.

### Why the public demo deliberately has no API key

With no provider configured the app runs in demo mode: retrieval, crisis
detection, the card library and the somatic guide are all genuinely executing,
while reply text comes from a fixed sample rather than a model. The banner on
the page says so.

Three reasons this is the right default rather than a limitation:

1. **Cost and abuse.** A public endpoint with a live key is an open invoice.
2. **Safety.** A public deployment that generates free-form text about
   mental health, with no rate limiting and no way to see who is using it, is
   not something to put on the internet casually. The parts that are safe to
   expose — sourced knowledge, deterministic crisis handling — are exposed.
3. **Privacy.** The demo carries only fabricated profile data. Real profiles
   never leave the user's machine, so there is nothing to deploy.

### Why the demo forces the BM25 backend

The free tier cannot hold `torch` plus an embedding model in memory. Backend
selection is a config value, and `shared.backend_name()` pins the deployed
build to BM25 and states so on the design page. Held-out retrieval is
correspondingly weaker there (36.7% vs 76.7%) — the page does not pretend
otherwise.

## Continuous integration

`.github/workflows/tests.yml` runs on every push:

1. `pytest -q` on Python 3.10 and 3.12
2. retrieval evaluation on both the dev and held-out sets
3. crisis-detection evaluation — **this step exits non-zero on any missed
   crisis case**, so a safety regression fails the build

No API key, no GPU and no network access to model hosts are required, because
the LLM sits behind a deterministic mock and the default retrieval path is
pure Python.
