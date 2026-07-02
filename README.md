# Semantic Book Recommender (Hybrid, Gemini-powered)

A semantic book recommender with a **hybrid retrieval pipeline**: it searches a
local vector database of ~5,200 books and can optionally **rewrite the query with
an LLM and fetch fresh results from the Google Books API** on demand. Multilingual
input (e.g. Turkish) is supported through the LLM query rewriter.

This project started from the freeCodeCamp course *"Build a Semantic Book
Recommender with LLMs"* and was extended with a dynamic Google Books module,
an LLM query rewriter, emotion-based sorting, and a caching layer. It uses
**Google Gemini** for embeddings and query rewriting (not OpenAI).

## Features

- **Semantic search** over a Chroma vector database built with Gemini embeddings.
- **Two search modes** (selectable in the UI):
  - **Simple** — local vector database only (fast, offline-ish).
  - **Advanced** — LLM rewrites the query into English Google Books search terms,
    fetches new books from the Google Books API, embeds + ingests them into the
    vector store, then runs the local search. Supports non-English input.
- **Category filter** (Fiction / Nonfiction / …) and **emotional tone sorting**
  (Happy, Surprising, Angry, Suspenseful, Sad) derived from per-book emotion scores.
- **Caching**: query results and fetched books are cached in SQLite to avoid
  repeated API calls (see `data/dynamic_books_cache.db`).

## Architecture

```
gradio-dashboard.py            # Gradio UI + app wiring
build_vector_db.py             # One-time / incremental Chroma index builder
modules/
  hybrid_recommender.py        # Orchestrates local search + external fetch/ingest
  query_rewriter.py            # LLM (Gemini) -> structured Google Books queries
  google_books_client.py       # Google Books API client (parallel multi-query)
  book_normalizer.py           # Volume -> BookRecord cleaning, ISBN parsing
  vector_ingester.py           # Adds embedded books into Chroma
  embeddings_utils.py          # Bounded retry + safe Chroma collection access
  persistence.py               # Thread-safe CSV / tagged_description writes
  book_cache.py                # SQLite cache (queries + books)
  category_mapping.py          # Maps raw categories -> Fiction/Nonfiction
  db_status.py                 # CLI: inspect CSV/SQLite/Chroma consistency
  types.py                     # BookRecord dataclass
  config.py                    # Central configuration constants
```

**Data layers** (kept consistent by the pipeline):
1. `books_with_emotions.csv` — tabular book metadata + emotion scores.
2. `tagged_description.txt` — `"<isbn13> <description>"` lines used for embeddings.
3. `chroma_db/` — the Chroma vector store.
4. `data/dynamic_books_cache.db` — SQLite cache of queries and fetched books.

> Note: `chroma_db/`, `data/*.db`, and `__pycache__/` are **runtime-generated and
> git-ignored**. Clone with an empty vector DB and build it locally (below).

## Setup

Requires **Python 3.11+**.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # + linting, tests, notebooks (optional)
```

### Environment variables

Copy `.env.example` to `.env` and fill in your keys:

```
GOOGLE_API_KEY=...           # required (Gemini embeddings + query rewrite)
GOOGLE_BOOKS_API_KEY=...     # optional but recommended for the Advanced mode
```

- `GOOGLE_API_KEY`: get one at <https://aistudio.google.com/apikey>.
- `GOOGLE_BOOKS_API_KEY`: optional. Without it the Google Books API still works,
  but on a **low anonymous quota** that can cause silent rate-limit failures under
  load. Get one at <https://console.cloud.google.com/apis/credentials>.

## Building the vector database

The app can build the index lazily on first launch, but building it explicitly is
recommended (Gemini's free tier is rate-limited, so this batches + retries):

```bash
python build_vector_db.py
```

This reads `tagged_description.txt`, embeds documents in batches, and populates
`chroma_db/`. It is resumable — re-running skips already-embedded books.

## Running the app

```bash
python gradio-dashboard.py
```

Then open the local URL Gradio prints. Enter a description (any language),
optionally pick a category / tone, choose **Simple** or **Advanced**, and search.

## Inspecting database consistency

```bash
python -m modules.db_status            # full report (needs GOOGLE_API_KEY for Chroma)
python -m modules.db_status --no-chroma
```

## Development

```bash
ruff check .        # lint
ruff format .       # format
pytest              # run tests
```

Config lives in `pyproject.toml`. A `.pre-commit-config.yaml` is provided:

```bash
pip install pre-commit && pre-commit install
```

CI (GitHub Actions, `.github/workflows/ci.yml`) runs ruff + pytest on every push
and pull request against Python 3.11 and 3.12.

## Notebooks

The original data-prep / modeling notebooks are included for reference:

- `data-exploration.ipynb` — text data cleaning
- `vector-search.ipynb` — vector database construction
- `text-classification.ipynb` — zero-shot Fiction/Nonfiction classification
- `sentiment-analysis.ipynb` — emotion extraction (tone scores)

The raw dataset can be downloaded from Kaggle via `kagglehub`; see the notebooks.

## Credits

Based on the freeCodeCamp course *"Build a Semantic Book Recommender with LLMs"*,
extended with a dynamic Google Books module, LLM query rewriting, and a hybrid
recommender.
