# BEAM Benchmark — Architecture & Implementation Guide

BEAM (Benchmark for Evaluating Agentic Memory) is an ICLR 2026 benchmark that tests memory recall across 100 conversations in four token-size buckets (100K, 500K, 1M, 10M tokens), with 20 probing questions per conversation spanning 10 memory ability types. This document explains how the pipeline works end-to-end and what you need to change to plug in a different memory provider.

---

## Table of Contents

1. [High-level architecture](#1-high-level-architecture)
2. [Dataset format and download](#2-dataset-format-and-download)
3. [Chat parsing: batches and chunks](#3-chat-parsing-batches-and-chunks)
4. [Timestamp extraction and preservation](#4-timestamp-extraction-and-preservation)
5. [Ingestion into mem0](#5-ingestion-into-mem0)
6. [Checkpointing and resumability](#6-checkpointing-and-resumability)
7. [Search, answer generation, and LLM judging](#7-search-answer-generation-and-llm-judging)
8. [Scoring: nuggets, cutoffs, and metrics](#8-scoring-nuggets-cutoffs-and-metrics)
9. [How mem0 is used (API details)](#9-how-mem0-is-used-api-details)
10. [The OSS server layer](#10-the-oss-server-layer)
11. [Running BEAM against a different memory provider](#11-running-beam-against-a-different-memory-provider)
12. [Key configuration reference](#12-key-configuration-reference)

---

## 1. High-level architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         run.py (orchestrator)                   │
│                                                                 │
│  Phase 1: INGESTION                Phase 2: EVAL                │
│  ┌──────────────────┐              ┌──────────────────────────┐ │
│  │ HuggingFace      │              │ For each question:       │ │
│  │ Dataset (cached) │              │                          │ │
│  │ beam_{size}.json │              │  1. Search memory store  │ │
│  └────────┬─────────┘              │  2. Generate answer      │ │
│           │                        │     (answerer LLM)       │ │
│           ▼                        │  3. Judge each nugget    │ │
│  parse_beam_chat()                 │     (judge LLM)          │ │
│  batch_to_chunks()                 │  4. Kendall tau-b        │ │
│  get_time_anchor_epoch()           │     (event_ordering)     │ │
│           │                        └──────────┬───────────────┘ │
│           ▼                                   │                 │
│  mem0.add(messages,                           ▼                 │
│    user_id, timestamp)            compute_beam_metrics()        │
│           │                       save unified JSON             │
│           ▼                                                     │
│  IngestionCheckpoint                                            │
│  (per-chunk progress)                                           │
└─────────────────────────────────────────────────────────────────┘
             │                              │
             ▼                              ▼
    ┌─────────────────┐          ┌──────────────────┐
    │  Mem0Client     │          │    LLMClient      │
    │  (oss | cloud)  │          │  (OpenAI /        │
    │                 │          │   Anthropic /     │
    │  OSS:  REST →   │          │   Azure)          │
    │  localhost:8888 │          └──────────────────┘
    │                 │
    │  Cloud: REST →  │
    │  api.mem0.ai    │
    │  (V3 + polling) │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Memory store   │
    │  (Qdrant OSS /  │
    │   Mem0 Cloud)   │
    └─────────────────┘
```

Two entirely separate phases run sequentially for each conversation:

- **Phase 1 — Ingestion**: parse and chunk the raw chat, push every chunk into the memory store together with the batch's timestamp.
- **Phase 2 — Eval**: for each of the 20 probing questions, search the memory store, generate an answer, and judge it against rubric nuggets.

---

## 2. Dataset format and download

**Source**: HuggingFace — `Mohammadta/BEAM` (100K / 500K / 1M) and `Mohammadta/BEAM-10M` (10M).

Each conversation record contains:

| Field | Description |
|---|---|
| `conversation_id` | Unique identifier |
| `conversation_seed` | Metadata (category, persona, etc.) |
| `user_profile` | Synthetic user profile used to generate the conversation |
| `chat` | The raw dialogue (format varies by size — see §3) |
| `probing_questions` | Dict keyed by question type → list of question dicts |

The dataset is downloaded once via the `datasets` library and cached locally as `datasets/beam/beam_{size}.json`. Subsequent runs load the cached file directly.

```python
# run.py:97 — download_dataset()
ds = hf_load(HF_DATASET_NAME, split=HF_SPLIT_MAP[size])
# result cached to: datasets/beam/beam_{size}.json
```

---

## 3. Chat parsing: batches and chunks

BEAM conversations are long (100K–10M tokens) and are pre-organized into **batches** (sessions of dialogue that share a time anchor). The parser handles three distinct HuggingFace storage formats:

```
Format 1 (100K / 500K / 1M) — 2D list of turn lists:
  chat = [[turn, turn, ...], [turn, turn, ...], ...]
         ^── batch 0         ^── batch 1

Format 2 (10M) — plan-based nested dicts:
  chat = [{"plan-0": [{"turns": [...]}, ...], "plan-1": [...]}]
          ^── session dict with plan keys

Format 3 (batch-dict) — list of dicts with "turns" key:
  chat = [{"turns": [turn, turn, ...]}, ...]
```

`parse_beam_chat()` (run.py:192) normalizes all three formats into a flat `list[list[dict]]` — a list of batches, each batch being a list of turn dicts.

### Chunking for ingestion

Each batch is then sliced into **chunks of 2 turns** (`CHUNK_SIZE = 2`):

```
batch (N turns)  →  chunk_0 [turn_0, turn_1]
                     chunk_1 [turn_2, turn_3]
                     ...
```

`batch_to_chunks()` (run.py:255) also normalizes role labels (`human` → `user`, anything non-`assistant` → `user`) so mem0 receives standard `user`/`assistant` message pairs.

### Data hierarchy

```
conversation
└── batch[]        ← one "session day"; all turns share a time_anchor
    └── chunk[]    ← CHUNK_SIZE consecutive turns; unit of mem0.add()
        └── turn   ← individual user/assistant message
```

Chunks never cross batch boundaries — `batch_to_chunks()` is called per batch, so a chunk is always fully within one session day and all its turns share the same ingestion timestamp.

### Why CHUNK_SIZE = 2?

`CHUNK_SIZE` is not mem0-specific; it is a tradeoff any extraction-based memory system faces:

- **Too small (1 turn)**: a single message often lacks enough context to extract a meaningful fact. "Yes, I'd love that!" is uninterpretable without the preceding turn.
- **Too large (full batch)**: risks exceeding the LLM context window used for extraction, and tends to produce noisier or less precise memories.

Two turns (one exchange: question + answer, or statement + response) is the minimal conversational unit that gives extraction enough context. For a provider with a larger context window, or one that stores raw messages rather than extracting facts, you could increase `CHUNK_SIZE` or skip chunking entirely and send a full batch per call.

---

## 4. Timestamp extraction and preservation

Every batch in the BEAM dataset includes `time_anchor` fields on turns — ISO-8601-like date strings indicating when that batch occurred in the simulated conversation timeline (e.g., `"2023-03-15"`).

```python
# run.py:275 — get_time_anchor_epoch()
def get_time_anchor_epoch(turns: list[dict]) -> int | None:
    for turn in turns:
        anchor = turn.get("time_anchor")
        if anchor:
            dt = dateparse(anchor.replace("-", " "))
            return int(dt.timestamp())  # unix epoch
    return None
```

The epoch integer is passed to `mem0.add()` as the `timestamp` parameter. This tells mem0 when these memories were "observed", enabling temporal reasoning questions to be answered correctly. Without it, all memories would appear to have been added at ingestion time (today), making the temporal ordering meaningless.

In the OSS server, `timestamp` is forwarded in the add payload. In the cloud V3 API, it is also accepted as-is.

---

## 5. Ingestion into mem0

```python
# run.py:343 — ingest_conversation()
user_id = f"beam_{chat_size}_{conv_idx}_{run_id}"

for batch_idx, batch_turns in enumerate(batches):
    chunks = batch_to_chunks(batch_turns)
    time_epoch = get_time_anchor_epoch(batch_turns)

    for chunk_idx, messages in enumerate(chunks):
        response = await mem0.add(messages, user_id, timestamp=time_epoch)
```

Each conversation gets a **unique, deterministic `user_id`**. The run_id component ensures that separate benchmark runs don't collide in the memory store, while `beam_{size}_{idx}` allows resuming a run to recover the same user_id from the ingestion checkpoint.

The response from `mem0.add()` contains the list of memory operations performed (ADD, UPDATE, DELETE) — logged to a per-conversation debug file at `output_dir/debug/beam_{size}_{idx}_ingestion.txt` when `--debug` is set.

---

## 6. Checkpointing and resumability

Two checkpoint classes (utils.py) handle crash recovery:

### `IngestionCheckpoint`

Saves two files per conversation key (`{size}_{conv_idx}`):

- `_progress_{key}.json` — updated after every successful chunk, contains the set of completed chunk keys and the `user_id`.
- `_ingestion_{key}.json` — written once when all chunks complete; the presence of this file marks the conversation as fully ingested.

On restart, the runner loads `_progress_{key}.json` and skips already-done chunks; if `_ingestion_{key}.json` exists, it skips the entire conversation.

### Question-level checkpoints

Each processed question is saved to `output_dir/{question_id}.json`. On resume, the runner loads all existing JSON files and builds an `existing_ids` set; any question already in that set is skipped.

---

## 7. Search, answer generation, and LLM judging

`process_question()` (run.py:662) handles the full eval pipeline for one question.

### Step 1 — Search

```python
search_results = await mem0.search(question_text, user_id, top_k=top_k)
# top_k default: 200 — retrieve generously, evaluate at multiple cutoffs
```

Results are returned sorted by score descending. The search call is timed and the latency recorded.

### Step 2 — Answer generation (at each cutoff)

The default cutoff is `top_100`, meaning only the top-100 retrieved memories are passed to the answerer. Multiple cutoffs can be evaluated in one pass:

```python
for c in cutoffs:           # e.g., [100]
    sliced = formatted[:c]

    # Sort chronologically before sending to the answerer
    sliced = sorted(sliced, key=lambda m: m.get("created_at", "") or "")

    gen_prompt = get_beam_answer_generation_prompt(question_text, sliced, top_k=c)
    generated_answer = await answerer.generate(system="", user=gen_prompt)
```

The answerer prompt (prompts.py:29 / 104) includes nine rules instructing the model to scan all memories, combine cross-references, prefer newer memories on contradiction, and not invent information. Memories are numbered and prefixed with their `[YYYY-MM-DD]` date string when available.

### Step 3 — Nugget judging

Each rubric has one or more **nuggets** — atomic facts that the answer should contain. Every nugget is judged independently by the judge LLM:

```python
for nugget in rubric:
    ns = await judge_single_nugget(question, nugget, generated_answer, judge_llm)
    # returns {"score": 0.0|0.5|1.0, "reason": "..."}
```

The judge prompt (prompts.py:170) asks the LLM to:
1. Classify the nugget as a positive requirement or negative constraint.
2. Parse any compound sub-requirements.
3. Apply semantic, numeric, and style tolerance rules.
4. Return `{"score": 0.0|0.5|1.0, "reason": "..."}` as JSON.

The judge uses `generate_structured()` which forces JSON output mode (OpenAI `response_format=json_object`, Anthropic via system prompt instruction).

### Step 4 — Event ordering (Kendall tau-b, optional)

For questions of type `event_ordering`, an additional three-step pipeline runs:

```
LLM answer
    │
    ▼
[Step 1] Extract ordered events from answer
         → JSON list of event strings
    │
    ▼
[Step 2] Align each extracted event to a rubric event
         → predicted index (0-based) for each event
    │
    ▼
[Step 3] Compute Kendall tau-b
         predicted_indices vs reference_order=[0,1,2,...]
         τ_b ∈ [-1, 1]  → normalized to [0,1] for combination with nugget score
```

The final `score_with_tau` for event_ordering questions averages the nugget score and the normalized tau-b.

---

## 8. Scoring: nuggets, cutoffs, and metrics

### Score per question

```
question_score = mean(nugget_scores)   # each nugget: 0.0, 0.5, or 1.0
PASS if question_score >= 0.5
```

### Cutoff results

For each cutoff label (e.g., `top_100`), the `cutoff_results` dict records:
- `score` — mean nugget score
- `judgment` — "PASS" / "FAIL"
- `generated_answer` — full text
- `memories_evaluated` — number of memories passed to answerer
- `nugget_scores` — per-nugget breakdown

### Aggregate metrics

`compute_beam_metrics()` (run.py:810) computes at each cutoff:
- Overall accuracy (% questions with score ≥ 0.5)
- Overall avg_score
- Per-question-type breakdown (accuracy + avg_score for each of the 10 types)

### The 10 question types

| Type | What it tests |
|---|---|
| `abstention` | Withhold answer when evidence is absent |
| `contradiction_resolution` | Detect and reconcile conflicting statements |
| `event_ordering` | Reconstruct chronological sequence of events |
| `information_extraction` | Recall specific entities, dates, numbers |
| `instruction_following` | Sustained adherence to user-specified constraints |
| `knowledge_update` | Revise stored facts when new information appears |
| `multi_session_reasoning` | Integrate evidence from non-adjacent dialogue segments |
| `preference_following` | Adapt to evolving user preferences |
| `summarization` | Abstract and compress dialogue content |
| `temporal_reasoning` | Reason about time relations, durations, sequences |

---

## 9. How mem0 is used (API details)

`Mem0Client` (common/mem0_client.py) abstracts two backends behind one async interface.

### OSS mode (default)

Target: self-hosted FastAPI server at `http://localhost:8888` (or `MEM0_HOST`).

```
POST /memories
Body: {"messages": [...], "user_id": "...", "timestamp": 1678886400}
Response: {"results": [{"memory": "...", "event": "ADD"}, ...]}

POST /search
Body: {"query": "...", "user_id": "...", "limit": 200}
Response: {"results": [{"memory": "...", "score": 0.87, "id": "...", "created_at": "..."}, ...]}

DELETE /memories?user_id=...
```

The client normalizes OSS search results to a consistent dict format including `memory`, `score`, `id`, `created_at`, and optional `score_debug` (semantic / BM25 / entity boost breakdown).

### Cloud mode

Target: `https://api.mem0.ai` with `Authorization: Token {api_key}`.

```
POST /v3/memories/
Body: {"messages": [...], "user_id": "...", "timestamp": 1678886400}
Response: {"event_id": "evt_abc123"}

# Poll until SUCCEEDED or FAILED:
GET /v1/event/evt_abc123/
Response: {"status": "SUCCEEDED", "results": [...]}

POST /v3/memories/search/
Body: {"query": "...", "filters": {"user_id": "..."}, "top_k": 200}
Response: {"results": [...]}
```

The cloud add is asynchronous — the server processes memory extraction in the background and the client polls until done (up to `event_poll_timeout=300s`, checking every `event_poll_interval=0.5s`).

Both modes apply the same retry logic (5 attempts, exponential backoff starting at 5s) and use `aiohttp` for async HTTP.

---

## 10. The OSS server layer

`docker/mem0/main.py` is a thin FastAPI wrapper around the `mem0ai` Python SDK. It is configured via `/app/config.yaml` (mounted from the host) or environment variables.

### Stack

```
benchmark runner
      │  HTTP
      ▼
FastAPI server (port 8888)
      │  Python API
      ▼
mem0ai SDK
  ├── LLM: OpenAI / Azure / Ollama  (fact extraction)
  ├── Embedder: OpenAI / SageMaker  (vector embeddings)
  └── Vector store: Qdrant (port 6333)
```

The SDK does the heavy lifting: given a list of messages, it uses the configured LLM to extract atomic facts, compares them with existing memories (by embedding similarity), and decides whether to ADD, UPDATE, or DELETE entries in Qdrant.

### Config loading

The server loads `configs/*.yaml` (mapped into the container), expands `${ENV_VAR}` placeholders, and injects the Qdrant connection automatically if `vector_store` is not already specified. Example:

```yaml
# configs/openai.yaml
llm:
  provider: openai
  config:
    model: gpt-4o-mini
embedder:
  provider: openai
  config:
    model: text-embedding-3-small
    embedding_dims: 1536
```

---

## 11. Running BEAM against a different memory provider

The benchmark is coupled to mem0 through exactly one class: `Mem0Client`. Replacing it with a different provider requires implementing the same three async methods:

```python
class MyProviderClient:
    async def __aenter__(self): ...
    async def __aexit__(self, *exc): ...

    async def add(
        self,
        messages: list[dict],  # [{"role": "user"|"assistant", "content": "..."}]
        user_id: str,
        timestamp: int | None = None,  # unix epoch of the conversation batch
        **kwargs,
    ) -> dict | None:
        # Ingest the message chunk. Return {"results": [...]} or None on failure.
        ...

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        **kwargs,
    ) -> list[dict]:
        # Return results sorted by score descending.
        # Each result: {"memory": str, "score": float, "id": str, "created_at": str}
        ...

    async def delete_user(self, user_id: str) -> bool:
        # Delete all memories for this user. Return True on success.
        ...
```

Then in `async_main()` (run.py:991), replace the `Mem0Client(...)` instantiation:

```python
# Before:
mem0 = Mem0Client(mode=backend, host=args.mem0_host, ...)

# After:
mem0 = MyProviderClient(...)
```

### What must the replacement do?

| Concern | Details |
|---|---|
| **User isolation** | The benchmark assigns each conversation a unique `user_id`. Your provider must support per-user memory isolation so searches return only that conversation's memories. |
| **Timestamp ingestion** | The `timestamp` (unix epoch) should be stored with each memory. It is used by the answerer prompt to render `[YYYY-MM-DD]` dates for temporal questions. If your provider doesn't support it, temporal reasoning scores will degrade. |
| **Memory extraction** | The pipeline passes raw `user`/`assistant` message pairs and expects the provider to extract atomic memories. If your system stores raw messages instead, retrieval quality will differ from mem0's distilled-fact approach. |
| **Return format** | `search()` must return dicts with at least `memory` (str) and `score` (float). `created_at` is optional but enables chronological sorting in the answerer prompt. |
| **Async** | The runner is fully async (`asyncio`). Your client must use `async`/`await` throughout. |

### Elasticsearch / Elastic approach (example sketch)

A minimal Elasticsearch-backed implementation would:
1. On `add()`: call an LLM to extract facts from the messages, then `POST` each fact to an Elasticsearch index as a document with `user_id`, `created_at` (derived from `timestamp`), and the fact text embedded via a dense_vector field.
2. On `search()`: run a kNN search (or hybrid BM25+kNN) filtered by `user_id`, return hits as `{"memory": hit._source.text, "score": hit._score, "id": hit._id, "created_at": hit._source.created_at}`.
3. On `delete_user()`: `DELETE /index/_query` with a `term` filter on `user_id`.

---

## 12. Key configuration reference

### CLI flags

```
--project-name       Name for this run (required)
--chat-sizes         100K,500K,1M,10M (default: 100K)
--conversations      0-99 or 0,1,5 (default: 0-99)
--top-k              How many memories to retrieve per question (default: 200)
--top-k-cutoffs      Evaluation cutoffs, comma-separated (default: 100)
--answerer-model     Model for answer generation (default: gpt-5)
--judge-model        Model for rubric judging (default: gpt-5)
--provider           LLM provider: openai | anthropic | azure
--backend            Memory backend: oss | cloud (default: oss)
--mem0-host          Override mem0 server URL
--mem0-api-key       Cloud API key
--predict-only       Ingest + search only, skip answer + judge
--evaluate-only      Load existing predictions and compute metrics
--resume             Skip already-completed questions and ingestion
--debug              Write verbose debug files per conversation
--rpm                LLM requests per minute rate limit (default: 200)
```

### Environment variables

```
MEM0_HOST            OSS server URL (default: http://localhost:8888)
MEM0_API_KEY         Cloud API key
MEM0_ORGANIZATION_ID Cloud organization ID
MEM0_PROJECT_ID      Cloud project ID
MEM0_BACKEND         "oss" or "cloud" (overrides --backend)
OPENAI_API_KEY       For LLM calls (answerer + judge)
ANTHROPIC_API_KEY    When --provider anthropic
AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY  For Azure provider
```

### Output structure

```
results/beam/
├── predicted_{project_name}/
│   ├── _ingestion_{size}_{idx}.json    # ingestion complete marker
│   ├── _progress_{size}_{idx}.json     # partial ingestion progress
│   ├── {size}_{idx}_q{n}_{type}.json   # per-question result
│   └── debug/
│       └── beam_{size}_{idx}_ingestion.txt
└── beam_results_{timestamp}.json       # unified result with all metrics
```
