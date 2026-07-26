## Professional Methodology: Puls-Events RAG POC

---

### Step 1 — Repository & Project Governance

**Create your Git repository first. Everything lives in version control from day one.**

```
puls-events-rag/
├── src/
│   ├── ingestion/
│   ├── processing/
│   ├── vector_store/
│   ├── rag/
│   └── app/
├── tests/
├── data/
│   └── raw/
├── reports/
├── slides/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

**Immediate actions:**
- Add `.env.example` with placeholder keys
- Add `.gitignore` covering `.env`, `data/`, `__pycache__`, `*.idx`
- Create `dev` branch — never develop on `main`
- Write your README skeleton now; fill it as you build

---

### Step 2 — Environment Setup

**Create a clean virtual environment and pin every dependency immediately.**

```
langchain
langchain-mistralai
faiss-cpu
mistralai
python-dotenv
pandas
requests
ragas
streamlit
datasets
pytest
```

**Validation checkpoint:** Write a single `check_imports.py` that imports every library. If it runs cleanly, your environment is certified reproducible.

---

### Step 3 — Open Agenda Data Ingestion

**Understand the API before writing code.**

Open Agenda exposes a public REST API. Your ingestion strategy:

- Query by `location` (Caen / Normandie region code)
- Query by `timings` (from 12 months ago to future)
- Handle pagination — the API returns results in pages
- Store raw responses as JSON in `data/raw/` immediately
- Never transform in the same script that fetches

**Key fields to extract per event:**
- `uid` — unique identifier
- `title` — event name
- `description` — main text for vectorization
- `location.name` + `location.city`
- `timings` — list of start/end datetimes
- `slug` / `url` — source link

**Separation of concerns:**

| File | Responsibility |
|---|---|
| `fetch_events.py` | API calls only, saves raw JSON |
| `filter_events.py` | Date and location validation only |
| `clean_events.py` | Field extraction, null handling |

**Validation checkpoint:** After ingestion, print a summary — total events fetched, date range covered, events dropped due to missing fields.

---

### Step 4 — Unit Tests (Write These Now, Not Last)

**Your unit tests validate the data pipeline. Write them immediately after ingestion.**

Three tests are required by Jeremy:

**Test 1 — Date filter:**
Every event in your dataset must have at least one timing within the past 12 months or in the future.

**Test 2 — Location filter:**
Every event must be associated with the Normandie region.

**Test 3 — Data integrity:**
No event enters the pipeline with a null or empty description.

Run with `pytest tests/` — this must pass before you touch the vector store.

---

### Step 5 — Text Processing & Chunking

**Decide your chunking strategy based on your data, not on defaults.**

Event descriptions are typically short (100–400 words). This influences your choices:

| Decision | Recommendation for this use case | Reason |
|---|---|---|
| Chunk size | 300–400 tokens | Event descriptions are short; avoid splitting mid-event |
| Overlap | 50 tokens | Preserve context at boundaries |
| Strategy | Recursive character splitter | Respects sentence boundaries |
| Metadata | Attach to every chunk | Needed for citation in responses |

**Metadata to attach to each chunk:**
```
event_uid, title, city, start_date, end_date, source_url
```

This metadata travels with the vector into FAISS and is returned alongside search results. Without it, your chatbot cannot cite sources.

---

### Step 6 — Embedding Strategy

**Choose your embedding model deliberately.**

For this project, `mistral-embed` is the correct choice because:
- Same provider as your generation model — consistent tokenization
- Strong French language support
- No additional API contract needed

**Embedding pipeline:**
1. Load cleaned, chunked events
2. For each chunk, call the embedding API
3. Store the resulting vector alongside its metadata
4. Handle API rate limits with retry logic and sleep intervals

**Output:** Two parallel structures:
- A list of embedding vectors (numpy array)
- A list of metadata dictionaries (same order)

Order must be preserved perfectly. This is the contract between your FAISS index and your metadata store.

---

### Step 7 — FAISS Index Construction

**Build two artifacts and version both.**

| Artifact | File | Purpose |
|---|---|---|
| Vector index | `faiss_index.idx` | Fast similarity search |
| Metadata store | `metadata.json` | Maps index position → event info |

**Index type for POC:** `IndexFlatL2` — exact search, no configuration needed, sufficient for POC scale.

**Build script (`build_index.py`) must:**
1. Check if index already exists — skip rebuild if so
2. Load embeddings and metadata
3. Build and save FAISS index
4. Save metadata JSON
5. Print confirmation: number of vectors indexed

**One-command rebuild:**
```bash
python build_index.py --force
```

---

### Step 8 — RAG Chain Construction

**Build and test each layer in isolation before assembling.**

**Layer 1 — Retriever:**
Given a query, embed it, search FAISS, return top-k chunks with metadata. Test this alone. Do the results make semantic sense?

**Layer 2 — Prompt template:**
This is where most POCs fail. Your system prompt must:
- Instruct the model to answer **only** from provided context
- Specify French as the output language
- Explicitly instruct the model to say "I don't have this information" rather than invent
- Define the assistant's persona (cultural event advisor for Normandie)

**Layer 3 — Full chain (LangChain `RetrievalQA`):**
Assembles retriever + prompt + Mistral LLM. Set `temperature=0.1`.

**Layer 4 — Edge case testing:**
| Query type | Expected behavior |
|---|---|
| Specific event query | Returns event details with date |
| Out-of-scope query | Politely declines |
| Vague query | Returns most relevant matches |
| No matching events | States no information found |

---

### Step 9 — Annotated Test Dataset

**Build this before the UI. It is your evaluation foundation.**

Create 20–30 manually verified Q&A pairs from your actual indexed data. Cover all query categories:

| Category | Example question |
|---|---|
| Specific event | "Are there any jazz concerts in Caen in April?" |
| Date-based | "What is happening this weekend in Normandie?" |
| Category-based | "Any art exhibitions near Caen?" |
| Out-of-scope | "What is the weather in Caen?" |
| No match | "Are there any bullfighting events?" |

**Format:**
```
question | reference_answer | source_event_uid | category
```

Save as CSV. This becomes your RAGAS input.

---

### Step 10 — RAGAS Evaluation

**Run evaluation before building the interface. Numbers first, demo second.**

For each question in your test dataset, run the full RAG pipeline and collect:
- The generated answer
- The retrieved context chunks

Then compute four RAGAS metrics:

| Metric | What it measures | Target for POC |
|---|---|---|
| `faithfulness` | Answer grounded in context? | > 0.80 |
| `answer_relevancy` | Answer addresses the question? | > 0.75 |
| `context_precision` | Retrieved chunks are relevant? | > 0.70 |
| `context_recall` | All needed info was retrieved? | > 0.70 |

**These scores go into your technical report and your slides.** Without them, you have a demo, not a POC.

---

### Step 11 — Streamlit Interface

**Keep the UI minimal. Three components only.**

**Main panel:**
- Chat input field
- Response display

**Sidebar:**
- Retrieved source chunks displayed transparently
- Event title, date, location for each source

**Why the sidebar matters:** It proves to the product and marketing teams that the system is grounded in real event data. It is your most powerful demo asset.

No conversation history needed for the POC — Jeremy confirmed this explicitly.

---

### Step 12 — Technical Report

**Structure (5–10 pages):**

1. **Executive summary** — one paragraph, what was built and what it demonstrates
2. **System architecture** — one clear diagram covering the full pipeline
3. **Data pipeline** — Open Agenda source, filtering logic, volumes
4. **Technical choices** — embedding model, chunk strategy, FAISS index type, each justified in one sentence
5. **Evaluation results** — RAGAS scores as a table, example good/bad outputs
6. **POC limitations** — what this does not yet handle
7. **Production recommendations** — what changes at scale

---

### Step 13 — Presentation (10–15 slides)

| Slide | Content |
|---|---|
| 1 | Title + context |
| 2 | Problem being solved |
| 3 | What is RAG? (simple diagram) |
| 4 | Solution overview |
| 5 | Data source: Open Agenda |
| 6 | System architecture diagram |
| 7 | Chunking & embedding strategy |
| 8 | FAISS vector search |
| 9 | LangChain + Mistral integration |
| 10 | Evaluation methodology |
| 11 | RAGAS results table |
| 12 | Live demo |
| 13 | POC limitations |
| 14 | Production recommendations |
| 15 | Q&A |

---

### Execution Order Summary

| Day | Deliverable |
|---|---|
| 1 | Repo setup, environment, README skeleton |
| 2 | Open Agenda ingestion + unit tests passing |
| 3 | Chunking + embedding + FAISS index built |
| 4 | RAG chain built and tested layer by layer |
| 5 | Annotated dataset + RAGAS evaluation |
| 6 | Streamlit interface |
| 7 | Technical report + slides + demo rehearsal |

---

### Quality Gates — Never Skip These

| Gate | Condition to proceed |
|---|---|
| After Step 2 | `check_imports.py` runs cleanly |
| After Step 4 | All 3 pytest unit tests pass |
| After Step 7 | Index rebuilds from scratch in one command |
| After Step 8 | Edge case queries behave correctly |
| After Step 10 | RAGAS scores computed and documented |
| Before demo | Full pipeline runs end-to-end without errors |

---

Which step do you want to implement first? I'll give you the precise, production-quality code for that step alone.