# EchoMind

A journaling app that doesn't just store your entries — it figures out what psychological patterns keep showing up in them.

You write a journal entry. EchoMind breaks it into distinct sub-events, runs each one through a structured psychological extraction (emotions, automatic thoughts, cognitive distortions, core beliefs — grounded in CBT and appraisal theory), and embeds it two different ways: once for what happened, once for how you processed it. Once enough entries pile up, it clusters them and writes you a short, specific insight about the pattern it found — not generic journaling-app fluff, but something like "you keep catastrophizing when a task feels open-ended, and here's the belief driving it."

## How it works

1. **Entry submission** — you write, it gets split into 1–5 self-contained substories.
2. **Extraction** — each substory goes through an LLM call that returns a structured object: situation, appraisal, emotion (valence + intensity), the actual automatic thought in your voice, cognitive distortion (if any), action tendency, and the core belief it activates.
3. **Dual embedding** — the substory text gets embedded semantically (what happened), and the extracted psychological object gets embedded separately (how you processed it). Two different similarity spaces, on purpose.
4. **Clustering** — DBSCAN runs over the psychological embedding space. Clusters that hit a minimum size get treated as real patterns, not noise.
5. **Insight generation** — once a cluster crosses the threshold, an LLM writes a short, specific reflection on it and stores it. Re-runs only when the cluster actually changes.

## Stack

- **Backend:** FastAPI
- **LLM:** Groq (Llama 3.1 8B) for extraction and insight generation
- **Embeddings:** `all-MiniLM-L6-v2` (sentence-transformers)
- **Clustering:** DBSCAN (scikit-learn)
- **Database:** PostgreSQL + pgvector
- **Frontend:** single-page, no framework — plain HTML/CSS/JS

## Running it locally

```bash
git clone https://github.com/Mayuur15/EchoMind.git
cd EchoMind
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_key_here
DATABASE_URL=your_postgres_url_here
```

Then:

```bash
uvicorn backend:app --reload
```

Open `localhost:8000`.

## Known limitations

- Single-user right now — `user_id` is hardcoded. Multi-tenant support is the next thing to build.
- No retry/rollback if an LLM call fails mid-pipeline.
- Clustering thresholds (`eps=0.61`, min cluster size of 6) were tuned by hand on a small personal dataset, not grid-searched.

## Why these choices

**Two embeddings, not one.** Embedding raw text tells you two entries are similar because they talk about the same thing. Embedding the extracted psychological structure tells you two entries are similar because you *reacted* the same way, even if the surface content is completely different. That second signal is the one that actually surfaces a pattern worth noticing.

**DBSCAN over k-means.** You don't know how many psychological patterns someone has going in, and most entries shouldn't be forced into a cluster just because k-means needs somewhere to put them. DBSCAN lets noise stay noise.

**No RAG, no LangChain.** The pipeline is a fixed sequence of extraction → embed → cluster → summarize. There's no retrieval step where past context needs to get pulled back in at generation time, so adding a framework here would just be overhead.
