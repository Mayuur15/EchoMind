# EchoMind — spec.md

## About the Product

- What is EchoMind in one sentence? (not technical — what does it do for the person using it)

- What is the one thing it does that no other app does?

- Who is the ideal user?

- What does the "ohhhhh" moment look like for that user?

---

## About the Data Flow

- What happens from the moment a user submits a diary entry to the moment an insight appears?

- How many LLM calls happen per entry? What does each one do?

- How many embeddings are created per entry? What is each one used for?

- Where does pattern detection happen — on raw text, on embeddings, on extracted metadata, or some combination?

---

## About the Architecture

- What are all the tables in your database and what does each store?

- What is the difference between your two embeddings per chunk?

- How does the clustering work? What algorithm? What parameters?

- How does the system decide something is a pattern vs random noise?

---

## About the Stack

- Why Qwen3 over GPT-4 or Claude?

- Why all-MiniLM-L6-v2 over other embedding models?

- Why pgvector over a dedicated vector database like Pinecone?

- Why DBSCAN over k-means for clustering?

---

## About the Frontend

- What are the three pages and what does each one show?

- What does the insight visualization actually look like?

- What is the one visual that creates the "ohhhhh" moment?

---

## What You Haven't Figured Out Yet

- How do you evaluate chunking quality?

- How do you evaluate extraction quality?

- How do you prevent the LLM from hallucinating emotions that aren't in the entry?

- What's the minimum number of occurrences before something becomes a pattern?

---

## Decisions Already Made

- No RAG — why?

- No LangChain — why?

- Two embeddings per chunk — why?

- Embed extracted memory not raw text for pattern matching — why?
