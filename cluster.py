import psycopg2
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter, defaultdict
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from groq import Groq
import os
import json

load_dotenv()

THRESHOLD = 6 

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_connection():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    register_vector(conn)
    return conn

def fetch_all(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT me.memory_id, me.embedding,
               mo.emotion, mo.action_tendency, mo.situation,
               mo.appraisal, mo.cognition, mo.self_schema
        FROM memory_embeddings me
        JOIN memory_objects mo ON me.memory_id = mo.memory_id
        WHERE me.user_id = %s
    """, (user_id,))
    rows = cur.fetchall()
    cur.close(); conn.close()
    return rows

def parse_embedding(raw):
    if isinstance(raw, np.ndarray): return raw.astype(np.float32)
    if isinstance(raw, list):       return np.array(raw, dtype=np.float32)
    if isinstance(raw, str):        return np.array(json.loads(raw), dtype=np.float32)
    return np.array(raw, dtype=np.float32)

def build_clusters(user_id):
    rows = fetch_all(user_id)
    if len(rows) < 4:
        return None, None

    embeddings = np.array([parse_embedding(r[1]) for r in rows])
    emotions   = [r[2].get("primary", "unknown") if r[2] else "unknown" for r in rows]
    actions    = [r[3].get("urge") or "none"     if r[3] else "none"    for r in rows]
    situations = [r[4] if isinstance(r[4], str) else ""                 for r in rows]
    appraisals = [r[5] if r[5] else {}                                  for r in rows]
    cognitions = [r[6] if r[6] else {}                                  for r in rows]
    schemas    = [r[7] if r[7] else {}                                  for r in rows]

    labels = DBSCAN(eps=0.61, min_samples=2).fit(embeddings).labels_

    clusters = defaultdict(lambda: {
        "emotions": [], "actions": [], "situations": [],
        "appraisals": [], "cognitions": [], "schemas": []
    })

    for i, label in enumerate(labels):
        if label == -1: continue
        c = clusters[int(label)]
        c["emotions"].append(emotions[i])
        c["actions"].append(actions[i])
        if situations[i].strip():
            c["situations"].append(situations[i])
        if appraisals[i]:
            c["appraisals"].append(appraisals[i])
        if cognitions[i]:
            c["cognitions"].append(cognitions[i])
        if schemas[i]:
            c["schemas"].append(schemas[i])

    triggered = {cid: data for cid, data in clusters.items() if len(data["emotions"]) >= THRESHOLD}
    return triggered, len(rows)

# ── GROQ INTERPRETATION ────────────────────────────────────────────────────────

def interpret_cluster(cluster_data, cluster_id):
    emo_counts  = Counter(cluster_data["emotions"]).most_common(3)
    act_counts  = Counter(cluster_data["actions"]).most_common(2)
    thoughts    = [c.get("automatic_thought", "") for c in cluster_data["cognitions"] if isinstance(c, dict)]
    distortions = [c.get("cognitive_distortion") for c in cluster_data["cognitions"] if isinstance(c, dict) and c.get("cognitive_distortion")]

    schema_beliefs = []
    schema_domains = []
    for s in cluster_data["schemas"]:
        if isinstance(s, dict):
            if s.get("belief_activated"):
                schema_beliefs.append(s["belief_activated"])
            if s.get("schema_domain"):
                schema_domains.append(s["schema_domain"])

    summary_input = f"""
Cluster has {len(cluster_data['emotions'])} memories.
Top emotions: {', '.join(f"{e} ({n}x)" for e, n in emo_counts)}
Action tendencies: {', '.join(f"{a} ({n}x)" for a, n in act_counts)}
Automatic thoughts: {'; '.join(t for t in thoughts[:5] if t and t != 'None')}
Cognitive distortions: {', '.join(set(str(d) for d in distortions)) if distortions else 'none'}
Core beliefs: {'; '.join(schema_beliefs[:4]) if schema_beliefs else 'none detected'}
Schema domains: {', '.join(set(schema_domains)) if schema_domains else 'none'}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """You are a compassionate psychological pattern analyst for a journaling app called EchoMind.
Given cluster data from a user's journal memories, write a short insight in 3-4 sentences.
Be warm, specific, and direct. Name the pattern clearly. Reference the actual thoughts and schemas.
Do not use bullet points. Do not say 'I' or address the app. Write in second person ('you').
End with one concrete, actionable reframe or question for the user to sit with.
Keep it under 80 words."""
            },
            {"role": "user", "content": summary_input}
        ],
        temperature=0.4
    )
    return response.choices[0].message.content.strip()


def run_insights(user_id="mayuur"):
    print(f"\n[EchoMind] Running insight check for {user_id}...")

    clusters, total = build_clusters(user_id)

    if clusters is None:
        print("[EchoMind] Not enough memories yet.")
        return

    if not clusters:
        print(f"[EchoMind] No clusters have reached the threshold of {THRESHOLD} memories yet.")
        return

    print(f"\n[EchoMind] {total} memories analyzed. {len(clusters)} cluster(s) crossed threshold of {THRESHOLD}.\n")

    for cid, data in clusters.items():
        emo_counts = Counter(data["emotions"]).most_common(3)
        act_counts = Counter(data["actions"]).most_common(2)

        print(f"── Cluster {cid} ({len(data['emotions'])} memories) ──────────────────────")
        print(f"  Emotions:          {', '.join(f'{e} ({n}x)' for e, n in emo_counts)}")
        print(f"  Action tendencies: {', '.join(f'{a} ({n}x)' for a, n in act_counts)}")
        print(f"  Generating insight...")

        insight = interpret_cluster(data, cid)
        print(f"\n  INSIGHT: {insight}\n")

