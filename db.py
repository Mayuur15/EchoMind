import psycopg2
import os
from dotenv import load_dotenv
import json
load_dotenv()

def get_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))
def insert_entry( userid: str,entry: str ) -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO entries (user_id , raw_text) VALUES (%s , %s) RETURNING entry_id",
        (userid,entry)
    )
    entry_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return entry_id
def insert_substory(entry_id: str, user_id: str, text: str, topic: str) -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO substories (entry_id, user_id, text, topic) VALUES (%s, %s, %s, %s) RETURNING substory_id",
        (entry_id, user_id, text, topic)
    )
    substory_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return str(substory_id)

def insert_memory_object(substory_id: str, entry_id: str, user_id: str, memorial: dict) -> str:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO memory_objects 
        (substory_id, entry_id, user_id, situation, appraisal, emotion, cognition, action_tendency, self_schema)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING memory_id""",
        (
            substory_id, entry_id, user_id,
            json.dumps(memorial.get("situation")),
            json.dumps(memorial.get("appraisal")),
            json.dumps(memorial.get("emotion")),
            json.dumps(memorial.get("cognition")),
            json.dumps(memorial.get("action_tendency")),
            json.dumps(memorial.get("self_schema"))
        )
    )
    memory_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return str(memory_id)

def insert_insight(user_id: str, cluster_id: int, cluster_size: int, 
                   insight: str, dominant_emotion: str, dominant_action: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO cluster_insights 
            (user_id, cluster_id, cluster_size, insight, dominant_emotion, dominant_action)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING insight_id
    """, (user_id, cluster_id, cluster_size, insight, dominant_emotion, dominant_action))
    insight_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    
def get_clusters(user_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT cluster_id , cluster_size , insight , dominant_emotion , dominant_action FROM cluster_insights WHERE user_id = %s",(user_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def insert_embeddings(substory_id: str, memory_id: str, user_id: str, 
                      semantic_vector: list, psychological_vector: list):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO substory_embeddings (substory_id, user_id, embedding) VALUES (%s, %s, %s)",
        (substory_id, user_id, semantic_vector)
    )
    cur.execute(
        "INSERT INTO memory_embeddings (memory_id, user_id, embedding) VALUES (%s, %s, %s)",
        (memory_id, user_id, psychological_vector)
    )
    conn.commit()
    cur.close()
    conn.close()

