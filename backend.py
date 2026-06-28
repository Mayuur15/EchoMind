from groq import Groq
import json
from sentence_transformers import SentenceTransformer
from db import insert_entry, insert_substory, insert_memory_object, insert_embeddings
from dotenv import load_dotenv
import os
from cluster import run_insights
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')
app = FastAPI()
class Input(BaseModel):
    user_input: str
SUBSTORY_PROMPT = """
You are a psychological journaling assistant. Your job is to extract distinct substories from a journal entry.

A substory is a self-contained event, experience, thought, or feeling within the entry that can stand alone — something that happened, an interaction, a decision, a moment of emotion tied to a specific thing.

Rules:
- Extract between 1 and 5 substories. Most entries have 2-3.
- Each substory must be grounded in what the person actually wrote. Do not invent, assume, or extrapolate.
- Write each substory as a first-person reconstruction in 2-4 sentences, preserving the person's voice.
- If the entire entry is one continuous experience with no distinct parts, return exactly 1 substory.
- Assign a single topic label (one or two words max) to each substory.
- Return ONLY valid JSON. No preamble, no explanation, no markdown, no backticks.

Output format:
{
  "substories": [
    {
      "id": 1,
      "text": "first-person reconstruction of this substory in 2-4 sentences",
      "topic": "one or two word label"
    }
  ]
}
"""

MEMORIAL_OBJECT_PROMPT = """
You are a psychological analyst. Given a substory extracted from a journal entry, extract a structured memorial object that captures the psychological and emotional content of that substory.

Rules:
- Base everything strictly on what is written. Do not invent or assume.
- For enum fields, you MUST pick from the provided options only.
- If a field genuinely does not apply, use null — do not force-fit.
- automatic_thought should be a raw first-person thought the person was having, reconstructed verbatim in their voice. Example: "I always mess things up when it matters."
- valence is a float from -1.0 (most negative) to 1.0 (most positive). 0.0 is neutral.
- intensity is a float from 0.0 (barely felt) to 1.0 (overwhelming).
- Return ONLY valid JSON. No preamble, no explanation, no markdown, no backticks.
- The root element must be an object, not an array.

Output format:
{
  "situation": {
    "context": "one sentence describing what happened",
    "domain": one of ["work", "relationships", "health", "identity", "achievement", "leisure", "family"]
  },
  "appraisal": {
    "goal_relevance": one of ["high", "medium", "low"],
    "goal_congruence": one of ["helped", "blocked", "neutral"],
    "agency": one of ["self", "other", "circumstance", "unknown"],
    "coping_potential": one of ["high", "medium", "low"],
    "certainty": one of ["certain", "uncertain"]
  },
  "emotion": {
    "primary": "single emotion word",
    "valence": float between -1.0 and 1.0,
    "intensity": float between 0.0 and 1.0,
    "secondary": ["emotion1", "emotion2"] or []
  },
  "cognition": {
    "automatic_thought": "first-person raw thought in the person's voice",
    "cognitive_distortion": one of ["catastrophizing", "mind_reading", "all_or_nothing", "personalization", "fortune_telling", "should_statements", "comparison", "minimization", null]
  },
  "action_tendency": {
    "urge": one of ["approach", "avoid", "freeze", "attack", "withdraw", "seek_support", "ruminate", null],
    "actual_behavior": "what the person actually did"
  },
  "self_schema": {
    "belief_activated": "core belief the person revealed, in first person",
    "schema_domain": one of ["abandonment", "mistrust", "defectiveness", "failure", "dependence", "vulnerability", "enmeshment", "subjugation", "self_sacrifice", "approval_seeking", "negativity", "emotional_inhibition", "unrelenting_standards", null]
  }
}
"""

def clean_response(raw: str) -> dict:
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found:\n{repr(raw)}")
    return json.loads(raw[start:end+1])

def extract_substory(story_entry: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SUBSTORY_PROMPT},
            {"role": "user", "content": story_entry}
        ],
        temperature=0.3
    )
    return clean_response(response.choices[0].message.content)

def extract_memorial_object(substory_text: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": MEMORIAL_OBJECT_PROMPT},
            {"role": "user", "content": substory_text}
        ],
        temperature=0.2
    )
    return clean_response(response.choices[0].message.content)
def extract_imp(memorial:dict) -> str:
    appraisal = memorial["appraisal"] 
    emotion = memorial["emotion"]
    cognition = memorial["cognition"]
    action_tendency = memorial["action_tendency"]
    self_schema = memorial["self_schema"]
    return f"""
appraisal: {appraisal['goal_relevance']} {appraisal['goal_congruence']} {appraisal['agency']} {appraisal['coping_potential']} {appraisal["certainty"]}
emotion: {emotion['primary']} {emotion['secondary']} {emotion["valence"]} {emotion["intensity"]}
action: {action_tendency['urge']} {action_tendency['actual_behavior']}
cognition: {cognition['automatic_thought']} {cognition['cognitive_distortion']}
schema: {self_schema['belief_activated']} {self_schema['schema_domain']}
"""

def encode(text: str) -> list:
    vector = model.encode(text)
    return vector.tolist()

def process_entry(user_id: str,entry:str):
    entry_id = insert_entry(user_id, entry)
    print(f"Entry id : {entry_id}")
    substories = extract_substory(entry)
    for substory in substories["substories"]:
        print(f"Processing substory : {substory["topic"]}")
        print(substory["text"])
        substory_id = insert_substory(entry_id,user_id,substory["text"],substory["topic"])
        memorial = extract_memorial_object(substory["text"])
        print(memorial)
        memorial_id = insert_memory_object(substory_id,entry_id,user_id,memorial)
        semantic_vector = encode(substory["text"])
        psychological_vector = encode(extract_imp(memorial))
        insert_embeddings(substory_id,memorial_id,user_id,semantic_vector,psychological_vector)
    print("Entry fully processed")
    run_insights(user_id)
    return "success"
@app.get("/")
async def home():
    return FileResponse("index.html")
  
@app.post("/submit")
async def run(data : Input) :
    result = process_entry("mayuur",data.user_input)
    return {
        "result" : result
    }
