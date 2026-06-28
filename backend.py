from groq import Groq
import json
from sentence_transformers import SentenceTransformer
from db import insert_entry, insert_substory, insert_memory_object, insert_embeddings
from dotenv import load_dotenv
import os
from cluster import run_insights
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')
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

  

if __name__ == "__main__":
    process_entry("mayuur", """Didn't get much done today and I'm weirdly okay with it. The day kind of took over before I had a chance to plan it properly — Arjun texted in the morning asking if I wanted to come with him and a few others to that new place near the main road for lunch, and I said yes mostly because I'd been inside for three days straight and my room was starting to feel like a verdict.
Lunch was loud and good. Six people, two of whom I barely know, one of whom talked about crypto for twenty minutes straight. I ate too much and laughed more than I expected to. Priya was there which I didn't know beforehand — she brought up the hackathon in front of everyone and said "Mayuur's the one keeping the ML side honest" which I didn't know how to receive so I just nodded. Arjun gave me a look afterward. I told him to shut up.
The Uthara thing happened on the walk back. She was coming the other direction with a friend, and we kind of just — stopped. Her friend kept walking a little ahead, doing that thing people do when they're giving you space they've decided you need. We talked for maybe ten minutes. Nothing significant on paper. She asked about EchoMind, I explained the clustering part, she said "so it basically knows you better than you know yourself" and I said that's the idea and she smiled in a way that I've been thinking about since.
I don't know what to do with any of this. I got back to my room and sat on my bed for a while not looking at my phone. Then I looked at my phone. Then I put it face down. Rohan called and I let it ring out, which I feel bad about — I'll call him tomorrow. Made dinner at 9, ate it standing up, forgot to clean the pan.
The thing about today is I can't tell if it was good or if I'm just telling myself it was. But I keep coming back to that smile and I think that's probably an answer of some kind.""")
