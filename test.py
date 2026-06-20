from groq import Groq
import json

client = Groq(api_key="")

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

story_entry ="""today was kind of all over the place. had a dbms class in the morning and 
the prof was going way too fast, couldn't keep up with the normalization 
stuff at all. felt pretty dumb honestly.

came back and worked on the substory extraction pipeline for echomind for 
like 3 hours. got the ollama call working finally. small thing but it felt 
good after the morning.

been thinking about the internship drive a lot. keep calculating if i have 
enough time to finish everything. probably overthinking it but it won't 
go away.
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

substories = extract_substory(story_entry)
for substory in substories["substories"]:
    memorial_object = extract_memorial_object(substory["text"])
    print(json.dumps(memorial_object, indent=2))
    