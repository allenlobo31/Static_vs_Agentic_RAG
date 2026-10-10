# AGENTIC RAG: retrieve -> grade sufficiency (CRAG) -> rewrite query / expand to
# neighbouring chunk if info is missing -> generate -> verify groundedness (Self-RAG)
# -> abstain if it still can't be supported.


import os
import re
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise SystemExit("No API key found. Add GEMINI_API_KEY=your-key to the .env file.")
client = genai.Client(api_key=API_KEY)




# --- SAME KNOWLEDGE BASE / CHUNKING AS static_rag.py ---------------------------
DOCUMENT = (
    "One Piece: The World Government. "
    "Officially, nobody rules the world government, and no king or anyone else holds sole power over it. "
    "The Marines, led by a Fleet Admiral and the Admirals, keep the peace across the world. "
    "The World Government is bound to the Celestial Dragons: it serves them, protecting and "
    "benefiting them rather than the whole world. "
    "The Five Elders are the highest visible rank and control the whole operation. "
    "The Celestial Dragons consider themselves gods. "
    "The faces of the Five Elders are known only to the Fleet Admiral, the Admirals and the Celestial Dragons. "
    "THE SECRET RULER. "
    "Much later in the story, Imu is introduced as the supreme ruler of the world who rules the world government in secret. "
    "Imu rules the whole world and world government, and the Five Elders serve directly under Imu, who truly rules the world. "
    "The Five Elders and world government hides that it is been ruled by world government. to profit them."
    "No one other than the Five Elders has seen Imu, until recently, when Sabo saw Imu. "
    "BACKGROUND. "
    "The capital of the World Government is Mary Geoise, a Holy Land on the Red Line, and its "
    "headquarters is Pangaea Castle, which holds the Empty Throne. The throne is said to symbolize "
    "that no single person from the twenty families of Celestial Dragons can rule the world, but in "
    "truth it is occupied in secret by Imu."
)

QUESTION = "Does anyone rule the World Government?"
GROUND_TRUTH = "YES - Imu is the secret supreme ruler; the Five Elders serve directly under Imu."





# --- NAIVE CHUNKING ------------------------------------------------------------
def chunk_by_words(text, size=45):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


CHUNKS = chunk_by_words(DOCUMENT)





# --- RETRIEVAL -----------------------------------------------------------------
STOP = {"a", "an", "the", "i", "am", "is", "are", "of", "to", "for", "in", "ago",
        "any", "that", "have", "than", "more", "be", "or", "and", "within"}


def tokens(text):
    return {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP}


def retrieve(query, top_k=1, exclude=()):
    q = tokens(query)
    scored = sorted((-len(q & tokens(c)), i) for i, c in enumerate(CHUNKS) if i not in exclude)
    return [(i, CHUNKS[i]) for s, i in scored[:top_k] if s < 0]





# --- GEMINI HELPERS ------------------------------------------------------------
def llm(prompt, as_json=False):
    cfg = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json" if as_json else None,
    )
    text = client.models.generate_content(model=MODEL, contents=prompt, config=cfg).text
    if not as_json:
        return text
    try:
        return json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip())
    except json.JSONDecodeError:
        return None  # caller treats unparseable output as "could not verify"






# --- AGENT STEPS ---------------------------------------------------------------
def grade_context(question, context):
    """CRAG-style check: is the context sufficient AND complete?"""
    prompt = (
        "You are a strict retrieval grader. Decide whether the CONTEXT contains everything "
        "needed to answer the QUESTION reliably.\n"
        "Mark it NOT sufficient if the context refers to conditions, exceptions, sections or "
        "details that are not included, or appears cut off.\n"
        'Reply as JSON: {"sufficient": true|false, "missing_info": "<what is missing>", '
        '"search_query": "<short keyword query to find the missing info>"}\n\n'
        f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    )
    return llm(prompt, as_json=True) or {"sufficient": False, "missing_info": "unparseable grader output",
                                         "search_query": question}


def generate(question, context):
    prompt = (
        "Answer the question using ONLY the context. If the context does not fully support an "
        "answer, say you cannot answer. Do not assume details that are not stated.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:"
    )
    return llm(prompt)


def verify(question, context, answer):
    """Self-RAG-style check: is every claim in the answer supported by the context?"""
    prompt = (
        "Check whether the ANSWER is fully supported by the CONTEXT.\n"
        'Reply as JSON: {"grounded": true|false, "unsupported_claims": ["..."]}\n\n'
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER: {answer}"
    )
    return llm(prompt, as_json=True) or {"grounded": False, "unsupported_claims": ["verifier output unparseable"]}


def agentic_rag(question, max_rounds=4):
    print(f"Question: {question}\n")
    seen = {}            # chunk_id -> text
    query = question

    for rnd in range(1, max_rounds + 1):
        print(f"--- Round {rnd} | query: '{query}'")
        hits = retrieve(query, top_k=1, exclude=seen)

        # Fallback: rewritten query found nothing new -> pull the neighbour of the last chunk
        if not hits and seen:
            nxt = max(seen) + 1
            if nxt < len(CHUNKS) and nxt not in seen:
                hits = [(nxt, CHUNKS[nxt])]
                print(f"[Routing] No new match; expanding to neighbouring chunk {nxt}")
        if not hits:
            print("[Routing] Nothing left to retrieve.")
            break

        for i, text in hits:
            seen[i] = text
            print(f"[Retrieved chunk {i}] {text}")

        context = "\n".join(seen[i] for i in sorted(seen))
        grade = grade_context(question, context)
        print(f"[CRAG grade] sufficient={grade.get('sufficient')} | missing: {grade.get('missing_info')}")

        if grade.get("sufficient"):
            answer = generate(question, context)
            check = verify(question, context, answer)
            print(f"[Self-RAG verify] grounded={check.get('grounded')} {check.get('unsupported_claims') or ''}")
            if check.get("grounded"):
                print(f"\n[Agentic RAG answer]\n{answer}\n")
                print(f"[Ground truth from the FULL document]\n{GROUND_TRUTH}")
                return answer
            print("[Self-Correction] Answer not grounded; retrieving more context.")

        # Dynamic query formulation from what the grader says is missing
        query = grade.get("search_query") or grade.get("missing_info") or question

    abstain = "I can't answer this reliably: the retrieved documents don't contain enough verified information."
    print(f"\n[Agentic RAG answer]\n{abstain}\n")
    return abstain


if __name__ == "__main__":
    agentic_rag(QUESTION)