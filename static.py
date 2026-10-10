# STATIC RAG with Gemini: retrieve top-1 chunk -> stuff into prompt -> answer.
# No checking of whether the retrieved chunk is actually sufficient.



import os
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash") 

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise SystemExit("No API key found. Add GEMINI_API_KEY=your-key to the .env file "
                     "(same folder you run the script from).")
client = genai.Client(api_key=API_KEY)




# --- KNOWLEDGE BASE ------------------------------------------------------------
# The first sentence says "subject to the conditions in the next section".
# The actual conditions (Enterprise = 7 days only) live in the next part of the doc.
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
    "Much later in the story, Imu is introduced as the supreme ruler of the world and the shadow ruler "
    "of the World Government. The Five Elders serve directly under Imu, who truly rules the world. "
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
def chunk_by_words(text, size=24):
    """Fixed-size chunking with no overlap and no respect for sections.
    This is the common 'naive' setup that cuts related info apart."""
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


CHUNKS = chunk_by_words(DOCUMENT)





# --- RETRIEVAL -----------------------------------------------------------------
# Lexical overlap scoring keeps the demo deterministic. In production you would
# use embeddings; top-k retrieval can still return a partial chunk there too.
STOP = {"a", "an", "the", "i", "am", "is", "are", "of", "to", "for", "in", "ago",
        "any", "that", "have", "than", "more", "be", "or", "and", "within"}


def tokens(text):
    out = set()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w not in STOP:
            out.add(w.rstrip("s"))  # crude stemming
    return out


def retrieve(query, top_k=1):
    q = tokens(query)
    scored = sorted(
        ((len(q & tokens(c)), i) for i, c in enumerate(CHUNKS)), reverse=True
    )
    return [(i, CHUNKS[i]) for score, i in scored[:top_k] if score > 0]




# --- GENERATION ----------------------------------------------------------------
def llm(prompt):
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
    return resp.text


def static_rag(question):
    print(f"Question: {question}\n")
    hits = retrieve(question, top_k=1)
    context = "\n".join(text for _, text in hits)
    print(f"[Retrieved chunk(s) {[i for i, _ in hits]}]\n{context}\n")

    prompt = (
        "Answer the question using the context below.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    answer = llm(prompt)  # no sufficiency check, no verification
    print(f"[Static RAG answer]\n{answer}\n")
    print(f"[Ground truth from the FULL document]\n{GROUND_TRUTH}")
    return answer




if __name__ == "__main__":
    static_rag(QUESTION)