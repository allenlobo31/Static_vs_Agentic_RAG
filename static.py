# week6/static_rag.py
# STATIC RAG with Gemini: retrieve top-1 chunk -> stuff into prompt -> answer.
# No checking of whether the retrieved chunk is actually sufficient.
#
# Setup:  pip install google-genai
#         export GEMINI_API_KEY="your-key"      (Windows: set GEMINI_API_KEY=...)
# Run:    python static_rag.py

import os
import re
from google import genai
from google.genai import types

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # override via env if needed
client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment


# --- KNOWLEDGE BASE ------------------------------------------------------------
# The first sentence says "subject to the conditions in the next section".
# The actual conditions (Enterprise = 7 days only) live in the next part of the doc.
DOCUMENT = (
    "Acme Cloud Refund Policy. Customers may request a full refund within 30 days "
    "of purchase, subject to the conditions listed in the next section. "
    "CONDITIONS AND EXCEPTIONS. Enterprise tier purchases are only refundable within "
    "7 days of purchase. Accounts that have stored more than 100 GB of data are not "
    "eligible for any refund. "
    "REFUND PROCESSING. Approved refunds are returned to the original payment method "
    "within 5 to 10 business days. "
    "SUPPORT. Contact billing@acme.example to start a refund request."
)

QUESTION = "I bought the Acme Cloud Enterprise tier 20 days ago. Am I eligible for a full refund?"
GROUND_TRUTH = "NO - Enterprise purchases are only refundable within 7 days."


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