import os
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from config import get_azure_client
import json

EMBED_CACHE = "embeddings_cache.json"


client = get_azure_client()

# -------------------------
# Config
# -------------------------
DOC_DIR = "documents"
EMB_MODEL = "aicso-embed"


# -------------------------
# Dataclass for documents
# -------------------------
@dataclass
class PolicyDocument:
    key: str
    content: str
    embedding: List[float]


# -------------------------
# Utils
# -------------------------
def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts in one API call."""
    response = client.embeddings.create(
        model=EMB_MODEL,
        input=texts
    )
    return [d.embedding for d in response.data]


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


# -------------------------
# Load all documents + cache embeddings
# -------------------------
def load_policy_documents() -> List[PolicyDocument]:
    raw_docs = {
        "card_issues": load_text_file(os.path.join(DOC_DIR, "card_policy.txt")),
        "transfer_issues": load_text_file(os.path.join(DOC_DIR, "transfer_policy.txt")),
    }

    # ✅ Load cache if it exists
    if os.path.exists(EMBED_CACHE):
        print("Loading embeddings from cache...")
        with open(EMBED_CACHE, "r") as f:
            cached = json.load(f)

        return [
            PolicyDocument(
                key=k,
                content=raw_docs[k],
                embedding=v
            )
            for k, v in cached.items()
        ]

    print("Creating embeddings (one-time)...")
    embeddings = embed_batch(list(raw_docs.values()))

    docs = []
    cache = {}
    for (key, content), emb in zip(raw_docs.items(), embeddings):
        docs.append(PolicyDocument(key=key, content=content, embedding=emb))
        cache[key] = emb

    # ✅ Save cache
    with open(EMBED_CACHE, "w") as f:
        json.dump(cache, f)

    return docs


policy_docs = load_policy_documents()


# -------------------------
# Retrieval function
# -------------------------
def retrieve_relevant_doc(intent: str, user_text: str) -> Tuple[str, float]:
    if not user_text.strip():
        return None, 0.0

    user_emb = embed_batch([user_text])[0]

    # normalize intent key
    intent_key = intent.lower().strip()

    # map intent to doc keys
    intent_map = {
        "card_issue": "card_issues",
        "transfer_issue": "transfer_issues"
    }

    target_key = intent_map.get(intent_key)

    filtered_docs = [doc for doc in policy_docs if doc.key == target_key]

    if not filtered_docs:
        print("No matching document for intent:", intent)
        return None, 0.0

    scores = [cosine_similarity(user_emb, doc.embedding) for doc in filtered_docs]

    best_idx = int(np.argmax(scores))
    best_doc = filtered_docs[best_idx]

    print("Best match:", best_doc.key)

    return best_doc.content, scores[best_idx]
