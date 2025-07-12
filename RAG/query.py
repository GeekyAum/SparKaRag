import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import os
import requests

# ---------- CONFIG ----------
MODEL_NAME    = 'all-MiniLM-L6-v2'
GROQ_MODEL    = 'llama-3.3-70b-versatile'
INDEX_FILE    = 'faiss_index.index'
METADATA_FILE = 'metadata.json'
TOP_K         = 3
GROQ_API_KEY = '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' 
# ----------------------------

def embed_text(text, model):
    return model.encode([text])[0]

def load_faiss_index(path):
    return faiss.read_index(path)

def load_metadata(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def retrieve_with_sources(query, model, index, metadata, k=TOP_K):
    query_embedding = embed_text(query, model).reshape(1, -1)
    D, I = index.search(query_embedding, k)
    sources = []
    for idx in I[0]:
        entry = metadata[idx]
        sources.append({
            "topic": entry["topic"],
            "summary": entry["summary"],
            "comments": entry["comments"]
        })
    return sources

def load_ner_context():
    ner_file = "RAG/extra_context/context_store.json"
    if os.path.exists(ner_file):
        with open(ner_file) as f:
            return json.load(f)
    return None

def build_prompt(query, sources, ner_context=None):
    prompt = f"Answer the following question. Use only the summaries and optionally the sentiment context. Cite each source like [S1], [S2], etc.\n\n"
    prompt += f"QUESTION: {query}\n\n"
    prompt += "SOURCES:\n"
    for i, s in enumerate(sources, start=1):
        prompt += f"[S{i}] Topic {s['topic']} Summary:\n{s['summary']}\n\n"
    if ner_context:
        prompt += "\n[Extra Context: Entity-Level Sentiment]\n"
        prompt += json.dumps(ner_context, indent=2)
    return prompt

def expand_citations(answer, sources):
    def repl(match):
        idx = int(match.group(1)) - 1
        comments = sources[idx]["comments"]
        snippet = "\n".join(f"  - “{c}”" for c in comments[:3])
        return f"[S{idx+1}]\nComments:\n{snippet}\n"
    return re.sub(r"\[S(\d+)\]", repl, answer)

def groq_chat(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Request failed with status {response.status_code}: {response.text}")
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content'].strip()

def main():
    print("🔍 Query Mode: Ask me anything about the product.")

    index = load_faiss_index(INDEX_FILE)
    metadata = load_metadata(METADATA_FILE)
    model = SentenceTransformer(MODEL_NAME)

    while True:
        query = input("\n🔸 Your Question (or type 'exit'): ").strip()
        if query.lower() == 'exit':
            break

        sources = retrieve_with_sources(query, model, index, metadata)
        ner_context = load_ner_context()
        prompt = build_prompt(query, sources, ner_context)

        try:
            raw_answer = groq_chat(prompt)
            final_answer = expand_citations(raw_answer, sources)
            print("\n🧠 Answer with Sources:")
            print(final_answer)
        except Exception as e:
            print("❌ Failed to get a response:", e)

if __name__ == "__main__":
    main()
