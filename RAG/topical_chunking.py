import os
import time
import json
import requests
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import hdbscan
from concurrent.futures import ThreadPoolExecutor

# ---------- CONFIG ----------
INPUT_FILE = "motorolaedge50fusion_comments.txt"
OUTPUT_DIR = "clusters"
MODEL_NAME = 'all-MiniLM-L6-v2'
BATCH_SIZE = 64
MAX_WORKERS = 4
MIN_CLUSTER_SIZE = 10
TOP_K_CLUSTERS = 20

# Groq Settings
GROQ_MODEL = 'llama-3.3-70b-versatile'
GROQ_API_KEY = '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'
# ----------------------------

def groq_chat(prompt, retries=3):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }

    for _ in range(retries):
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        elif response.status_code == 429:
            time.sleep(2)
        else:
            print("❌ Groq API Error:", response.status_code, response.text)
            break
    return "Unknown_Topic"

def read_comments(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        comments = list(executor.map(lambda x: x.strip(), lines))

    return [c for c in comments if c]

def embed_comments(comments, model, batch_size=BATCH_SIZE):
    embeddings = []
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

def cluster_with_hdbscan(embeddings, min_cluster_size=MIN_CLUSTER_SIZE):
    print("🔍 Reducing dimensionality with UMAP (parallel)...")
    reducer = UMAP(n_components=15, n_neighbors=15, metric='cosine', n_jobs=-1)
    reduced = reducer.fit_transform(embeddings)

    print("🔍 Running HDBSCAN clustering (parallel)...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom',
        core_dist_n_jobs=-1
    )
    labels = clusterer.fit_predict(reduced)

    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(idx)

    print(f"✅ Found {len(clusters)} clusters.")
    return list(clusters.values())

def label_cluster(comments_subset):
    sample = comments_subset[:10]
    prompt = (
        "Summarize the common theme in the following product comments into a short topic name (3–6 words)/ The name of the file should be the summarized phrase only:\n\n" +
        "\n".join(f"- {c}" for c in sample)
    )
    label = groq_chat(prompt)
    return label.replace(" ", "_")[:60].replace("/", "_")

def save_clusters(clusters, comments, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def write_cluster(idx_cluster):
        idx, cluster = idx_cluster
        cluster_comments = [comments[i] for i in cluster]
        label = label_cluster(cluster_comments)
        filename = f"topic_{idx}_{label}.txt"
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            for c in cluster_comments:
                f.write(c + '\n')

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(write_cluster, enumerate(clusters))

    print(f"✅ Saved {len(clusters)} labeled topical chunks to: {output_dir}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Topical chunking for product comments')
    parser.add_argument('--input', default=INPUT_FILE, help='Input comments file')
    args = parser.parse_args()
    
    input_file = args.input
    
    print("[1] Reading comments...")
    comments = read_comments(input_file)
    if not comments:
        print("❌ No comments found. Exiting.")
        return

    print("[2] Embedding comments...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = embed_comments(comments, model)

    print("[3] Clustering with HDBSCAN...")
    clusters = cluster_with_hdbscan(embeddings)

    if not clusters:
        print("❌ No clusters found. Exiting.")
        return

    print(f"[4] Selecting top {TOP_K_CLUSTERS} clusters by size...")
    clusters = sorted(clusters, key=lambda c: len(c), reverse=True)[:TOP_K_CLUSTERS]

    print(f"[5] Saving {len(clusters)} final clusters...")
    save_clusters(clusters, comments, OUTPUT_DIR)

if __name__ == "__main__":
    main()
