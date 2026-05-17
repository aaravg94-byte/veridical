import json
import os
import requests
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────
INPUT_FILE   = "papers.json"
OUTPUT_FILE  = "papers.json"
N_CLUSTERS   = 6     # number of topic groups to find
# ───────────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise SystemExit("Error: GROQ_API_KEY not set. Run: export GROQ_API_KEY='your-key'")

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

print("Loading embedding model (first run downloads ~90MB)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> list[float]:
    """Get a vector embedding for a piece of text — runs locally, no API needed."""
    return embedder.encode(text).tolist()


def get_cluster_label(papers_in_cluster: list[dict]) -> str:
    """Ask Groq to name a cluster based on its papers."""
    titles = "\n".join(f"- {p['title']}" for p in papers_in_cluster[:6])
    claims = "\n".join(f"- {p['claim']}" for p in papers_in_cluster[:6] if p.get("claim") and p["claim"] != "skipped")

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You name research clusters. Respond with ONLY a 2-5 word topic label, nothing else. No punctuation at the end."},
            {"role": "user", "content": f"Papers in this cluster:\n{titles}\n\nKey claims:\n{claims}\n\nGive this cluster a 2-5 word topic label:"}
        ],
        "temperature": 0.1,
        "max_tokens": 20,
    }
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=HEADERS,
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        papers = json.load(f)

    # Only cluster papers with real claims
    valid = [p for p in papers if p.get("claim") and p["claim"] != "skipped"]
    print(f"\nEmbedding {len(valid)} papers...\n")

    # Step 1: embed each claim
    embeddings = []
    for i, paper in enumerate(valid):
        print(f"  [{i+1}/{len(valid)}] Embedding: {paper['title'][:65]}...")
        vec = get_embedding(paper["claim"])
        embeddings.append(vec)

    matrix = np.array(embeddings)
    print(f"\n✓ Got {len(embeddings)} embeddings (each is {len(embeddings[0])} dimensions)\n")

    # Step 2: cluster with KMeans
    print(f"Clustering into {N_CLUSTERS} topic groups...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(matrix)

    # Assign cluster IDs back to papers
    for i, paper in enumerate(valid):
        paper["cluster"] = int(cluster_ids[i])

    # Step 3: label each cluster
    print("Labelling clusters...\n")
    cluster_labels = {}
    for cid in range(N_CLUSTERS):
        cluster_papers = [p for p in valid if p["cluster"] == cid]
        label = get_cluster_label(cluster_papers)
        cluster_labels[cid] = label
        print(f"  Cluster {cid}: '{label}' ({len(cluster_papers)} papers)")

    # Step 4: find most similar pairs within each cluster (for contradiction detection in Step 4)
    print("\nFinding closest paper pairs per cluster...")
    pairs = []
    for cid in range(N_CLUSTERS):
        indices = [i for i, c in enumerate(cluster_ids) if c == cid]
        if len(indices) < 2:
            continue
        sub_matrix = matrix[indices]
        sim = cosine_similarity(sub_matrix)
        np.fill_diagonal(sim, -1)  # ignore self-similarity
        for row in range(len(indices)):
            best = int(np.argmax(sim[row]))
            pairs.append({
                "paper_a": valid[indices[row]]["id"],
                "paper_b": valid[indices[best]]["id"],
                "cluster": cid,
                "similarity": float(sim[row][best]),
                "verdict": None   # filled in Step 4
            })

    # Deduplicate pairs (a,b) and (b,a) are the same
    seen = set()
    unique_pairs = []
    for pair in pairs:
        key = tuple(sorted([pair["paper_a"], pair["paper_b"]]))
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    # Save everything
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    with open("clusters.json", "w", encoding="utf-8") as f:
        json.dump({"labels": cluster_labels, "pairs": unique_pairs}, f, indent=2)

    # Print summary
    print(f"\n✓ Saved clusters to 'clusters.json'")
    print(f"  {len(unique_pairs)} paper pairs queued for contradiction detection\n")
    print("── Cluster summary ─────────────────────────────────────\n")
    for cid in range(N_CLUSTERS):
        cluster_papers = [p for p in valid if p["cluster"] == cid]
        print(f"  [{cid}] {cluster_labels[cid]}")
        for p in cluster_papers[:3]:
            print(f"       · {p['title'][:65]}...")
        if len(cluster_papers) > 3:
            print(f"       · ... and {len(cluster_papers) - 3} more")
        print()


if __name__ == "__main__":
    main()
