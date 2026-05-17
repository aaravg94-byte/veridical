import json
import os
import time
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL        = "llama-3.3-70b-versatile"

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready.")


def groq(system, user, max_tokens=500):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers, json=payload, timeout=30
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def fetch_papers(topic, max_results=25):
    """Fetch papers from Semantic Scholar."""
    r = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": topic,
            "limit": max_results,
            "fields": "title,abstract,authors,year,externalIds",
        },
        headers={
            "User-Agent": "Veridical/1.0",
            "x-api-key": os.environ.get("SEMANTIC_SCHOLAR_KEY", ""),
        },
        timeout=30,
    )
    r.raise_for_status()
    papers = []
    for p in r.json().get("data", []):
        if not p.get("abstract"):
            continue
        arxiv_id = p.get("externalIds", {}).get("ArXiv")
        url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else f"https://semanticscholar.org/paper/{p.get('paperId','')}"
        papers.append({
            "id":        p.get("paperId", ""),
            "title":     p.get("title", ""),
            "authors":   [a["name"] for a in p.get("authors", [])],
            "abstract":  p.get("abstract", ""),
            "published": str(p.get("year", "")),
            "url":       url,
        })
    return papers


def extract_claim(paper):
    raw = groq(
        system="""Extract the single most important claim from this paper abstract.
Respond ONLY with valid JSON:
{"claim": "one sentence", "methodology": "empirical study|theoretical|survey|benchmark|other", "confidence": "high|medium|low"}""",
        user=f"Title: {paper['title']}\nAbstract: {paper['abstract']}",
        max_tokens=150,
    )
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def compare_claims(pa, pb):
    raw = groq(
        system="""Compare two scientific claims.
Respond ONLY with valid JSON:
{"verdict": "agree|disagree|unrelated", "reason": "one sentence"}""",
        user=f"Claim A: {pa['claim']}\nClaim B: {pb['claim']}",
        max_tokens=100,
    )
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data  = request.json
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "No topic provided"}), 400

    try:
        # 1. Fetch papers
        papers = fetch_papers(topic, max_results=25)
        if not papers:
            return jsonify({"error": "No papers found for this topic. Try a different search term."}), 404

        papers = papers[:20]

        # 2. Extract claims
        valid = []
        for p in papers:
            try:
                result = extract_claim(p)
                p["claim"]       = result.get("claim", "")
                p["methodology"] = result.get("methodology", "")
                p["confidence"]  = result.get("confidence", "")
                valid.append(p)
                time.sleep(0.5)
            except Exception:
                continue

        if len(valid) < 3:
            return jsonify({"error": "Not enough papers with extractable claims. Try a more specific topic."}), 400

        # 3. Embed + cluster
        claims    = [p["claim"] for p in valid]
        matrix    = np.array(embedder.encode(claims))
        n_clusters = min(4, len(valid) // 3)
        kmeans    = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        ids       = kmeans.fit_predict(matrix)
        for i, p in enumerate(valid):
            p["cluster"] = int(ids[i])

        # Label clusters
        cluster_labels = {}
        for cid in range(n_clusters):
            cp     = [p for p in valid if p["cluster"] == cid]
            titles = "\n".join(f"- {p['title']}" for p in cp[:4])
            label  = groq(
                system="Give this research cluster a 2-4 word label. Respond with ONLY the label, nothing else.",
                user=f"Papers:\n{titles}",
                max_tokens=15,
            )
            cluster_labels[cid] = label.strip()

        # 4. Contradiction detection
        sim = cosine_similarity(matrix)
        np.fill_diagonal(sim, -1)
        contradictions = []
        agreements     = []
        seen = set()
        for i in range(len(valid)):
            j   = int(np.argmax(sim[i]))
            key = tuple(sorted([i, j]))
            if key in seen or valid[i]["cluster"] != valid[j]["cluster"]:
                continue
            seen.add(key)
            try:
                result  = compare_claims(valid[i], valid[j])
                verdict = result.get("verdict", "unrelated")
                reason  = result.get("reason", "")
                if verdict == "disagree":
                    contradictions.append({
                        "title_a": valid[i]["title"], "claim_a": valid[i]["claim"],
                        "title_b": valid[j]["title"], "claim_b": valid[j]["claim"],
                        "reason": reason,
                    })
                elif verdict == "agree":
                    agreements.append({
                        "title_a": valid[i]["title"],
                        "title_b": valid[j]["title"],
                        "reason": reason,
                    })
                time.sleep(0.5)
            except Exception:
                continue

        # 5. Generate report
        all_claims = "\n".join(f"- {p['claim']}" for p in valid[:15])
        summary = groq(
            system="Write a 3-sentence executive summary of this research field. Be specific and direct. No bullet points.",
            user=f"Topic: {topic}\n\nClaims:\n{all_claims}",
            max_tokens=200,
        )
        verdict = groq(
            system="Write a 2-sentence frank verdict: what is settled, what is contested in this field. Be direct.",
            user=f"Topic: {topic}\nSummary: {summary}\nContradictions: {len(contradictions)}\nAgreements: {len(agreements)}",
            max_tokens=150,
        )

        # Build clusters output
        clusters_out = []
        for cid in range(n_clusters):
            cp = [p for p in valid if p["cluster"] == cid]
            clusters_out.append({
                "label":  cluster_labels.get(cid, f"Cluster {cid}"),
                "papers": [{"title": p["title"], "claim": p["claim"], "url": p["url"], "published": p["published"]} for p in cp],
            })

        return jsonify({
            "topic":          topic,
            "paper_count":    len(valid),
            "summary":        summary,
            "verdict":        verdict,
            "clusters":       clusters_out,
            "contradictions": contradictions,
            "agreements":     agreements,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/")
def index():
    return app.send_static_file("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
