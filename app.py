
Copy

import json
import os
import time
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)
 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-2.0-flash"
 
print("Ready.")
 
 
def gemini(prompt, max_tokens=500):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_API_KEY}",
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": max_tokens},
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
 
 
def fetch_papers(topic, max_results=20):
    r = requests.get(
        "https://api.openalex.org/works",
        params={
            "search": topic,
            "per-page": max_results,
            "filter": "has_abstract:true",
            "sort": "relevance_score:desc",
            "mailto": "veridical@example.com",
        },
        timeout=30,
    )
    r.raise_for_status()
    papers = []
    for p in r.json().get("results", []):
        abstract_raw = p.get("abstract_inverted_index")
        if not abstract_raw:
            continue
        words = sorted(abstract_raw.items(), key=lambda x: x[1][0])
        abstract = " ".join(w for w, _ in words)
        doi = p.get("doi", "")
        papers.append({
            "id": p.get("id", ""),
            "title": p.get("title", ""),
            "authors": [a["author"]["display_name"] for a in p.get("authorships", [])[:5]],
            "abstract": abstract,
            "published": str(p.get("publication_year", "")),
            "url": doi if doi else p.get("id", ""),
        })
    return papers
 
 
def extract_claim(paper):
    prompt = f"""Extract the single most important claim from this paper abstract.
Respond ONLY with valid JSON, no markdown:
{{"claim": "one sentence", "methodology": "empirical study|theoretical|survey|benchmark|other", "confidence": "high|medium|low"}}
 
Title: {paper['title']}
Abstract: {paper['abstract'][:500]}"""
    raw = gemini(prompt, max_tokens=150)
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)
 
 
def compare_claims(claim_a, claim_b):
    prompt = f"""Compare these two scientific claims and determine their relationship.
Respond ONLY with valid JSON, no markdown:
{{"verdict": "agree|disagree|unrelated", "reason": "one sentence"}}
 
Claim A: {claim_a}
Claim B: {claim_b}"""
    raw = gemini(prompt, max_tokens=100)
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)
 
 
@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.json
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "No topic provided"}), 400
 
    try:
        papers = fetch_papers(topic, max_results=20)
        if not papers:
            return jsonify({"error": "No papers found. Try a different topic."}), 404
        papers = papers[:20]
 
        valid = []
        for p in papers:
            try:
                result = extract_claim(p)
                p["claim"] = result.get("claim", "")
                p["methodology"] = result.get("methodology", "")
                p["confidence"] = result.get("confidence", "")
                if p["claim"]:
                    valid.append(p)
                time.sleep(0.3)
            except Exception:
                continue
 
        if len(valid) < 3:
            return jsonify({"error": "Not enough extractable claims. Try a more specific topic."}), 400
 
        claims = [p["claim"] for p in valid]
        vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        matrix = vectorizer.fit_transform(claims).toarray()
        n_clusters = min(4, len(valid) // 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        ids = kmeans.fit_predict(matrix)
        for i, p in enumerate(valid):
            p["cluster"] = int(ids[i])
 
        cluster_labels = {}
        for cid in range(n_clusters):
            cp = [p for p in valid if p["cluster"] == cid]
            titles = "\n".join(f"- {p['title']}" for p in cp[:4])
            label = gemini(f"Give this research cluster a 2-4 word label. Respond with ONLY the label, nothing else.\n\nPapers:\n{titles}", max_tokens=15)
            cluster_labels[cid] = label.strip()
            time.sleep(0.3)
 
        sim = cosine_similarity(matrix)
        np.fill_diagonal(sim, -1)
        contradictions = []
        agreements = []
        seen = set()
        for i in range(len(valid)):
            j = int(np.argmax(sim[i]))
            key = tuple(sorted([i, j]))
            if key in seen or valid[i]["cluster"] != valid[j]["cluster"]:
                continue
            seen.add(key)
            try:
                result = compare_claims(valid[i]["claim"], valid[j]["claim"])
                verdict = result.get("verdict", "unrelated")
                reason = result.get("reason", "")
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
                time.sleep(0.3)
            except Exception:
                continue
 
        all_claims = "\n".join(f"- {p['claim']}" for p in valid[:15])
        summary = gemini(
            f"Write a 3-sentence executive summary of this research field. Be specific and direct. No bullet points.\n\nTopic: {topic}\n\nClaims:\n{all_claims}",
            max_tokens=200,
        )
        verdict_text = gemini(
            f"Write a 2-sentence frank verdict: what is settled, what is contested. Be direct.\n\nTopic: {topic}\nSummary: {summary}\nContradictions: {len(contradictions)}\nAgreements: {len(agreements)}",
            max_tokens=150,
        )
 
        clusters_out = []
        for cid in range(n_clusters):
            cp = [p for p in valid if p["cluster"] == cid]
            clusters_out.append({
                "label": cluster_labels.get(cid, f"Cluster {cid}"),
                "papers": [{"title": p["title"], "claim": p["claim"], "url": p["url"], "published": p["published"]} for p in cp],
            })
 
        return jsonify({
            "topic": topic,
            "paper_count": len(valid),
            "summary": summary,
            "verdict": verdict_text,
            "clusters": clusters_out,
            "contradictions": contradictions,
            "agreements": agreements,
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
 