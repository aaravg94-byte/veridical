import json
import os
import time
import requests
import networkx as nx

# ── Configuration ──────────────────────────────────────────────
PAPERS_FILE   = "papers.json"
CLUSTERS_FILE = "clusters.json"
GRAPH_FILE    = "graph.json"
DELAY         = 1.0   # seconds between API calls
# ───────────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise SystemExit("Error: GROQ_API_KEY not set. Run: export GROQ_API_KEY='your-key'")

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}


def compare_claims(claim_a: str, claim_b: str, title_a: str, title_b: str) -> dict:
    """Ask Groq whether two claims agree, disagree, or are unrelated."""

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": """You compare two scientific claims and determine their relationship.
Respond ONLY with valid JSON in exactly this format, nothing else:
{
  "verdict": "agree" | "disagree" | "unrelated",
  "confidence": "high" | "medium" | "low",
  "reason": "one sentence explaining why"
}

Rules:
- agree: both claims point to the same conclusion
- disagree: the claims contradict or tension each other
- unrelated: the claims are about different enough topics that comparison is meaningless
- be willing to call disagreements — science has real contradictions"""
            },
            {
                "role": "user",
                "content": f"Paper A: {title_a}\nClaim A: {claim_a}\n\nPaper B: {title_b}\nClaim B: {claim_b}"
            }
        ],
        "temperature": 0.1,
        "max_tokens": 150,
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=HEADERS,
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def compute_evidence_score(paper_id: str, edges: list[dict]) -> dict:
    """Score a paper by how much support vs contradiction it has."""
    supporting  = [e for e in edges if paper_id in [e["a"], e["b"]] and e["verdict"] == "agree"]
    contradicting = [e for e in edges if paper_id in [e["a"], e["b"]] and e["verdict"] == "disagree"]

    support_score = len(supporting)
    conflict_score = len(contradicting)
    total = support_score + conflict_score

    if total == 0:
        consensus_strength = 0.5
    else:
        consensus_strength = round(support_score / total, 2)

    return {
        "support_count": support_score,
        "conflict_count": conflict_score,
        "consensus_strength": consensus_strength,   # 1.0 = full consensus, 0.0 = fully contested
    }


def main():
    # Load data
    with open(PAPERS_FILE, encoding="utf-8") as f:
        papers = json.load(f)

    with open(CLUSTERS_FILE, encoding="utf-8") as f:
        cluster_data = json.load(f)

    pairs  = cluster_data["pairs"]
    labels = cluster_data["labels"]

    # Build a lookup by paper ID
    paper_lookup = {p["id"]: p for p in papers}

    print(f"\nDetecting contradictions across {len(pairs)} paper pairs...\n")

    edges = []
    skipped = 0

    for i, pair in enumerate(pairs):
        pa = paper_lookup.get(pair["paper_a"])
        pb = paper_lookup.get(pair["paper_b"])

        if not pa or not pb:
            skipped += 1
            continue

        claim_a = pa.get("claim", "")
        claim_b = pb.get("claim", "")

        if not claim_a or not claim_b or claim_a == "skipped" or claim_b == "skipped":
            skipped += 1
            continue

        print(f"  [{i+1}/{len(pairs)}] Comparing:")
        print(f"    A: {pa['title'][:60]}...")
        print(f"    B: {pb['title'][:60]}...")

        try:
            result = compare_claims(claim_a, claim_b, pa["title"], pb["title"])
            verdict    = result.get("verdict", "unrelated")
            confidence = result.get("confidence", "low")
            reason     = result.get("reason", "")

            edges.append({
                "a":          pa["id"],
                "b":          pb["id"],
                "title_a":    pa["title"],
                "title_b":    pb["title"],
                "claim_a":    claim_a,
                "claim_b":    claim_b,
                "cluster":    pair["cluster"],
                "verdict":    verdict,
                "confidence": confidence,
                "reason":     reason,
            })

            icon = {"agree": "✓", "disagree": "✗", "unrelated": "~"}[verdict]
            print(f"    {icon} {verdict.upper()} ({confidence}) — {reason[:80]}\n")

        except Exception as e:
            print(f"    ? Error: {e}\n")
            skipped += 1

        time.sleep(DELAY)

    # Build NetworkX graph
    G = nx.Graph()

    for p in papers:
        if p.get("claim") and p["claim"] != "skipped":
            G.add_node(p["id"], title=p["title"], claim=p["claim"],
                       cluster=p.get("cluster"), methodology=p.get("methodology"),
                       confidence=p.get("confidence"))

    for edge in edges:
        if edge["verdict"] != "unrelated":
            G.add_edge(edge["a"], edge["b"],
                       verdict=edge["verdict"],
                       confidence=edge["confidence"],
                       reason=edge["reason"])

    # Compute evidence scores for each paper
    for paper in papers:
        if paper.get("claim") and paper["claim"] != "skipped":
            score = compute_evidence_score(paper["id"], edges)
            paper["evidence_score"] = score

    # Save updated papers
    with open(PAPERS_FILE, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    # Save graph
    graph_data = {
        "nodes": [{"id": n, **G.nodes[n]} for n in G.nodes],
        "edges": edges,
        "cluster_labels": labels,
        "stats": {
            "total_pairs":     len(pairs),
            "agreements":      len([e for e in edges if e["verdict"] == "agree"]),
            "contradictions":  len([e for e in edges if e["verdict"] == "disagree"]),
            "unrelated":       len([e for e in edges if e["verdict"] == "unrelated"]),
            "skipped":         skipped,
        }
    }

    with open(GRAPH_FILE, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    # Print summary
    stats = graph_data["stats"]
    print("\n── Evidence graph summary ──────────────────────────────\n")
    print(f"  Nodes (papers):     {G.number_of_nodes()}")
    print(f"  Edges (relations):  {G.number_of_edges()}")
    print(f"  Agreements:         {stats['agreements']}")
    print(f"  Contradictions:     {stats['contradictions']}")
    print(f"  Unrelated:          {stats['unrelated']}")

    # Most contested claims
    contested = sorted(
        [p for p in papers if p.get("evidence_score")],
        key=lambda p: p["evidence_score"]["conflict_count"],
        reverse=True
    )[:3]

    print("\n── Most contested claims ───────────────────────────────\n")
    for p in contested:
        sc = p["evidence_score"]
        print(f"  {p['title'][:65]}...")
        print(f"  Claim: {p['claim'][:80]}...")
        print(f"  Support: {sc['support_count']} | Conflicts: {sc['conflict_count']} | Consensus strength: {sc['consensus_strength']}\n")

    print(f"✓ Saved evidence graph to '{GRAPH_FILE}'")
    print(f"  Ready for Step 5 — report generation\n")


if __name__ == "__main__":
    main()
