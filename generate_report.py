import json
import os
import requests
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────
PAPERS_FILE  = "papers.json"
GRAPH_FILE   = "graph.json"
OUTPUT_FILE  = "report.md"
# ───────────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise SystemExit("Error: GROQ_API_KEY not set. Run: export GROQ_API_KEY='your-key'")

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}


def call_groq(system: str, user: str, max_tokens: int = 1000) -> str:
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=HEADERS,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def main():
    with open(PAPERS_FILE, encoding="utf-8") as f:
        papers = json.load(f)

    with open(GRAPH_FILE, encoding="utf-8") as f:
        graph = json.load(f)

    valid_papers = [p for p in papers if p.get("claim") and p["claim"] != "skipped"]
    cluster_labels = graph["cluster_labels"]
    edges = graph["edges"]
    stats = graph["stats"]

    contradictions = [e for e in edges if e["verdict"] == "disagree"]
    agreements     = [e for e in edges if e["verdict"] == "agree"]

    print("\nGenerating state-of-the-field report...\n")

    # ── Section 1: Executive summary ──────────────────────────
    print("  Writing executive summary...")
    all_claims = "\n".join(f"- {p['claim']}" for p in valid_papers[:20])
    summary = call_groq(
        system="You are a scientific analyst writing a clear, authoritative executive summary of a research field. Write 3-4 sentences. No bullet points. Be direct and specific.",
        user=f"Based on these recent research claims from {len(valid_papers)} papers, write an executive summary of the current state of this field:\n\n{all_claims}",
        max_tokens=300,
    )

    # ── Section 2: Cluster summaries ──────────────────────────
    print("  Writing cluster summaries...")
    cluster_summaries = {}
    for cid_str, label in cluster_labels.items():
        cid = int(cid_str)
        cluster_papers = [p for p in valid_papers if p.get("cluster") == cid]
        if not cluster_papers:
            continue
        claims_text = "\n".join(f"- {p['claim']}" for p in cluster_papers)
        summary_text = call_groq(
            system="You summarize a cluster of related research claims in 2 sentences. Be specific. No bullet points.",
            user=f"Cluster: {label}\n\nClaims:\n{claims_text}\n\nSummarize what this cluster of research collectively shows:",
            max_tokens=200,
        )
        cluster_summaries[cid] = {"label": label, "summary": summary_text, "count": len(cluster_papers)}

    # ── Section 3: Contradictions ──────────────────────────────
    print("  Analysing contradictions...")
    contradiction_section = ""
    if contradictions:
        for c in contradictions:
            contradiction_section += f"**{c['title_a']}** vs **{c['title_b']}**\n"
            contradiction_section += f"- Claim A: {c['claim_a']}\n"
            contradiction_section += f"- Claim B: {c['claim_b']}\n"
            contradiction_section += f"- Reason: {c['reason']}\n\n"
    else:
        contradiction_section = "No direct contradictions detected in this batch. Papers largely address distinct sub-problems within the field.\n"

    # ── Section 4: Verdict ─────────────────────────────────────
    print("  Writing field verdict...")
    contested = sorted(
        [p for p in valid_papers if p.get("evidence_score")],
        key=lambda p: p["evidence_score"]["conflict_count"],
        reverse=True
    )[:5]
    contested_text = "\n".join(f"- {p['title']}: {p['claim']}" for p in contested)

    verdict = call_groq(
        system="You are a senior scientist giving a frank, one-paragraph verdict on a research field. Be direct. Highlight what is settled, what is contested, and what the field still needs to figure out. No fluff.",
        user=f"Field summary: {summary}\n\nMost contested claims:\n{contested_text}\n\nContradictions found: {len(contradictions)}\nAgreements found: {len(agreements)}\n\nWrite a frank verdict on the current state of this field:",
        max_tokens=300,
    )

    # ── Assemble report ────────────────────────────────────────
    date_str = datetime.now().strftime("%B %d, %Y")
    report = f"""# Veridical — State of the Field Report
**Topic:** AI/ML Research (LLM Reasoning & Related)
**Generated:** {date_str}
**Papers analyzed:** {len(valid_papers)}

---

## Executive Summary

{summary}

---

## Research Clusters

*{len(valid_papers)} papers grouped into {len(cluster_summaries)} topic clusters.*

"""
    for cid, data in cluster_summaries.items():
        cluster_papers = [p for p in valid_papers if p.get("cluster") == cid]
        report += f"### {data['label']} ({data['count']} papers)\n\n"
        report += f"{data['summary']}\n\n"
        report += "**Papers in this cluster:**\n"
        for p in cluster_papers:
            report += f"- [{p['title']}]({p['url']}) — {p['claim'][:80]}...\n"
        report += "\n"

    report += f"""---

## Contradictions & Tensions

*{len(contradictions)} contradictions detected across {stats['total_pairs']} paper pairs.*

{contradiction_section}
---

## Evidence Graph Stats

| Metric | Value |
|--------|-------|
| Papers analyzed | {len(valid_papers)} |
| Pairs compared | {stats['total_pairs']} |
| Agreements | {stats['agreements']} |
| Contradictions | {stats['contradictions']} |
| Unrelated pairs | {stats['unrelated']} |

---

## Most Contested Claims

"""
    for p in contested:
        sc = p["evidence_score"]
        report += f"**{p['title']}**\n"
        report += f"- Claim: {p['claim']}\n"
        report += f"- Support: {sc['support_count']} | Conflicts: {sc['conflict_count']} | Consensus strength: {sc['consensus_strength']}\n\n"

    report += f"""---

## Field Verdict

{verdict}

---

*Generated by Veridical — automated research intelligence pipeline*
*Sources: arXiv | Model: Llama 3.3 70B via Groq | Embeddings: all-MiniLM-L6-v2*
"""

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✓ Report saved to '{OUTPUT_FILE}'")
    print(f"  Open it in any Markdown viewer or paste into notion.so\n")
    print("── Preview ─────────────────────────────────────────────\n")
    print(report[:800])
    print("...\n")


if __name__ == "__main__":
    main()
