# Veridical

An automated research intelligence pipeline that fetches scientific papers, extracts structured claims, detects contradictions, and generates weekly "state of the field" reports.

Built with Python, Groq (Llama 3.3 70B), and sentence-transformers.

---

## What it does

1. **Fetches papers** from arXiv by topic (100+ papers per run)
2. **Extracts structured claims** from each abstract using an LLM — methodology type, confidence level, and keywords
3. **Embeds and clusters** claims into topic groups using `all-MiniLM-L6-v2` sentence embeddings + KMeans
4. **Detects contradictions** by comparing claim pairs within clusters and building a weighted evidence graph
5. **Generates a report** summarising the field — what's settled, what's contested, and what's still open

---

## Example output

```
## Field Verdict

Recent research in AI/ML has yielded significant advancements in visual reasoning,
video generation, and NLP. Frameworks such as ATLAS and RefDecoder achieved superior
benchmark performance, while limitations in adaptive agent capabilities were
highlighted — the best-performing agent achieving only 25% accuracy in world event
prediction. Two direct contradictions were detected: unified embodied models claiming
strong cross-task performance were challenged by evaluation frameworks showing
persistent failures in fine-grained visual memory. The field is advancing rapidly
but lacks standardised evaluation — many claimed state-of-the-art results are not
directly comparable.
```

---

## Sample report structure

```
# Veridical — State of the Field Report
Topic: AI/ML Research (LLM Reasoning & Related)
Papers analyzed: 47

## Executive Summary
## Research Clusters (6 topic groups)
## Contradictions & Tensions
## Evidence Graph Stats
## Most Contested Claims
## Field Verdict
```

---

## Tech stack

| Component | Tool |
|-----------|------|
| Paper ingestion | arXiv API |
| Claim extraction | Groq API (Llama 3.3 70B) |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2, local) |
| Clustering | scikit-learn KMeans |
| Evidence graph | NetworkX |
| Report generation | Groq API (Llama 3.3 70B) |

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/veridical
cd veridical
```

**2. Create a virtual environment**
```bash
python3 -m venv research-env
source research-env/bin/activate
```

**3. Install dependencies**
```bash
pip install requests scikit-learn numpy networkx sentence-transformers
```

**4. Get a free Groq API key**

Sign up at [console.groq.com](https://console.groq.com) — no credit card required.

```bash
export GROQ_API_KEY="your-key-here"
```

---

## Usage

Run each step in order:

```bash
# Step 1 — fetch papers from arXiv
python3 fetch_papers.py

# Step 2 — extract structured claims with LLM
python3 extract_claims.py

# Step 3 — embed and cluster claims by topic
python3 cluster_claims.py

# Step 4 — detect contradictions, build evidence graph
python3 detect_contradictions.py

# Step 5 — generate state-of-the-field report
python3 generate_report.py
```

Output: `report.md` — a full Markdown report ready to read or publish.

**To change the topic**, edit the `TOPIC` variable at the top of `fetch_papers.py`:

```python
TOPIC = "CRISPR gene editing"   # or any research topic
```

---

## Output files

| File | Contents |
|------|----------|
| `papers.json` | All papers with extracted claims, methodology, confidence, cluster ID, evidence scores |
| `clusters.json` | Topic cluster labels + paper pairs queued for comparison |
| `graph.json` | Full evidence graph — nodes (papers), edges (agree/disagree/unrelated), stats |
| `report.md` | Final generated report |

---

## Project structure

```
veridical/
├── fetch_papers.py          # Step 1: arXiv ingestion
├── extract_claims.py        # Step 2: LLM claim extraction
├── cluster_claims.py        # Step 3: embedding + clustering
├── detect_contradictions.py # Step 4: contradiction detection + graph
├── generate_report.py       # Step 5: report generation
├── papers.json              # intermediate data
├── clusters.json            # intermediate data
├── graph.json               # evidence graph
└── report.md                # final output
```

---

## Limitations

- Clustering quality depends on how topically focused the paper set is — broad topics produce broader clusters
- Contradiction detection is LLM-based and not infallible; low-confidence verdicts should be treated as signals, not facts
- arXiv coverage is strong for CS/ML/physics but limited for clinical medicine (use PubMed API for biomedical topics)
- Free Groq tier has rate limits — the pipeline includes delays to stay within them

---

## Future work

- [ ] PubMed integration for biomedical papers
- [ ] Automated weekly scheduling (cron + email delivery)
- [ ] Web dashboard for interactive graph exploration
- [ ] Consensus shift tracking across multiple weekly runs
- [ ] PDF full-text parsing (currently abstracts only)

---

*Built with Python · Groq · sentence-transformers · arXiv API*
