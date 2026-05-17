import requests
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime
 
# ── Configuration ──────────────────────────────────────────────
TOPIC = "large language models reasoning"
MAX_PAPERS = 100                  # ← how many papers to fetch
OUTPUT_FILE = "papers.json"       # ← where results are saved
# ───────────────────────────────────────────────────────────────
 
ARXIV_API = "http://export.arxiv.org/api/query"
NAMESPACE = "{http://www.w3.org/2005/Atom}"
 
 
def fetch_papers(topic: str, max_results: int = 100) -> list[dict]:
    """Fetch papers from arXiv and return a list of structured dicts."""
 
    papers = []
    batch_size = 50        # arXiv recommends batches of 50–100
    fetched = 0
 
    print(f"\nSearching arXiv for: '{topic}'")
    print(f"Target: {max_results} papers\n")
 
    while fetched < max_results:
        # How many to grab this batch
        this_batch = min(batch_size, max_results - fetched)
 
        params = {
            "search_query": f"ti:{topic}+AND+cat:cs.CL",
            "start": fetched,
            "max_results": this_batch,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
 
        print(f"  Fetching papers {fetched + 1}–{fetched + this_batch}...")
        response = requests.get(ARXIV_API, params=params, timeout=30)
        response.raise_for_status()
 
        # Parse the XML response arXiv returns
        root = ET.fromstring(response.content)
        entries = root.findall(f"{NAMESPACE}entry")
 
        if not entries:
            print("  No more papers found.")
            break
 
        for entry in entries:
            # Pull out each field safely
            title = entry.findtext(f"{NAMESPACE}title", "").strip().replace("\n", " ")
            abstract = entry.findtext(f"{NAMESPACE}summary", "").strip().replace("\n", " ")
            published = entry.findtext(f"{NAMESPACE}published", "")
            arxiv_id = entry.findtext(f"{NAMESPACE}id", "").split("/abs/")[-1]
 
            # Authors (there can be multiple)
            authors = [
                a.findtext(f"{NAMESPACE}name", "")
                for a in entry.findall(f"{NAMESPACE}author")
            ]
 
            # Categories / subject tags
            categories = [
                c.get("term", "")
                for c in entry.findall(f"{NAMESPACE}category")
            ]
 
            papers.append({
                "id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "published": published[:10],   # just the date, e.g. 2024-03-15
                "categories": categories,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "claim": None,                 # filled in Step 2
                "cluster": None,               # filled in Step 3
            })
 
        fetched += len(entries)
 
        # Be polite — arXiv asks for a 3-second delay between requests
        if fetched < max_results:
            time.sleep(3)
 
    return papers
 
 
def save_papers(papers: list[dict], filename: str) -> None:
    """Save papers list to a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
 
    print(f"\n✓ Saved {len(papers)} papers to '{filename}'")
 
 
def print_preview(papers: list[dict], n: int = 3) -> None:
    """Print a quick preview of the first n papers."""
    print(f"\n── Preview (first {n} papers) ──────────────────────────\n")
    for i, p in enumerate(papers[:n], 1):
        authors_str = ", ".join(p["authors"][:2])
        if len(p["authors"]) > 2:
            authors_str += f" + {len(p['authors']) - 2} more"
 
        print(f"  [{i}] {p['title']}")
        print(f"       {authors_str} · {p['published']}")
        print(f"       {p['abstract'][:120]}...")
        print(f"       {p['url']}\n")
 
 
if __name__ == "__main__":
    start = datetime.now()
 
    # Step 1: fetch
    papers = fetch_papers(topic=TOPIC, max_results=MAX_PAPERS)
 
    # Step 2: preview
    print_preview(papers)
 
    # Step 3: save
    save_papers(papers, OUTPUT_FILE)
 
    elapsed = (datetime.now() - start).seconds
    print(f"   Done in {elapsed}s — ready for Step 2 (claim extraction)\n")