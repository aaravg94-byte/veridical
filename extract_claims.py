import json
import time
import os
import requests

# ── Configuration ──────────────────────────────────────────────
INPUT_FILE  = "papers.json"
OUTPUT_FILE = "papers.json"   # overwrites in place, adding claims
MODEL       = "llama-3.3-70b-versatile"
DELAY       = 1.5             # seconds between API calls (be polite)
# ───────────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise SystemExit("Error: GROQ_API_KEY not set. Run: export GROQ_API_KEY='your-key'")

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

SYSTEM_PROMPT = """You are a scientific claim extractor. Given a paper abstract, extract the single most important claim the paper makes.

Respond ONLY with valid JSON in exactly this format, nothing else:
{
  "claim": "one sentence stating the main finding or argument",
  "methodology": "empirical study | theoretical | survey | benchmark | other",
  "confidence": "high | medium | low",
  "keywords": ["keyword1", "keyword2", "keyword3"]
}

Rules:
- claim must be one clear, specific sentence
- do not start with "The paper" or "This paper"
- make the claim stand alone without needing to read the paper
- confidence is how strongly the paper supports the claim"""


def extract_claim(abstract: str, title: str) -> dict:
    """Send one abstract to Groq and return structured claim."""

    user_message = f"Title: {title}\n\nAbstract: {abstract}"

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.1,   # low = more consistent outputs
        "max_tokens": 200,
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=HEADERS,
        json=payload,
        timeout=30,
    )
    response.raise_for_status()

    raw = response.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown fences if model adds them
    raw = raw.replace("```json", "").replace("```", "").strip()

    return json.loads(raw)


def main():
    # Load papers
    with open(INPUT_FILE, encoding="utf-8") as f:
        papers = json.load(f)

    # Only process papers that don't have a claim yet
    todo = [p for p in papers if p.get("claim") is None]
    print(f"\nExtracting claims from {len(todo)} papers...\n")

    success = 0
    errors  = 0

    for i, paper in enumerate(papers):
        if paper.get("claim") is not None:
            continue  # already done, skip

        print(f"  [{i+1}/{len(papers)}] {paper['title'][:70]}...")

        try:
            result = extract_claim(paper["abstract"], paper["title"])
            paper["claim"]       = result.get("claim")
            paper["methodology"] = result.get("methodology")
            paper["confidence"]  = result.get("confidence")
            paper["keywords"]    = result.get("keywords", [])
            print(f"         → {paper['claim'][:80]}...")
            success += 1

        except Exception as e:
            print(f"         ✗ Error: {e}")
            paper["claim"] = None
            errors += 1

        # Save after every paper — so if it crashes you don't lose progress
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)

        time.sleep(DELAY)

    print(f"\n✓ Done — {success} claims extracted, {errors} errors")
    print(f"  Saved to '{OUTPUT_FILE}'\n")

    # Preview 3 claims
    print("── Preview ─────────────────────────────────────────────\n")
    for p in [x for x in papers if x.get("claim")][:3]:
        print(f"  Title:  {p['title'][:65]}...")
        print(f"  Claim:  {p['claim']}")
        print(f"  Method: {p['methodology']} | Confidence: {p['confidence']}")
        print(f"  Keys:   {', '.join(p['keywords'])}\n")


if __name__ == "__main__":
    main()
