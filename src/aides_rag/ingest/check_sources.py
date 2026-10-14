"""Check that the source URLs cited by the corpus still resolve (network required, not part of the tests).

Usage:
    uv run python -m aides_rag.ingest.check_sources --corpus data/processed/aids.jsonl --out data/processed/source_check.json
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HEADERS = {"User-Agent": "Mozilla/5.0 (aides-rag source check; student project)"}


def status(url: str, timeout: float = 15) -> str:
    try:
        req = urllib.request.Request(url, headers=HEADERS, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return str(resp.status)
    except urllib.error.HTTPError as e:
        return str(e.code)
    except Exception as e:  # DNS, TLS, timeout
        return type(e).__name__


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/processed/aids.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/source_check.json"))
    args = ap.parse_args()

    urls = sorted({u for line in args.corpus.read_text(encoding="utf-8").splitlines() for u in json.loads(line)["source_urls"]})
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = dict(zip(urls, pool.map(status, urls)))
    summary = Counter("ok" if s.startswith(("2", "3")) else s for s in results.values())
    args.out.write_text(json.dumps({"summary": summary, "results": results}, indent=2) + "\n", encoding="utf-8")
    print(f"{len(urls)} unique URLs: {dict(summary)}")


if __name__ == "__main__":
    main()
