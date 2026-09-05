"""Fetch Japanese sentences from llm-jp/magpie-sft-v1.0 (HF dataset).

Pulls assistant responses via the datasets-server /rows API, splits them into
sentences, and writes a plain-text corpus (one sentence per line) suitable
for make_gold.py.

Usage:
    python fetch_magpie.py [num_rows] [output.txt]

Defaults: 500 rows, references/data/magpie.txt
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

DATASET = "llm-jp/magpie-sft-v1.0"
API = "https://datasets-server.huggingface.co/rows"
PAGE = 100  # API max rows per request

MIN_LEN = 10
MAX_LEN = 200

CJK_RE = re.compile(r"[ぁ-んァ-ヶー一-鿿々〆]")


def fetch_rows(total: int) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while len(rows) < total:
        url = f"{API}?dataset={DATASET.replace('/', '%2F')}&config=default&split=train&offset={offset}&length={PAGE}"
        with urllib.request.urlopen(url, timeout=60) as res:
            payload = json.load(res)
        batch = payload.get("rows", [])
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    return rows[:total]


def sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        # Skip list markers, recipe bullets, code lines, pure-English lines.
        if not line or (line.startswith(("-", "*", "#", ">", "|", "```", "1.", "2.", "3.")) and not CJK_RE.search(line)):
            continue
        for sent in re.split(r"(?<=[。！？])\s*", line):
            sent = sent.strip()
            if MIN_LEN <= len(sent) <= MAX_LEN and CJK_RE.search(sent):
                out.append(sent)
    return out


def main() -> None:
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "data" / "magpie.txt"

    rows = fetch_rows(total)
    sents: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for msg in row["row"]["conversations"]:
            if msg["role"] != "assistant":
                continue
            for sent in sentences(msg["content"]):
                if sent not in seen:
                    seen.add(sent)
                    sents.append(sent)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sents) + "\n", encoding="utf-8")
    print(f"{len(rows)} conversations -> {len(sents)} sentences -> {out_path}")


if __name__ == "__main__":
    main()
