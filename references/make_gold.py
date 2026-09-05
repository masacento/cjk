"""Build a litsea-format gold corpus with fugashi (UniDic or IPAdic).

Segments each input line with MeCab (fugashi) and writes one sentence per
line with words separated by single spaces — the corpus format litsea's
extract/train pipeline expects.

Long compounds: MeCab/UniDic splits compounds into short morphemes
(国立/情報/学/研究/所). --merge-min-chars joins each maximal run of adjacent
morphemes whose major POS class is in --merge-pos back into one word when
the run's total length reaches the threshold, so scoring is not dominated
by compound granularity.

Usage:
    python make_gold.py input.txt gold.txt
    python make_gold.py input.txt gold.txt --merge-min-chars 4
    python make_gold.py input.txt gold.txt --dict ipadic --merge-min-chars 0
"""

import argparse
import sys
from pathlib import Path

from fugashi import Tagger

DEFAULT_MERGE_POS = "名詞,接尾辞,接頭詞"


def token_pos(tok) -> str:
    return getattr(tok.feature, "pos1", "") or ""


def tokenize_merged(text: str, tagger: Tagger, merge_min_chars: int, merge_pos: set[str]) -> list[str]:
    words: list[str] = []
    run: list[str] = []

    def flush():
        if not run:
            return
        if merge_min_chars > 0 and sum(len(w) for w in run) >= merge_min_chars:
            words.append("".join(run))
        else:
            words.extend(run)
        run.clear()

    for tok in tagger(text):
        if merge_min_chars > 0 and token_pos(tok) in merge_pos:
            run.append(str(tok))
        else:
            flush()
            words.append(str(tok))
    flush()
    return words


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="plain text, one sentence per line")
    ap.add_argument("output", type=Path, help="space-separated corpus (litsea format)")
    ap.add_argument("--dict", choices=["unidic", "ipadic"], default="unidic", help="MeCab dictionary (default: unidic)")
    ap.add_argument("--merge-min-chars", type=int, default=5,
                    help="join each maximal run of mergeable adjacent morphemes into one word "
                         "when the run's total length is >= N characters; 0 disables merging (default: 5)")
    ap.add_argument("--merge-pos", default=DEFAULT_MERGE_POS,
                    help=f"comma-separated major POS classes eligible for merging (default: {DEFAULT_MERGE_POS})")
    args = ap.parse_args()

    if args.dict == "ipadic":
        try:
            import ipadic  # pip install ipadic
            tagger = Tagger(f"-d {ipadic.DICDIR}")
        except ImportError:
            sys.exit("IPAdic requires: pip install ipadic")
    else:
        try:
            tagger = Tagger()
        except RuntimeError:
            sys.exit("UniDic requires: pip install unidic-lite (or unidic + python -m unidic download)")

    merge_pos = {p.strip() for p in args.merge_pos.split(",") if p.strip()}
    total = word_total = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            words = tokenize_merged(line, tagger, args.merge_min_chars, merge_pos)
            word_total += len(words)
            fout.write(" ".join(words) + "\n")

    print(f"wrote {total} sentences, {word_total} words -> {args.output}"
          f" (dict={args.dict}, merge>={args.merge_min_chars} chars on {sorted(merge_pos)})")


if __name__ == "__main__":
    main()
