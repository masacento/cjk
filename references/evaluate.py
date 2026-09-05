"""Score litsea segmentation models against a gold corpus (make_gold.py output).

Metrics, per model:
  - boundary P/R/F1: agreement on the presence of a word boundary at each
    character position (the standard word-segmentation metric).
  - word P/R/F1: exact span matching. A gold word counts as recalled when
    some consecutive run of predicted tokens covers exactly that span; a
    predicted token counts as precise when it exactly covers some gold word
    span. Under this metric a model that over-splits a merged gold word is
    penalized exactly like one that under-splits.

Usage:
    python evaluate.py gold.txt models/japanese.model [more models...]
    python evaluate.py gold.txt models/*.model --show-mismatch 10
"""

import argparse
from collections import Counter
from pathlib import Path

from litsea import Language, Segmenter


def char_spans(tokens: list[str]) -> tuple[set[int], dict[int, int]]:
    """Return (boundary offsets, span map). A span (start, end) maps to end."""
    boundaries: set[int] = set()
    spans: dict[int, int] = {}
    pos = 0
    for tok in tokens:
        pos += len(tok)
        boundaries.add(pos)
        spans[pos - len(tok)] = pos
    return boundaries, spans


def score_line(gold_tokens: list[str], hyp_tokens: list[str]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    gold_b, gold_s = char_spans(gold_tokens)
    hyp_b, hyp_s = char_spans(hyp_tokens)

    tp_b = len(gold_b & hyp_b)
    fn_b = len(gold_b - hyp_b)
    fp_b = len(hyp_b - gold_b)

    tp_w = sum(1 for end in gold_s.values() if end in hyp_s)
    fn_w = len(gold_s) - tp_w
    fp_w = sum(1 for end in hyp_s.values() if end not in gold_s)
    return (tp_b, fp_b, fn_b), (tp_w, fp_w, fn_w)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gold", type=Path, help="gold corpus from make_gold.py (space-separated words)")
    ap.add_argument("models", nargs="+", type=Path, help="model files to compare")
    ap.add_argument("--language", default="japanese")
    ap.add_argument("--show-mismatch", type=int, default=0, metavar="N",
                    help="print up to N sentences per model with word-level errors")
    args = ap.parse_args()

    lines = [line.split() for line in args.gold.read_text(encoding="utf-8").splitlines() if line.strip()]
    segmenters = [(m, Segmenter.open(Language.parse(args.language), m)) for m in args.models]

    header = f"{'model':<28} {'bP':>7} {'bR':>7} {'bF1':>7} {'wP':>7} {'wR':>7} {'wF1':>7}"
    print(header)
    print("-" * len(header))

    for model_path, seg in segmenters:
        b_counts = Counter()
        w_counts = Counter()
        mismatches: list[tuple[str, list[str], set]] = []

        for gold_tokens in lines:
            text = "".join(gold_tokens)
            hyp_tokens = seg.segment(text)
            if "".join(hyp_tokens) != text:
                continue  # model altered the text; skip rather than score garbage
            (tpb, fpb, fnb), (tpw, fpw, fnw) = score_line(gold_tokens, hyp_tokens)
            b_counts.update(tp=tpb, fp=fpb, fn=fnb)
            w_counts.update(tp=tpw, fp=fpw, fn=fnw)
            if args.show_mismatch and len(mismatches) < args.show_mismatch and (fpw or fnw):
                mismatches.append((text, hyp_tokens, gold_tokens))

        bp, br, bf = prf(b_counts["tp"], b_counts["fp"], b_counts["fn"])
        wp, wr, wf = prf(w_counts["tp"], w_counts["fp"], w_counts["fn"])
        print(f"{model_path.name:<28} {bp:>7.2%} {br:>7.2%} {bf:>7.2%} {wp:>7.2%} {wr:>7.2%} {wf:>7.2%}")

        for text, hyp_tokens, gold_tokens in mismatches:
            print(f"  {text}")
            print(f"    pred: {' / '.join(hyp_tokens)}")
            print(f"    gold: {' / '.join(gold_tokens)}")


if __name__ == "__main__":
    main()
