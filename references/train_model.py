"""Train a litsea segmentation model from a gold corpus, via the litsea binding.

The corpus must be in litsea format (one sentence per line, words separated
by spaces) — the output of make_gold.py.

Usage:
    python train_model.py gold.txt my.model
    python train_model.py gold.txt my.model --iterations 50000
    python train_model.py new.txt my.model --init existing.model   # incremental
"""

import argparse
from pathlib import Path

from litsea import Extractor, Trainer


def fmt_pct(value: float) -> str:
    # litsea's metrics report percentages already scaled to 0-100.
    return f"{value:.2f}%" if value > 1.0 else f"{value:.2%}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", type=Path, help="space-separated corpus from make_gold.py")
    ap.add_argument("model", type=Path, help="output model path")
    ap.add_argument("--language", default="japanese", help="language for feature extraction (default: japanese)")
    ap.add_argument("--threshold", type=float, default=0.0001, help="weak-classifier accuracy threshold (default: 0.0001)")
    ap.add_argument("--iterations", type=int, default=20000, help="max AdaBoost iterations (default: 20000)")
    ap.add_argument("--build-dir", type=Path, default=Path(__file__).resolve().parent / "build",
                    help="directory for intermediate feature files (default: references/build)")
    ap.add_argument("--init", type=Path, default=None,
                    help="resume training from this existing model instead of starting fresh")
    args = ap.parse_args()

    args.build_dir.mkdir(parents=True, exist_ok=True)
    features = args.build_dir / (args.corpus.stem + ".features")

    print(f"extracting features: {args.corpus} -> {features}")
    Extractor(args.language).extract(args.corpus, features)

    trainer = Trainer(args.threshold, args.iterations, features)
    if args.init:
        print(f"loading initial model: {args.init}")
        trainer.load_model(str(args.init))

    print(f"training: threshold={args.threshold} iterations={args.iterations}")
    metrics = trainer.train(args.model)

    print(f"accuracy  : {fmt_pct(metrics.accuracy)} ({metrics.true_positives + metrics.true_negatives}"
          f" / {metrics.num_instances})")
    print(f"precision : {fmt_pct(metrics.precision)}")
    print(f"recall    : {fmt_pct(metrics.recall)}")
    print(f"model     : {args.model}")


if __name__ == "__main__":
    main()
