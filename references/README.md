# References — building and evaluating the Japanese model

Tooling that builds gold segmentation data with fugashi (MeCab + UniDic) as
the teacher, trains models through the litsea binding's training API, and
scores them against the bundled ones. The shipped `models/japanese.model`
came out of this pipeline.

## Setup

`build_japanese_model.sh` creates the venv on first run, so this is only
needed to use the scripts on their own:

```bash
python3 -m venv .venv
.venv/bin/pip install litsea fugashi unidic-lite
# For IPAdic: .venv/bin/pip install ipadic
```

## Full pipeline (build_japanese_model.sh)

Fetches the corpora, trains, evaluates, and copies the result into
`models/japanese.model`:

```bash
./build_japanese_model.sh
```

Settings live in variables at the top of the script:

| Variable | Default | Meaning |
|---|---|---|
| `THRESHOLD` | 0.004 | Weak-classifier acceptance threshold; size/accuracy trade-off |
| `ITERATIONS` | 20000 | Max AdaBoost iterations |
| `MAGPIE_ROWS` / `WIKI_ARTICLES` | 500 / 300 | Conversations / articles to fetch |
| `SEED` | 42 | Train/test split seed |

`THRESHOLD` is the knob that matters: 0.004 gives a 7.3 KB model at 96.5% /
94.6% hold-out boundary F1. The measured sweep from 0.001 to 0.05 lives in
[TRAINING.md](../TRAINING.md); other values can be filled in with
`train_model.py --threshold <value>`.

One weakness survives retraining: short fixed phrases with no surrounding
context get over-split (`こんにちは` → `こん`/`に`/`ち`/`は`). Adding greetings
to the corpus did not fix it cleanly, so protect such phrases on the caller
side.

## Tools

| Script | Purpose |
|---|---|
| `fetch_magpie.py` | Fetch Japanese sentences from HF: llm-jp/magpie-sft-v1.0 (Apache-2.0) |
| `fetch_wikipedia.py` | Fetch Japanese sentences from HF: wikimedia/wikipedia (20231101.ja, CC BY-SA 3.0 / GFDL) |
| `make_gold.py` | Build litsea-format gold corpora with fugashi (UniDic/IPAdic); `--merge-min-chars` controls compound merging |
| `train_model.py` | Train via the litsea binding's Extractor+Trainer (`--threshold` / `--iterations` / `--init`) |
| `evaluate.py` | Compare models against gold: boundary F1 and word-span F1 |
| `main.py` | Compare Go port output vs Rust litsea (verifies the Go models) |

## Common usage

```bash
# Build gold data (raw UniDic segmentation)
.venv/bin/python make_gold.py data/train.txt data/train.gold.txt --merge-min-chars 0

# Train
.venv/bin/python train_model.py data/train.gold.txt data/my.model --threshold 0.004

# Evaluate (multi-model comparison)
.venv/bin/python evaluate.py data/test.gold.txt ../models/*.model data/my.model --show-mismatch 5
```

## Notes

- `data/` and `build/` are generated artifacts (gitignored). Feature files in
  `build/` grow to hundreds of MB for large corpora.
- Always evaluate on a held-out test set; in-sample evaluation just reads
  100% and means nothing.
