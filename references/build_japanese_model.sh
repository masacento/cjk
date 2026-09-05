#!/bin/bash
# Rebuild the Go-embedded japanese.model from scratch.
#
# Pipeline:
#   1. fetch llm-jp/magpie-sft-v1.0 + ja Wikipedia
#      (HF: wikimedia/wikipedia, 20231101.ja) sentences
#   2. split train / test
#   3. build gold corpora with fugashi (MeCab + UniDic), no compound merging
#   4. train an AdaBoost model on the mixed train corpus (threshold=0.004)
#   5. evaluate against both hold-out test sets
#
# Usage:
#   ./build_japanese_model.sh
#
# Requires: python3, network access. Creates .venv on first run.

set -euo pipefail
cd "$(dirname "$0")"

THRESHOLD=0.004
ITERATIONS=20000
SEED=42

WIKI_ARTICLES=300
WIKI_TRAIN_SENTS=15000
WIKI_TEST_SENTS=500
MAGPIE_ROWS=500

if [ ! -x .venv/bin/python ]; then
    echo "== creating venv =="
    python3 -m venv .venv
    .venv/bin/pip install --quiet litsea fugashi unidic-lite
fi
PY=.venv/bin/python

mkdir -p data build

echo "== 1. fetch corpora =="
$PY fetch_magpie.py "$MAGPIE_ROWS" data/magpie.txt
$PY fetch_wikipedia.py "$WIKI_ARTICLES" data/wiki_ja.txt

echo "== 2. split train / test (seed=$SEED) =="
$PY - "$SEED" "$WIKI_TEST_SENTS" "$WIKI_TRAIN_SENTS" <<'EOF'
import random, sys
from pathlib import Path

seed = int(sys.argv[1])
wiki_test_n = int(sys.argv[2])
wiki_train_n = int(sys.argv[3])

chat = Path("data/magpie.txt").read_text(encoding="utf-8").splitlines()
wiki = Path("data/wiki_ja.txt").read_text(encoding="utf-8").splitlines()

rng = random.Random(seed)
rng.shuffle(chat)
chat_test_n = len(chat) // 10  # 10% held out
chat_test, chat_train = chat[:chat_test_n], chat[chat_test_n:]
rng.shuffle(wiki)
wiki_test, wiki_train = wiki[:wiki_test_n], wiki[wiki_test_n:wiki_test_n + wiki_train_n]

Path("data/magpie_train.txt").write_text("\n".join(chat_train) + "\n", encoding="utf-8")
Path("data/magpie_test.txt").write_text("\n".join(chat_test) + "\n", encoding="utf-8")
Path("data/wiki_train.txt").write_text("\n".join(wiki_train) + "\n", encoding="utf-8")
Path("data/wiki_test.txt").write_text("\n".join(wiki_test) + "\n", encoding="utf-8")
print(f"chat train={len(chat_train)} test={len(chat_test)} / wiki train={len(wiki_train)} test={len(wiki_test)}")
EOF

echo "== 3. build gold corpora (fugashi + UniDic, no merge) =="
$PY make_gold.py data/magpie_train.txt data/magpie_train.gold.txt --merge-min-chars 0
$PY make_gold.py data/magpie_test.txt data/magpie_test.gold.txt --merge-min-chars 0
$PY make_gold.py data/wiki_train.txt data/wiki_train.gold.txt --merge-min-chars 0
$PY make_gold.py data/wiki_test.txt data/wiki_test.gold.txt --merge-min-chars 0
cat data/magpie_train.gold.txt data/wiki_train.gold.txt > data/mixed_train.gold.txt

echo "== 4. train (threshold=$THRESHOLD, iterations=$ITERATIONS) =="
$PY train_model.py data/mixed_train.gold.txt data/japanese.model \
    --threshold "$THRESHOLD" --iterations "$ITERATIONS"

cp data/japanese.model ../models/japanese.model
echo "copied to models/japanese.model"

echo "== 5. evaluate on hold-out test sets =="
echo "--- magpie test ---"
$PY evaluate.py data/magpie_test.gold.txt data/japanese.model
echo "--- ja-wikipedia test ---"
$PY evaluate.py data/wiki_test.gold.txt data/japanese.model

ls -l data/japanese.model
echo "done: references/data/japanese.model"
