# Training

The bundled `japanese.model` is not the original Litsea model. It is retrained
on a mixed Japanese corpus so that English words embedded in Japanese text
stay intact (the upstream UD-trained model shreds unknown English words like
`GitHub` → `GitH`/`u`/`b`).

## Corpus

~20,000 sentences, 15,000 of them Wikipedia:

- [llm-jp/magpie-sft-v1.0](https://huggingface.co/datasets/llm-jp/magpie-sft-v1.0)
  (Apache-2.0) — Japanese instruction-tuning chats by LLM-jp
- Japanese Wikipedia
  ([wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia),
  config `20231101.ja`, CC BY-SA 3.0 / GFDL)

## Method

- **Teacher**: [fugashi](https://github.com/polm/fugashi) (MeCab) with UniDic,
  raw short-unit segmentation (no compound merging)
- **Learner**: Litsea's AdaBoost trainer, `threshold=0.004`, max 20,000
  iterations

## Accuracy

Held-out boundary F1 on the fugashi/UniDic gold, against the upstream
`japanese.model`:

| Test set | upstream | shipped (t=0.004) | t=0.001 |
|---|---|---|---|
| magpie-sft test (590 sentences) | 90.32% | **96.49%** | 97.60% |
| ja-Wikipedia test (500 sentences) | 90.93% | **94.56%** | 95.25% |

Both test sets are hold-outs from the two corpora the model was trained on
(split seed 42), so these are in-domain numbers.

## Known weaknesses

Short fixed phrases with no surrounding context:
`こんにちは` still comes back over-split as `こん`/`に`/`ち`/`は`. This is a
corpus-vocabulary gap that supplementing greetings during retraining did not
fix cleanly; if it matters for your use case, protect fixed phrases in the
caller before segmentation.

## Threshold / size / accuracy trade-off

Measured on hold-out boundary F1 (merge=0 gold). The t=0.004 row is the
shipped model.

| threshold | size | train time | magpie test | wiki test |
|---|---|---|---|---|
| 0.001 | 20.7KB | 15 min | 97.60% | 95.25% |
| **0.004 (shipped)** | **7.3KB** | **3 min** | **96.49%** | **94.56%** |
| 0.005 | 5.8KB | 1m22s | 95.93%* | 94.36%* |
| 0.01 | 3.4KB | 34s | 94.21%* | 92.32%* |
| 0.05 | 612B | 12s | 87.93%* | 88.28%* |
| (upstream model, 1.3KB) | — | — | 90.32% | 90.93% |

`*` Earlier sweep values, measured before the corpus switched to
magpie-sft-v1.0 (see git history of this file). Kept as order-of-magnitude
references; the 0.001/0.004 rows are the current corpus measurements.

## Licenses

Training-data licenses: magpie-sft-v1.0 is Apache-2.0 (LLM-jp); the
Wikipedia text is CC BY-SA 3.0 / GFDL. The retrained model weights are
distributed under this repository's MIT license.

## Reproducing

```bash
cd references
./build_japanese_model.sh
```

It fetches the corpora, builds gold data, trains, evaluates on both hold-out
test sets, and copies the result to `models/japanese.model`. The individual
scripts are documented in [references/README.md](references/README.md).
