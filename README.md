# cjk

Japanese and Chinese text runs on without spaces between words, and Korean's
spaces fall between phrases rather than words. `cjk` finds the breaks: hand it
a sentence, get back the words. It is a pure Go port of
[Litsea](https://github.com/mosuka/litsea), a compact Rust library that does
the same job.

Finding those breaks normally takes a dictionary, and in Go that means
[kagome](https://github.com/ikawaha/kagome) with UniDic, linked into your
binary. `cjk` works from small models that ship inside the package, so nothing
beyond `go get` is involved:

```text
cjk              ▉ 1.7 MB
kagome + UniDic  ████████████████████████▎ 48.4 MB
```

The dictionary earns its size if you need readings and lemmas, which `cjk`
cannot give you. For splitting alone it comes close: 96.5% boundary F1 against
UniDic's own segmentation on chat text.
([TRAINING.md](TRAINING.md)).

## Installation

```
go get github.com/masacento/cjk
```

## Usage

### Segment text

`Default` returns a shared, concurrency-safe `Segmenter` backed by the
embedded model for the language.

```go
package main

import (
	"fmt"

	"github.com/masacento/cjk"
)

func main() {
	ja, err := cjk.Default(cjk.Japanese)
	if err != nil {
		panic(err)
	}

	fmt.Println(ja.Segment("機械学習ライブラリでHugging Faceを使う"))
	// [機械 学習 ライブラリ で Hugging   Face を 使う]

	zh, _ := cjk.Default(cjk.Chinese)
	fmt.Println(zh.Segment("我是中国人"))
	// [我 是 中国 人]

	ko, _ := cjk.Default(cjk.Korean)
	fmt.Println(ko.Segment("한국어 단어 분할 테스트입니다"))
	// [한국어   단어   분할   테스트 입니다]
}
```

Korean text keeps its inter-word spaces: each space is returned as its own
token, so joining the tokens reproduces the input exactly.

### Get byte offsets

`SplitWords` returns each word with its byte range in the original string,
which is what you want for reporting spans (e.g. from an NER pipeline).

```go
for _, t := range ja.SplitWords("機械学習ライブラリでHugging Faceを使う") {
	fmt.Printf("%d-%d %s\n", t.Start, t.End, t.Text)
}
// 0-6 機械
// 6-12 学習
// 12-27 ライブラリ
// 27-30 で
// 30-37 Hugging
// 37-38 (space)
// 38-42 Face
// 42-45 を
// 45-51 使う
```

### Language detection

```go
cjk.HasCJK("Steve Jobs founded Apple") // false
cjk.HasCJK("日本語")                    // true
cjk.HasHangul("한국어")                  // true

lang, ok := cjk.DetectCJK("これはテストです。") // cjk.Japanese, true
```

### Custom models

Models trained with Litsea's `extract` / `train` commands can be loaded from a
file or bytes:

```go
ada, err := cjk.LoadAdaBoost("my_model.model")
seg := cjk.NewSegmenter(cjk.Japanese, ada)

// or: ada, err := cjk.LoadAdaBoostFromBytes(modelBytes)
```

## API

| Identifier | Description |
|---|---|
| `Default(l Language) (*Segmenter, error)` | Shared segmenter backed by the embedded model |
| `NewSegmenter(l Language, ada *AdaBoost) *Segmenter` | Segmenter from a custom AdaBoost model |
| `NewPosSegmenter(l Language, pos *AveragedPerceptron) *Segmenter` | Segmenter with POS tagging; POS models are loaded, not embedded |
| `(*Segmenter).Segment(sentence string) []string` | Segment into words |
| `(*Segmenter).SplitWords(sentence string) []Token` | Segment into words with byte offsets |
| `(*Segmenter).SegmentWithPos(sentence string) ([]Word, error)` | Segment and tag with UPOS |
| `LoadAdaBoost(path string) (*AdaBoost, error)` | Load an AdaBoost model file |
| `LoadAdaBoostFromBytes(data []byte) (*AdaBoost, error)` | Load an AdaBoost model from bytes |
| `LoadAveragedPerceptron(path string) (*AveragedPerceptron, error)` | Load a POS model file |
| `DetectCJK(text string) (Language, bool)` | Detect Japanese / Chinese / Korean |
| `HasCJK(text string) bool`, `HasHangul(text string) bool` | Script detection |
| `Language` (`Japanese`, `Chinese`, `Korean`), `ParseLanguage` | Language values and parsing |
| `Upos` (`NOUN`, `VERB`, `ADP`, …) | The 17 Universal POS tags |

## How it works

Litsea (and this port) treats word segmentation as binary classification per
character boundary: AdaBoost over character n-gram features decides whether a
boundary exists before each character. POS tagging adds a second stage — a
word-level Averaged Perceptron tagger restricted by a candidate-tag lexicon,
using the UPOS tagset from Universal Dependencies.

The weights are compiled once at load time into tables keyed by packed integer
feature keys, so scoring a character builds no feature strings and allocates
nothing per position.

## The Japanese model

The bundled `japanese.model` is not Litsea's original. It is retrained on
Japanese chat and Wikipedia text so that English words inside Japanese
sentences stay whole — the upstream model cuts `GitHub` into `GitH`/`u`/`b`.
Corpus, training settings, evaluation numbers, licenses, and the
threshold/size/accuracy trade-off are in [TRAINING.md](TRAINING.md).

## License

This project is distributed under the MIT License — the same license as
Litsea. See [LICENSE](LICENSE).

Training-data licenses for the retrained `japanese.model`:
[llm-jp/magpie-sft-v1.0](https://huggingface.co/datasets/llm-jp/magpie-sft-v1.0)
is Apache-2.0 (LLM-jp); the Japanese Wikipedia text
([wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia))
is CC BY-SA 3.0 / GFDL. The retrained model weights themselves are
distributed under this repository's MIT license.
