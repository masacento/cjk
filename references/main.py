"""Reference segmentation outputs from Rust litsea, for comparison with the Go port.

Runs the same sentence cases as the Go package's golden tests (golden_test.go)
and the mixed-script investigation through the litsea Python binding, printing
each result in Go %q format so the outputs can be diffed directly.

Usage:
    pip install litsea
    python main.py                    # run all built-in cases
    python main.py "任意の文"          # run one sentence (japanese model)
    python main.py "text" english     # run one sentence with a chosen model

Models are loaded from ../models relative to this script. english.model is not
bundled here; English cases run only if you place litsea's english.model in
that directory.
"""

import json
import sys
from pathlib import Path

try:
    from litsea import Language, Segmenter
except ImportError:
    sys.exit("litsea is required: pip install litsea")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

LANGUAGES = {
    "japanese": Language.JAPANESE,
    "chinese": Language.CHINESE,
    "korean": Language.KOREAN,
    "english": Language.ENGLISH,
}

# Same cases as TestGoldenSegment* in golden_test.go.
GOLDEN_CASES = {
    "japanese": [
        "これはテストです。",
        "私の猫は可愛い。",
        "東京都に住んでいます。",
        "字",
        "こんにちは",
        "価格は1000円です。",
        "RustでNLPを実装する。",
    ],
    "chinese": [
        "这是一个测试。",
        "我喜欢吃中国菜。",
        "他在北京工作。",
        "好",
        "2024年的春天。",
    ],
    "korean": [
        "이것은 테스트입니다.",
        "나는 고양이를 좋아한다.",
        "한국어 형태소 분석기.",
        "글",
        "2024년 봄.",
    ],
}

# Same mixed Japanese/English cases as the english.model investigation.
MIXED_CASES = [
    "iPhoneはAppleの製品です。",
    "機械学習ライブラリのHugging Faceを使う。",
    "これはOpenAIのGPT-4です。",
    "Stack Overflowで質問しました。",
    "GitHubでコードを管理する。",
    "iOS 17では新機能が追加された。",
    "I love 日本の文化。",
    "こんにちはworld、これはテストです。",
    "The quick brown fox jumps over the lazy dog という英文。",
]

# Run only if the model file exists (not bundled in this repository).
ENGLISH_CASES = [
    "This is a test.",
    "I don't know.",
    "Google's search engine.",
]


def go_quote(words):
    """Format a word list the way Go's fmt %q prints a []string."""
    return "[" + " ".join(json.dumps(w, ensure_ascii=False) for w in words) + "]"


def run(label, language_name, sentences):
    model = MODELS_DIR / f"{language_name}.model"
    if not model.exists():
        print(f"# {label}: skipped ({model} not found)")
        return
    seg = Segmenter.open(LANGUAGES[language_name], str(model))
    print(f"# {label} ({language_name}.model)")
    for s in sentences:
        print(f"{language_name} {json.dumps(s, ensure_ascii=False)} => {go_quote(seg.segment(s))}")


def main():
    if len(sys.argv) > 1:
        text = sys.argv[1]
        language_name = sys.argv[2] if len(sys.argv) > 2 else "japanese"
        if language_name not in LANGUAGES:
            sys.exit(f"unknown language: {language_name} (expected one of {', '.join(LANGUAGES)})")
        run("ad-hoc", language_name, [text])
        return

    for language_name, sentences in GOLDEN_CASES.items():
        run("golden", language_name, sentences)
    run("mixed JA/EN", "japanese", MIXED_CASES)
    run("english reference", "english", ENGLISH_CASES)


if __name__ == "__main__":
    main()
