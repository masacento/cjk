package cjk

import "testing"

// Golden tests: snapshot the segmentation output of every pre-trained model
// in models/ so that refactoring can be verified to preserve behavior.
//
// These snapshots capture the current behavior of the bundled models. If a
// behavior change is intentional (e.g. retraining a model), update the
// affected expectations in the same PR and call the change out explicitly in
// the PR description.
//
// Note: japanese.model was retrained on a mixed corpus (lambda-chat +
// ja-Wikipedia sentences, gold from fugashi/UniDic). See
// references/build_japanese_model.sh for the reproducible build.

func assertGolden(t *testing.T, seg *Segmenter, cases []struct {
	input string
	want  []string
}) {
	t.Helper()
	for _, c := range cases {
		got := seg.Segment(c.input)
		if len(got) != len(c.want) {
			t.Errorf("Segment(%q) = %q, want %q", c.input, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("Segment(%q) = %q, want %q", c.input, got, c.want)
				break
			}
		}
	}
}

func TestGoldenSegmentJapanese(t *testing.T) {
	ada, err := LoadAdaBoost("models/japanese.model")
	if err != nil {
		t.Fatalf("failed to load japanese.model: %v", err)
	}
	seg := NewSegmenter(Japanese, ada)

	assertGolden(t, seg, []struct {
		input string
		want  []string
	}{
		{"これはテストです。", []string{"これ", "は", "テスト", "です", "。"}},
		{"私の猫は可愛い。", []string{"私", "の", "猫", "は", "可愛", "い", "。"}},
		{"東京都に住んでいます。", []string{"東京", "都", "に", "住ん", "でい", "ます", "。"}},
		// Edge case: single character (whole sentence is one word).
		{"字", []string{"字"}},
		{"こんにちは", []string{"こん", "に", "ち", "は"}},
		// Digits and mixed scripts.
		{"価格は1000円です。", []string{"価格", "は", "1000", "円", "です", "。"}},
		{"RustでNLPを実装する。", []string{"Rust", "で", "NLP", "を", "実装", "する", "。"}},
		// Long English words inside Japanese text must stay intact — the
		// retrained model learned this from English-heavy mixed corpora.
		{"機械学習フレームワークのTensorFlowを使って画像認識を試した。", []string{"機械", "学習", "フレームワーク", "の", "TensorFlow", "を", "使っ", "て", "画像", "認識", "を", "試し", "た", "。"}},
		{"GitHubでPull Requestを出したらCIが失敗しました。", []string{"GitHub", "で", "Pull", " ", "Request", "を", "出し", "たら", "CI", "が", "失敗", "し", "まし", "た", "。"}},
		{"Kubernetesクラスタ上でMicroservicesを運用している。", []string{"Kubernetes", "クラスタ", "上", "で", "Microservices", "を", "運用", "し", "て", "いる", "。"}},
	})

	if got := seg.Segment(""); got != nil {
		t.Errorf("Segment(\"\") = %q, want nil", got)
	}
}

func TestGoldenSegmentChinese(t *testing.T) {
	ada, err := LoadAdaBoost("models/chinese.model")
	if err != nil {
		t.Fatalf("failed to load chinese.model: %v", err)
	}
	seg := NewSegmenter(Chinese, ada)

	assertGolden(t, seg, []struct {
		input string
		want  []string
	}{
		{"这是一个测试。", []string{"这", "是", "一个", "测试", "。"}},
		// Not a gold segmentation: "我喜"/"欢吃" shows the bundled model's
		// current behavior, which the snapshot pins as-is.
		{"我喜欢吃中国菜。", []string{"我喜", "欢吃", "中国", "菜", "。"}},
		{"他在北京工作。", []string{"他", "在", "北京", "工作", "。"}},
		{"好", []string{"好"}},
		{"2024年的春天。", []string{"2024", "年", "的", "春天", "。"}},
	})

	if got := seg.Segment(""); got != nil {
		t.Errorf("Segment(\"\") = %q, want nil", got)
	}
}

func TestGoldenSegmentKorean(t *testing.T) {
	ada, err := LoadAdaBoost("models/korean.model")
	if err != nil {
		t.Fatalf("failed to load korean.model: %v", err)
	}
	seg := NewSegmenter(Korean, ada)

	assertGolden(t, seg, []struct {
		input string
		want  []string
	}{
		// The model is trained on a space-preserving corpus, so inter-eojeol
		// spaces surface as their own tokens and joining the tokens
		// reproduces the input exactly.
		{"이것은 테스트입니다.", []string{"이것은", " ", "테스트", "입니다", "."}},
		{"나는 고양이를 좋아한다.", []string{"나는", " ", "고양이를", " ", "좋아한다", "."}},
		{"한국어 형태소 분석기.", []string{"한국어", " ", "형태소", " ", "분석기", "."}},
		{"글", []string{"글"}},
		{"2024년 봄.", []string{"2024년", " ", "봄."}},
	})

	if got := seg.Segment(""); got != nil {
		t.Errorf("Segment(\"\") = %q, want nil", got)
	}
}

// TestGoldenEmbeddedMatchesFile guards the embedded models: the models baked
// into the binary with go:embed must load to segmentations identical to the
// files in models/.
func TestGoldenEmbeddedMatchesFile(t *testing.T) {
	for _, tc := range []struct {
		language Language
		model    string
		sentence string
	}{
		{Japanese, "japanese.model", "これはテストです。"},
		{Chinese, "chinese.model", "这是一个测试。"},
		{Korean, "korean.model", "이것은 테스트입니다."},
	} {
		ada, err := LoadAdaBoost("models/" + tc.model)
		if err != nil {
			t.Fatalf("failed to load %s: %v", tc.model, err)
		}
		fromFile := NewSegmenter(tc.language, ada)
		embedded, err := Default(tc.language)
		if err != nil {
			t.Fatalf("Default(%v): %v", tc.language, err)
		}
		got, want := embedded.Segment(tc.sentence), fromFile.Segment(tc.sentence)
		if len(got) != len(want) {
			t.Errorf("embedded model diverges from %s on %q: %q vs %q", tc.model, tc.sentence, got, want)
			continue
		}
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("embedded model diverges from %s on %q: %q vs %q", tc.model, tc.sentence, got, want)
				break
			}
		}
	}
}
