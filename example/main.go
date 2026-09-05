package main

import (
	"fmt"

	"github.com/masacento/cjk"
)

func must(l cjk.Language) *cjk.Segmenter {
	seg, err := cjk.Default(l)
	if err != nil {
		panic(err)
	}
	return seg
}

func main() {
	ja := must(cjk.Japanese)
	fmt.Println(ja.Segment("これはテストです。"))
	fmt.Println(ja.Segment("機械学習ライブラリでHugging Faceを使う"))

	fmt.Println("--- SplitWords ---")
	for _, t := range ja.SplitWords("機械学習ライブラリでHugging Faceを使う") {
		fmt.Printf("%d-%d %q\n", t.Start, t.End, t.Text)
	}

	zh := must(cjk.Chinese)
	fmt.Println(zh.Segment("我是中国人"))

	ko := must(cjk.Korean)
	fmt.Println(ko.Segment("한국어 단어 분할 테스트입니다"))

	fmt.Println("--- Detect ---")
	for _, s := range []string{"Steve Jobs founded Apple", "日本語", "한국어", "这是中文"} {
		fmt.Printf("%q HasCJK=%v lang=%v ok=%v\n", s, cjk.HasCJK(s), func() cjk.Language {
			l, _ := cjk.DetectCJK(s)
			return l
		}(), func() bool {
			_, ok := cjk.DetectCJK(s)
			return ok
		}())
	}
}
