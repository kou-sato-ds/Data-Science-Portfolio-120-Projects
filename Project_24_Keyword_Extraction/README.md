# Project 24: Keyword Extraction from Research Papers

## 📌 概要
大量の学術論文（NIPS Papers）から、TF-IDFを用いて重要キーワードを自動抽出するプロジェクトです。

## 🛠️ 実施したこと（実装力のポイント）
- **テキスト前処理**: 正規表現を用いたクリーニング、ストップワード除去の実装。
- **ベクトル化**: `CountVectorizer` と `TfidfTransformer` を用いた単語の重み付け。
- **キーワード抽出**: 疎行列（Sparse Matrix）をソートし、上位スコアの単語を特定するロジックの構築。

## 🚀 学んだこと
- `docs`（コーパス）の概念と、NLPにおけるデータの持ち方。
- Scikit-learnのバージョン変更に伴う `stop_words` の型エラー対処。