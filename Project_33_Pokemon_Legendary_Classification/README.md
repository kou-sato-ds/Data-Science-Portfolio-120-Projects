# Project 33: Find Legendary Pokemon

## 概要
ポケモンのステータス（種族値）やタイプ、色などのデータから、そのポケモンが「伝説のポケモン」かどうかを判定する機械学習モデルです。

## 実装内容
- **データ読み込み**: `Pokemon Data.csv` のインポート
- **前処理**: 
  - `pd.get_dummies` によるカテゴリ変数（Type, Color）の数値化（One-Hot Encoding）
  - 欠損値の補完（fillna）
- **モデル**: ランダムフォレスト（RandomForestClassifier）を使用
- **評価**: `classification_report` による適合率・再現率の確認

## 実行結果
- **Accuracy**: 約 97%
- **伝説ポケモンの再現率 (Recall)**: 約 62%
  - 不均衡データ（伝説ポケモンが極端に少ない）における評価指標の重要性を確認しました。

## 今後の展望
- Djangoを用いたWebアプリ化
- Herokuへのデプロイ