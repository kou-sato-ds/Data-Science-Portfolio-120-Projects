# Project 59: Customer Churn Prediction

## 1. 概要
銀行顧客の属性データ（Churn Modeling.csv）から、その顧客が離脱するかどうかを予測するモデルを構築します。

## 2. 実装の型 (黄金のプロセス)
本プロジェクトでは、特に「実装力」を重視し、以下のフローを徹底します。
1. **Import**: Pandas, Scikit-learn, XGBoost等の導入
2. **Read**: `Churn Modeling.csv` の読み込み
3. **Features**: LabelEncoding, 不要カラムの削除、特徴量エンジニアリング
4. **Split (K-fold / Stratify)**: 層化抽出を用いた交差検証の準備
5. **Fit**: 分類モデルの学習
6. **Predict/Submit**: 予測結果の出力と精度の確認