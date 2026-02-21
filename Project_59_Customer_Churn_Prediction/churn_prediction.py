# ===================================================
# Project 59: Customer Churn Prediction
# Goal: 実装力強化 (Import/Read/Features/Split/Fit/Predict)
# ===================================================

# [1] Import
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# [2] Read
# ※ファイルが同じフォルダにあることを確認してください
df = pd.read_csv("Churn Modeling.csv")
# [3] Features (データ前処理)
# 不要な列を削除
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
# カテゴリ変数の数値変換 (Label Encoding)
le = LabelEncoder()
df['Geography'] = le.fit_transform(df['Geography'])
df['Gender'] = le.fit_transform(df['Gender'])

# 特徴量(X)と目的変数(y)に分ける
X = df.drop('Exited', axis=1)
y = df['Exited']

# [4] Split (Stratified K-fold)
# 離脱比率を維持して5分割する「わたしのこだわり」設定
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# [5] Fit (学習) & [6] Predict/Submit (評価)
accuracies = []

for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # モデル構築
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 予測と精度確認   
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Fold {i+1}: Accuracy = {acc:.4f}")

print(f"Average Accuracy: {np.mean(accuracies):.4f}")