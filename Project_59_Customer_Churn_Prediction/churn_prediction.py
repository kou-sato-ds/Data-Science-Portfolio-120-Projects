# ==========================================
# Project 59: Customer Churn Prediction
# Goal: 実装力の証明 (K-fold, Stratify, Split)
# ==========================================

# [1] Import
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# [2] Read
df = pd.read_csv("Churn Modeling.csv")

# [3] Features
# 不要な列の削除（予測に寄与しないID系）
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# カテゴリ変数のエンコーディング (Geography, Gender)
le = LabelEncoder()
df['Geography'] = le.fit_transform(df['Geography'])
df['Gender'] = le.fit_transform(df['Gender'])

X = df.drop('Exited', axis=1)
y = df['Exited']

# [4] Split (Stratified K-fold)
# 離脱(1)と継続(0)の比率を維持したまま5分割
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# [5] Fit & [6] Submit (Evaluation)
print(f"{'Fold':<5} | {'Accuracy':<10}")
print("-" * 20)

accuracies = []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    # データの分割
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # モデルの構築と学習
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 予測
    y_pred = model.predict(X_test)
    
    # 評価
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"{i+1:<5} | {acc:.4f}")

print("-" * 20)
print(f"Mean Accuracy: {np.mean(accuracies):.4f}")