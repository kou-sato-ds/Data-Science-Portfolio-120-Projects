# ==========================================
# Step 1: Import
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier 

# ==========================================
# Step 2: Read
# ==========================================
df = pd.read_csv("Table_1.csv")
print("--- Data Head ---")
print(df.head())
print("\n--- Data Info ---")
print(df.info())

# ==========================================
# Step 3: Features (Preprocessing)
# ==========================================
# 1. 目的変数の数値化
df['Stay/Left'] = df['Stay/Left'].map({'Left': 1, 'Stay': 0})

# 2. 特徴量と目的変数に分離
X = df.drop('Stay/Left', axis=1)
y = df['Stay/Left']

# 3. 不要な列の削除（予測のノイズになるもの）
cols_to_drop = ['table id', 'name', 'phone number']
X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

# 4. カテゴリ変数をダミー変数化
X = pd.get_dummies(X, drop_first=True)

# 5. 文字列として残っている列を強制的に数値化、または削除
# (Table_1特有の "> 1" などの文字列を処理するため)
X = X.select_dtypes(include=[np.number]) 
X = X.fillna(0)

# ==========================================
# Step 4 & 5: Stratified K-Fold & Training
# ==========================================
print("\n--- 学習開始 (Stratified K-Fold) ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_val)
    score = accuracy_score(y_val, preds)
    print(f"Fold {fold+1} Accuracy: {score:.4f}")

# ==========================================
# Step 6: Evaluation
# ==========================================
print("\n--- 最終評価 (混同行列) ---")
print(confusion_matrix(y_val, preds))

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- 特徴量重要度（TOP5） ---")
print(importances.head(5))