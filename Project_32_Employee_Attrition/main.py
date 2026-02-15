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
# 明日、Table_1.csvをフォルダに配置してから以下のコメントを外して写経開始
# df = pd.read_csv("Table_1.csv")
# print(df.head())

# ==========================================
# Step 3: Features (Preprocessing)
# ==========================================
# 実装力を鍛えるポイント：文字データを数値に変換する工程をここに書く
# 例: df['Stay/Left'] = df['Stay/Left'].map({'Stay': 0, 'Left': 1})

# ==========================================
# Step 4: K-fold Stratified Split
# ==========================================
# 佐藤さんのこだわり：層化分割（Stratified）の実装
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==========================================
# Step 5: Training & Prediction
# ==========================================
# モデルの学習ロジックを写経

# ==========================================
# Step 6: Submit / Evaluation
# ==========================================
# confusion_matrix などで結果を可視化