import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. データの読み込み
df = pd.read_csv("Pokemon Data.csv")

# 2. 特徴量の作成（数値 + 文字データを数値化したもの）
# ここで Type_1 や Color を 0/1 のフラグに変換しています
X = pd.get_dummies(df[['Type_1', 'Color', 'Total', 'HP', 'Attack', 'Defense']], drop_first=True)
y = df['isLegendary']

# 3. 欠損値処理
X = X.fillna(0)

# 4. 分割・学習・予測（いつもの型！）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

# 5. 評価
print(classification_report(y_test, preds))