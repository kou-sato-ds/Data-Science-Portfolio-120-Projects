# ==========================================
# Step 1: Import
# ==========================================
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download('stopwords')
nltk.download('wordnet')

# ==========================================
# Step 2: Read
# ==========================================
# ファイルパスは環境に合わせて適宜変更してください
df = pd.read_csv('papers.csv')

# ==========================================
# Step 3: Features (Preprocessing)
# ==========================================
stop_words = list(set(stopwords.words("english")))

def pre_process(text):
    text = text.lower()
    # HTMLタグの除去
    text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", text)
    # 特殊文字と数字の除去
    text = re.sub("(\\d|\\W)+"," ", text)
    return text

# 前処理を適用して docs を作成
docs = df['paper_text'].apply(pre_process)

# ==========================================
# Step 4: Vectorization (TF-IDF)
# ==========================================
# CountVectorizerで単語をカウント
cv = CountVectorizer(max_df=0.8, stop_words=stop_words, max_features=10000)
word_count_vector = cv.fit_transform(docs)

# TfidfTransformerで重み付け
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)

# ==========================================
# Step 5: Extract Logic (Helper Functions)
# ==========================================
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results

def get_keywords(idx, docs):
    # 該当する論文のベクトルを取得
    tf_idf_vector = tfidf_transformer.transform(cv.transform([docs[idx]]))
    # 疎行列をソート可能な形式に変換
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    # 特徴量名を取得
    feature_names = cv.get_feature_names_out()
    # 上位10個を抽出
    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)
    return keywords

def print_results(idx, keywords, df):
    print("\n===== Title =====")
    print(df['title'][idx])
    print("\n===== Abstract =====")
    print(df['abstract'][idx])
    print("\n===== Keywords =====")
    for k in keywords:
        print(f"{k}: {keywords[k]}")

# ==========================================
# Step 6: Evaluation
# ==========================================
print("行列の形状:", word_count_vector.shape)
print("TF-IDFの学習が完了しました。\n")

idx = 120
keywords = get_keywords(idx, docs)
print_results(idx, keywords, df)