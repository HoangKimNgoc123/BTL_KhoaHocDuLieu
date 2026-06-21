import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

reviews = pd.read_csv("chatgpt_reviews.csv")

reviews['at'] = pd.to_datetime(reviews['at'])

reviews = reviews.dropna(subset=['content'])

reviews['review_length'] = reviews['content'].str.len()

rating_counts = reviews['score'].value_counts().sort_index()

fig, ax = plt.subplots(1,2, figsize=(12,5))

# Bar Chart
sns.barplot(
    x=rating_counts.index,
    y=rating_counts.values,
    ax=ax[0]
)

ax[0].set_title("Rating Distribution")
ax[0].set_xlabel("Score")
ax[0].set_ylabel("Reviews")

# Pie Chart
ax[1].pie(
    rating_counts.values,
    labels=rating_counts.index,
    autopct='%1.1f%%'
)

ax[1].set_title("Rating Share")

plt.show()

avg_score = reviews['score'].mean()

print(f"Average Rating: {avg_score:.2f}")

reviews['YearMonth'] = reviews['at'].dt.strftime('%Y-%m')

monthly_reviews = (
    reviews.groupby('YearMonth')
    .size()
    .reset_index(name='ReviewCount')
)

monthly_score = (
    reviews.groupby('YearMonth')['score']
    .mean()
    .reset_index(name='AvgScore')
)

fig, ax = plt.subplots(2,1, figsize=(14,10))

sns.lineplot(
    data=monthly_reviews,
    x='YearMonth',
    y='ReviewCount',
    marker='o',
    ax=ax[0]
)

ax[0].set_title("Monthly Review Volume")
ax[0].tick_params(axis='x', rotation=45)

sns.lineplot(
    data=monthly_score,
    x='YearMonth',
    y='AvgScore',
    marker='o',
    ax=ax[1]
)

ax[1].set_title("Monthly Average Rating")
ax[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

version_stats = reviews.groupby(
    'appVersion'
).agg(
    review_count=('score','count'),
    avg_score=('score','mean')
).reset_index()

version_stats = version_stats[
    version_stats['review_count'] >= 100
]

top_versions = version_stats.sort_values(
    'review_count',
    ascending=False
).head(10)

plt.figure(figsize=(12,6))

sns.barplot(
    data=top_versions,
    x='appVersion',
    y='review_count'
)

plt.xticks(rotation=45)
plt.title("Top 10 Most Reviewed Versions")

plt.show()

best_versions = version_stats.nlargest(
    10,
    'avg_score'
)

worst_versions = version_stats.nsmallest(
    10,
    'avg_score'
)

fig, ax = plt.subplots(
    1,
    2,
    figsize=(14,5)
)

sns.barplot(
    data=best_versions,
    x='avg_score',
    y='appVersion',
    ax=ax[0]
)

ax[0].set_title("Best Versions")

sns.barplot(
    data=worst_versions,
    x='avg_score',
    y='appVersion',
    ax=ax[1]
)

ax[1].set_title("Worst Versions")

plt.show()

reviews['review_length'] = (
    reviews['content']
    .fillna('')
    .str.len()
)

corr = reviews[
    ['review_length',
     'thumbsUpCount']
].corr().iloc[0,1]

print(
    f"Correlation = {corr:.3f}"
)

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=reviews,
    x='review_length',
    y='thumbsUpCount',
    alpha=0.4
)

plt.title(
    f"Review Length vs ThumbsUp (r={corr:.3f})"
)

plt.show()

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Làm sạch dữ liệu
def clean_text(text):

    text = str(text).lower()

    remove_words = [
        'chatgpt',
        'gpt',
        'app',
        'ai'
    ]

    for word in remove_words:
        text = text.replace(word, '')

    text = re.sub(r'[^a-z\s]', ' ', text)

    return text

reviews['clean_content'] = (
    reviews['content']
    .fillna('')
    .apply(clean_text)
)

positive = reviews[
    reviews['score'] >= 4
]['clean_content']

negative = reviews[
    reviews['score'] <= 2
]['clean_content']

custom_stopwords = list(ENGLISH_STOP_WORDS) + [
    'good',
    'great',
    'nice',
    'just',
    'really',
    'like',
    'use',
    'using',
    'get',
    'one',
    'also'
]

def get_top_tfidf_words(texts, top_n=20):

    vectorizer = TfidfVectorizer(
        stop_words=custom_stopwords,
        max_features=5000
    )

    X = vectorizer.fit_transform(texts)

    scores = X.sum(axis=0).A1

    words = pd.DataFrame({
        'word': vectorizer.get_feature_names_out(),
        'score': scores
    })

    return words.sort_values(
        'score',
        ascending=False
    ).head(top_n)

pos_words = get_top_tfidf_words(positive)
neg_words = get_top_tfidf_words(negative)

fig, ax = plt.subplots(
    1,
    2,
    figsize=(14,6)
)

sns.barplot(
    data=pos_words,
    x='score',
    y='word',
    ax=ax[0]
)

ax[0].set_title(
    'Positive Keywords (TF-IDF)'
)

sns.barplot(
    data=neg_words,
    x='score',
    y='word',
    ax=ax[1]
)

ax[1].set_title(
    'Negative Keywords (TF-IDF)'
)

plt.tight_layout()
plt.show()

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# 1. ĐỌC VÀ LỌC DỮ LIỆU
df = pd.read_csv('chatgpt_reviews.csv')
df = df.dropna(subset=['content'])

# Chỉ lấy các đánh giá có từ 10 từ trở lên (để đảm bảo có "chủ đề" thực sự)
df['word_count'] = df['content'].apply(lambda x: len(str(x).split()))
df_long = df[df['word_count'] >= 10].copy()

# 2. LOẠI BỎ TỪ NHIỄU (CUSTOM STOP WORDS)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
custom_stop_words = list(ENGLISH_STOP_WORDS) + [
    'app', 'ai', 'chatgpt', 'gpt', 'chat', 'good', 'nice', 'great', 'awesome',
    'excellent', 'best', 'love', 'really', 'just', 'like', 'very', 'amazing',
    'useful', 'helpful', 'application', 'use', 'using', 'used', 'make', 'way',
    'help', 'helps', 'helped', 'lot', 'thank', 'thanks'
]

# max_df=0.85 (bỏ từ xuất hiện >85% văn bản), min_df=10 (phải xuất hiện ít nhất 10 lần)
vectorizer = TfidfVectorizer(stop_words=custom_stop_words, max_df=0.85, min_df=10, max_features=2000, ngram_range=(1,2))
X = vectorizer.fit_transform(df_long['content'].astype(str))

# 3. TÌM CHỦ ĐỀ VỚI NMF (Chạy thử 4 Cụm Chủ Đề)
num_topics = 4
nmf = NMF(n_components=num_topics, random_state=42)
W = nmf.fit_transform(X)
H = nmf.components_
terms = vectorizer.get_feature_names_out()

# 4. IN KẾT QUẢ
print("CÁC NHÓM CHỦ ĐỀ CHÍNH (NMF):\n" + "-"*40)
df_long['topic'] = W.argmax(axis=1)

for i in range(num_topics):
    top_words_idx = H[i].argsort()[::-1][:10]
    top_words = [terms[idx] for idx in top_words_idx]
    print(f"CỤM {i}: {', '.join(top_words)}")

    # In 2 đánh giá tiêu biểu nhất cho cụm
    sample_idx = W[:, i].argsort()[::-1][:2]
    print(" ➔ Ví dụ:")
    for idx in sample_idx:
        print(f"    - {df_long.iloc[idx]['content']}")
    print("-" * 40)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 1. Giảm chiều dữ liệu (W từ thuật toán NMF) xuống 2D bằng PCA để vẽ lên mặt phẳng
pca = PCA(n_components=2, random_state=42)
pca_features = pca.fit_transform(W)

# 2. Gắn tọa độ vào dataframe
df_long['pca_x'] = pca_features[:, 0]
df_long['pca_y'] = pca_features[:, 1]

# 3. Định nghĩa tên cụm cụ thể cho dễ hiểu (thay vì chỉ ghi số 0, 1, 2, 3)
topic_names = {
    0: "Tính năng, Giao diện & Voice",
    1: "Chất lượng Hỏi - Đáp",
    2: "Cập nhật Dữ liệu & Thông tin",
    3: "Công việc & Học tập"
}
df_long['Tên Cụm Chủ Đề'] = df_long['topic'].map(topic_names)

# 4. VẼ BIỂU ĐỒ DUY NHẤT
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x='pca_x', y='pca_y',
    hue='Tên Cụm Chủ Đề', # Phân loại màu sắc theo tên cụm
    palette='Set2',       # Bảng màu thân thiện, dễ nhìn
    data=df_long,
    alpha=0.8,            # Độ trong suốt của các điểm để thấy rõ vùng giao nhau
    s=50                  # Kích thước chấm
)

# Căn chỉnh tiêu đề và viền
plt.title('Biểu Đồ Phân Cụm Đánh Giá ChatGPT Theo Chủ Đề (PCA)', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Thành phần đặc trưng 1')
plt.ylabel('Thành phần đặc trưng 2')

# Đặt bảng chú thích (Legend) ra ngoài biểu đồ cho đỡ che mất dữ liệu
plt.legend(title='Các Nhóm Chủ Đề Chính', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# Hiển thị
plt.show()