import torch
from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import random
import re
import nltk
import umap
from tqdm import tqdm
from nltk.corpus import stopwords
import os
import pickle

# 전처리 도구 준비
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# 사용자 이름 추출 함수
def extract_user_name(file_path):
    base_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    return name_without_ext

# STEP 1: CSV 파일 불러오기
input_csv = "C:/Users/lsj39/desktop/cs372_project/csv_datas/merged_data_maximum.csv"
df = pd.read_csv(input_csv)

# STEP 2: 텍스트 전처리 함수 정의
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\bhttpurl\b|\burl\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\brt\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower().strip()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_space]
    return " ".join(tokens)

# STEP 2.5: 전처리 실행 + 캐시 저장
cache_dir = "C:/Users/lsj39/desktop/cs372_project/cache3"
cache_csv_path = os.path.join(cache_dir, "cleaned_cache_max.csv")
force_refresh = True

if os.path.exists(cache_csv_path) and not force_refresh:
    print("전처리된 CSV 파일 불러오는 중...")
    cleaned_df = pd.read_csv(cache_csv_path)
    print(f"전처리 후 남은 문서 수: {len(cleaned_df)}개")
else:
    print("🧹 전처리 실행 중...")
    cleaned_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row["text"])
        label = row["label"]
        origin_file = row.get("origin_file", "unknown_file")
        user_name = extract_user_name(origin_file)
        cleaned = clean_text(text)

        if 80 >= len(cleaned.strip()) > 60:
            cleaned_rows.append({
                "original_text": text,
                "cleaned_text": cleaned,
                "label": label,
                "user_name": user_name
            })

    cleaned_df = pd.DataFrame(cleaned_rows)
    os.makedirs(cache_dir, exist_ok=True)
    cleaned_df.to_csv(cache_csv_path, index=False, encoding="utf-8-sig")
    print(f" 전처리 결과 CSV 저장 완료: {cache_csv_path}")
    print(f" 전처리 후 남은 문서 수: {len(cleaned_rows)}개")

# STEP 2.6: 샘플링
LIMIT = 720000
if len(cleaned_df) > LIMIT:
    sampled_df = cleaned_df.sample(n=LIMIT, random_state=42).reset_index(drop=True)
else:
    sampled_df = cleaned_df

# 데이터 분리
original_texts = sampled_df["original_text"].tolist()
cleaned_texts = sampled_df["cleaned_text"].tolist()
retained_labels = sampled_df["label"].tolist()
user_names = sampled_df["user_name"].tolist()

# STEP 3: 임베딩
embedding_model = SentenceTransformer("all-mpnet-base-v2", device='cuda')
print("임베딩 중...")
embeddings = embedding_model.encode(cleaned_texts, show_progress_bar=True, batch_size=64)

# STEP 4: UMAP 차원 축소
print("UMAP 차원 축소 중...")
umap_model = umap.UMAP(n_neighbors=10, n_components=40, min_dist=0.0, metric='cosine')
reduced_embeddings = umap_model.fit_transform(embeddings)

# STEP 5: KMeans 클러스터링
print("KMeans 클러스터링 중...")
kmeans = KMeans(n_clusters=120, verbose=1, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(reduced_embeddings)

sil_score = silhouette_score(reduced_embeddings, clusters)
print(f"\n[ 실루엣 스코어] {sil_score:.4f}")

# STEP 6: BERTopic 모델링
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    calculate_probabilities=False,
    min_topic_size=80,
    verbose=True
)

print("BERTopic 구조 초기화 중...")
topic_model.fit(cleaned_texts, embeddings)
_, topics = topic_model.fit_transform(cleaned_texts, embeddings)

# STEP 7: 결과 저장
topic_info = topic_model.get_topic_info()
topic_name_map = {row["Topic"]: row["Name"] for _, row in topic_info.iterrows()}
valid_topics = set(range(kmeans.n_clusters))

final_data = [
    (orig, clean, label, uname, topic, topic_name_map.get(topic, "Unknown"))
    for orig, clean, label, uname, topic in zip(
        original_texts, cleaned_texts, retained_labels, user_names, topic_model.topics_
    )
    if topic in valid_topics
]

output_df = pd.DataFrame(final_data, columns=[
    'original_text', 'cleaned_text', 'label', 'user_name', 'topic', 'topic_name'
])

output_csv = "C:/Users/lsj39/Desktop/cs372_project/csv_datas/final_cleaned_topic_results_maximum_samelength2.csv"
output_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

# STEP 8: 시각화 및 요약
print("\n[ 토픽 요약]")
print(topic_info)
topic_model.visualize_topics().show()
