from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

# 데이터 불러오기
with open('all_xml_texts.txt', 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f if line.strip()]

# 문장 수 제한
LIMIT = 500000
if len(texts) > LIMIT:
    texts = random.sample(texts, LIMIT)

# 임베딩
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("문장 임베딩 중...")
embeddings = []
for text in tqdm(texts, desc="Embedding 진행 중"):
    embeddings.append(embedding_model.encode(text, show_progress_bar=False))
embeddings = np.array(embeddings)

# HDBSCAN 클러스터링 설정
hdbscan_model = HDBSCAN(
    min_cluster_size=44,   # 최소 군집 크기 (ex: 20개 이상 문장이 모여야 하나의 주제 인정)
    min_samples=10,        # 군집의 밀도 (ex: 밀도 기준을 완화하거나 빡세게 설정)
    metric='euclidean',    # 거리 기준
    cluster_selection_method='eom'
)

# BERTopic 생성 (HDBSCAN 모델 적용)
topic_model = BERTopic(
    language="multilingual",
    hdbscan_model=hdbscan_model
)

# 학습
topics, probs = topic_model.fit_transform(texts, embeddings)

# 결과 저장
df = pd.DataFrame({
    'text': texts,
    'topic': topics,
    'probability': probs
})

df.to_csv('results_with_topics.csv', index=False)

# 결과 출력
print("\n[전체 주제 요약]")
print(topic_model.get_topic_info())
topic_model.visualize_topics().show()