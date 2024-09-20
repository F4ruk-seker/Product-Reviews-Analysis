from collections import Counter

import joblib
import pandas as pd
from typing import Iterator


logreg = joblib.load('trained/logreg_model_2024-09-20_11-24-57.pkl')


def analysis_comments(comments: list[str]) -> Iterator[dict]:
    comment_series = pd.Series(comments)
    results = logreg.predict(comment_series)
    for comment, result in zip(comments, results):
        yield {'comment': comment, 'result': result}

# test
# df['Görüş'] = df['Görüş'].str.lower()
# df['Görüş'] = df['Görüş'].str.replace(r'[^\w\s]', '')
# df['Görüş'] = df['Görüş'].str.replace(r'\d', '')
import re


def temizle(text):
    # Emojileri kaldır
    text = re.sub(r'[^\w\s,]', '', text)

    # Fazla boşlukları kaldır
    text = re.sub(r'\s+', ' ', text).strip()

    return text


with open('comment.txt', 'r', encoding='utf-8') as cf:
    comments = [temizle(text) for text in cf.read().split('\n')]

print(comments)
analysed = list(analysis_comments(comments))

for _ in analysed:
    print(_)

# Olumsuz yorumları filtreleme
def repiter(param: str, comments):
    negative_comments = [c['comment'] for c in comments if c['result'] == param]

    # Yorumların tekrar sayısını bulma
    comment_counts = Counter(negative_comments)

    # En çok ve en az tekrar eden yorumları bulma
    most_common = comment_counts.most_common(3)  # En çok tekrar eden 3
    least_common = comment_counts.most_common()[-3:]  # En az tekrar eden 3

    # Sonuçları yazdırma
    print(f"En Çok Tekrar Eden {param} Yorumlar:")
    for comment, count in most_common:
        print(f"- {comment} (Tekrar: {count})")

    print(f"\nEn Az Tekrar Eden {param} Yorumlar:")
    for comment, count in least_common:
        print(f"- {comment} (Tekrar: {count})")

    print('\n')


repiter('Olumlu', analysed)
repiter('Olumsuz', analysed)
repiter('Tarafsız', analysed)
