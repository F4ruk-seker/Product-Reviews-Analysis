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
for _ in analysis_comments(comments):
    print(_)