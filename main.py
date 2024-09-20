# pandas
from datetime import datetime

import pandas as pd
import numpy as np
# import tweepy
import nltk
nltk.download('all')

from textblob import Word, TextBlob
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from warnings import filterwarnings
from collections import Counter
import joblib
import os

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression


filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 400)
pd.set_option('display.max_rows', None)

# df = df.reset_index()

df = pd.read_csv('magaza_yorumlari_duygu_analizi.csv', encoding='utf-16')

print(df.head())
# Görüş,Durum

print(df["Durum"].value_counts())

'''
Durum
Olumlu     |  4253
Olumsuz    |  4238
Tarafsız   |  2938
'''


# karakter temizleme

df['Görüş'] = df['Görüş'].str.lower()
df['Görüş'] = df['Görüş'].str.replace(r'[^\w\s]', '')
df['Görüş'] = df['Görüş'].str.replace(r'\d', '')
# nltk.download('stopwords')

sw = stopwords.words('turkish')
df['Görüş'] = df['Görüş'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
nltk.download("punkt")
df['Görüş'].apply(lambda x: TextBlob(x).words).head()
nltk.download('wordnet')
df['Görüş'] = df['Görüş'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# Tüm metinleri tek bir string halinde birleştirip split ediyoruz
all_words = " ".join(df['Görüş']).split()

# Kelime frekanslarını hesaplıyoruz
word_counts = Counter(all_words)

# DataFrame'e çevirip sıklığa göre sıralıyoruz
tf1 = pd.DataFrame(word_counts.items(), columns=["kelimeler", "sıklık"]).sort_values(by="sıklık", ascending=False)

tf1.columns = ["kelimeler","sıklık"]

# sıra1lanan verinin i1lk 10
tf1.sort_values(by="sıklık",ascending=False).head(10)

X = df["Görüş"]
y = df["Durum"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 46)

logreg = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ('clf', LogisticRegression(n_jobs=1, C=1e5)),])
logreg.fit(X_train, y_train)

accuracy = model_selection.cross_val_score(logreg, X_test, y_test, cv=5).mean()

yeni_yorum = pd.Series("Sipariş ettiğim zaman fazla geçmeden teslim aldım ve paketiyle birlikte güzel kargolanmıştı. üründen memnunum, 1.5 aydır kullanıyorum.")
result = logreg.predict(yeni_yorum)

print(dir(result))

# eğitilen veriyi kaydet


# Modeli eğittikten sonra kaydetme

# Bugünün tarih ve saatini almak için formatlı zaman bilgisi
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Dosya adını belirleme
model_file_path = f"trained/logreg_model_{current_time}.pkl"

# Modeli kaydetme
os.makedirs('trained', exist_ok=True)  # Dizini oluşturur eğer yoksa
joblib.dump(logreg, model_file_path)

print(f"Model {model_file_path} olarak kaydedildi.")