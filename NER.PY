import pandas as pd 
import numpy as np 
import spacy
v = pd.read_csv('mtsamples.csv')
v = v.dropna(subset=['transcription'])
v = v[[ 'medical_specialty', 'transcription']]
print(v["medical_specialty"].value_counts().head(10))
nlp = spacy.load("en_core_web_sm")
samples = v["transcription"].iloc[0]
doc = nlp(samples)
for ent in doc.ents:
    print(ent.text, ent.label_)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
v['sentiment'] = v['transcription'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
v['sentiment_label'] = v['sentiment'].apply(lambda x: 'Positive' if x >= 0.5 else ("negative" if x <= -0.05 else "Neutral"))
print(v['sentiment_label'].value_counts())
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
top5 = v['medical_specialty'].value_counts().head(5).index
v5 = v[v["medical_specialty"].isin(top5)]
x_train, x_test, y_train, y_test = train_test_split( v5['transcription'], v5['medical_specialty'], test_size=0.2, random_state=42)
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
x_train_tf = tfidf.fit_transform(x_train)
x_test_tf = tfidf.transform(x_test)
model = LogisticRegression(max_iter=1000)
model.fit(x_train_tf, y_train)
print(classification_report(y_test, model.predict(x_test_tf)))
