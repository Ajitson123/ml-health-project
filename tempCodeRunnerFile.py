import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import  CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
clinical_notes = [
    "Patient presents with a history of hypertension and diabetes. Blood pressure is elevated at 150/95 mmHg. Patient reports occasional headaches and dizziness. No chest pain or shortness of breath. Medications include lisinopril and metformin. Recommend lifestyle modifications and follow-up in 2 weeks.",
    "elderly patient with a history of chronic obstructive pulmonary disease (COPD) and smoking. Patient reports increased shortness of breath and productive cough. Oxygen saturation is 88%. Medications include albuterol and tiotropium. Recommend pulmonary function tests and smoking cessation counseling.",
    "female patient with a history of breast cancer. Patient reports fatigue and weight loss. No new lumps or breast pain. Medications include tamoxifen and vitamin D supplements. Recommend follow-up with oncologist and routine mammogram.",
    "patient presents with a history of heart failure and atrial fibrillation. Patient reports palpitations and shortness of breath. Blood pressure is 110/70 mmHg. Medications include warfarin and furosemide. Recommend echocardiogram and cardiology follow-up.",
    "diabetic patient with a history of neuropathy and retinopathy. Patient reports numbness and tingling in the feet. Blood sugar is elevated at 250 mg/dL. Medications include insulin and gabapentin. Recommend blood sugar monitoring and ophthalmology follow-up.",
    "patient presents with a history of stroke and hypertension. Patient reports weakness on the left side and difficulty speaking. Blood pressure is elevated at 160/100 mmHg. Medications include aspirin and lisinopril. Recommend physical therapy and neurology follow-up.",
    "patient presents with a history of kidney disease and anemia. Patient reports fatigue and decreased urine output. Blood pressure is elevated at 145/90 mmHg. Medications include erythropoietin and iron supplements. Recommend nephrology follow-up and blood work.",
]
labels = [1, 1, 1, 0, 0, 0, 0]
print("=== CLINICAL NLP PIPELINE ===")
print(f"Clinical notes: {len(clinical_notes)}")
print('\n--- Step 1: Tokenization ---')
sample = clinical_notes[0]
print(f"Sample clinical note: {sample}")
word_tokens = word_tokenize(sample)
print(f"Word tokens: {word_tokens}")
sent_tokens = sent_tokenize(sample)
print(f"Sentence tokens: {sent_tokens}")

print('\n--- Step 2: Stopword Removal ---')
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in word_tokens if word.lower() not in stop_words]
print(f"Filtered tokens: {filtered_tokens}")

print('\n--- Step 3: Stemming ---')
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
print(f"Stemmed tokens: {stemmed_tokens}")

print('\n--- Step 4: Lemmatization ---')
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
print(f"Lemmatized tokens: {lemmatized_tokens}")

# FULL PREPROCESSING FUNCTION
import re
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens
processed_notes = [preprocess_text(note) for note in clinical_notes]
print(f"\n--- Step 5: Full Preprocessing ---")
for i, note in enumerate(processed_notes[:3]):
    print(f"Processed note {i+1}: {note}")

print('\n--- step 6: Bag of Words (BoW) ---')
bow = CountVectorizer(max_features=20)
x_bow = bow.fit_transform(clinical_notes)
print(f"bow shape: {x_bow.shape}")
print(f"vocabulary sample: {list(bow.vocabulary_.items())[:7]}")

print('\n--- step 7: TF-IDF ---')
tfidf = TfidfVectorizer(max_features=20)
x_tfidf = tfidf.fit_transform(clinical_notes)
print(f"tfidf shape: {x_tfidf.shape}")
df_tfidf = pd.DataFrame(x_tfidf.toarray(), columns=tfidf.get_feature_names_out())
print(df_tfidf.round(3))

print('\n--- step 8: Word2Vec ---')
w2v_model = Word2Vec(sentences=processed_notes, vector_size=50, window=5, min_count=1, workers=4, epochs=100)
print("Words similar to 'patient':")
try:
    similar_words = w2v_model.wv.most_similar('patient', topn=5)
    for word, score in similar_words:
        print(f"{word}: {score:.4f}")
except:
    print("Not enough data for similarity analysis with 'patient'.")
print("\nWords similar to 'blood':")
try:
    similar_words = w2v_model.wv.most_similar('blood', topn=5)
    for word, score in similar_words:
        print(f"{word}: {score:.4f}")
except:
    print("Not enough data for similarity analysis with 'blood'.")
print(f"\nVector for 'chest'(first 5 dimensions): {w2v_model.wv['chest'][:5]}")

print('\n --- Step 9: ML ON CLINICAL TEXT ---')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

y = np.array(labels)
lr = LogisticRegression(random_state=42)
lr_scores = cross_val_score(lr, x_tfidf, y, cv=3, scoring='accuracy')
print(f"Logistic Regression Accuracy: {lr_scores.mean():.4f} (+/- {lr_scores.std():.4f})")

nb = MultinomialNB()
nb_scores = cross_val_score(nb, x_tfidf, y, cv=3, scoring='accuracy')
print(f"Naive Bayes Accuracy: {nb_scores.mean():.4f} (+/- {nb_scores.std():.4f})")

print("\n=== END OF PIPELINE ===")