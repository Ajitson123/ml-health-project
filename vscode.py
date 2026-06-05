import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
text = """Patient came with complaints of fever, cough, and shortness of breath. He has a history of hypertension and diabetes. On examination, his temperature was 10
2°F, blood pressure was 150/90 mmHg, and oxygen saturation was 92%. He was diagnosed with pneumonia and started on antibiotics. He is also advised to monitor his blood sugar levels closely."""
print("=" * 50)
print("Original Text:")
print(text)
#1. Word Tokenization:
print('\n1. Word Tokenization:')
tokens = word_tokenize(text)
print(tokens)
#2 Sentence Tokenization:
print('\n2. sentence Tokenization:')
sentences = nltk.sent_tokenize(text)
for i, sentence in enumerate(sentences):
    print(f"Sentence {i + 1}: {sentence}")
#3. Stop Words Removal:
print('\n3. Stop Words Removal:')
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
print("before:", tokens)
print("after:", filtered_tokens)
#4. Stemming:
print('\n4. Stemming:')
stemmer = SnowballStemmer("english")
stemmed = [stemmer.stem(token) for token in filtered_tokens]
print("stemmed:", stemmed)
#5. Lemmatization:
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(token) for token in filtered_tokens]
print("lemmatized:", lemmatized)
#6. Bag of Words:
print('\n6. Bag of Words:')
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform([text])
print("Vocabulary:", vectorizer.get_feature_names_out())
print("Bag of Words Representation:")
print(X.toarray())