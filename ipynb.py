# -*- coding: utf-8 -*-
# Tokopedia Product Reviews 2025 — EDA and Sentiment Analysis Notebook

# 1) Imports — Core Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Download NLTK resources
nltk.download('stopwords')

# 2) Load Dataset
df = pd.read_csv('/kaggle/input/tokopedia-product-reviews-2025/data.csv')
df.head(), df.shape

# View columns
df.columns

# Summary statistics
df.describe(include='all')

# Count missing values
df.isnull().sum()

# Distribution of Ratings
sns.countplot(x='rating', data=df)
plt.title('Rating Count Distribution')
plt.show()


# Indonesian stopwords
stop_words = set(stopwords.words('indonesian'))
# Optionally add custom stop words
extra_stop = ['tokopedia', 'produk', 'sangat', 'banget']
stop_words.update(extra_stop)

stemmer = SnowballStemmer(language='indonesian')

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs, numbers and non-letters
    text = re.sub(r'http\S+|[^a-zA-Z\s]', ' ', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords & stem
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply cleaning
df['clean_review'] = df['review'].astype(str).apply(clean_text)
df.head()


# Combine all reviews
text_all = " ".join(df['clean_review'])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_all)
plt.figure(figsize=(15,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words in Reviews')
plt.show()


X = df['clean_review']
y = df['sentiment']   # assumed label column (positive/negative)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

