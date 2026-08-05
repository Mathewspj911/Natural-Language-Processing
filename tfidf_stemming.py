import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download resources (only first time)
nltk.download('punkt')

# Sample text corpus
corpus = [
    "Hello!!! I have 2 apples, and I'm running fast.",
    "Text normalization is important for NLP tasks like sentiment analysis.",
    "Dogs are running faster than the foxes in the forest."
]

def stem_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

# Apply stemming
stemmed_corpus = [stem_text(doc) for doc in corpus]
print("Stemmed Corpus:", stemmed_corpus)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(stemmed_corpus)

print("\nFeature Names:", vectorizer.get_feature_names_out())
print("\nTF-IDF Matrix:\n", X.toarray())