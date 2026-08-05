# 🐍 Text Classification with Naive Bayes & SVM (Brown Corpus)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import nltk
from nltk.corpus import brown

# Make sure Brown corpus is downloaded
nltk.download('brown')

# ---------------------------
# Helper function for metrics
# ---------------------------
def get_metrics(true_labels, predicted_labels):
    print("Accuracy:", np.round(metrics.accuracy_score(true_labels, predicted_labels), 2))
    print("Precision:", np.round(metrics.precision_score(true_labels, predicted_labels, average='weighted'), 2))
    print("Recall:", np.round(metrics.recall_score(true_labels, predicted_labels, average='weighted'), 2))
    print("F1 Score:", np.round(metrics.f1_score(true_labels, predicted_labels, average='weighted'), 2))

def train_predict_evaluate_model(classifier, train_features, train_labels, test_features, test_labels):
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    get_metrics(true_labels=test_labels, predicted_labels=predictions)
    return predictions

# ---------------------------
# Load Brown Corpus
# ---------------------------
categories = ['news', 'editorial', 'reviews', 'religion']  # pick a few categories
docs = []
labels = []

for cat in categories:
    for fileid in brown.fileids(categories=cat):
        docs.append(" ".join(brown.words(fileid)))
        labels.append(cat)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(docs, labels, test_size=0.3, random_state=42)

# ---------------------------
# Feature extraction
# ---------------------------
# Bag of Words
bow_vectorizer = CountVectorizer(stop_words='english')
bow_train_features = bow_vectorizer.fit_transform(X_train)
bow_test_features = bow_vectorizer.transform(X_test)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_train_features = tfidf_vectorizer.fit_transform(X_train)
tfidf_test_features = tfidf_vectorizer.transform(X_test)

# ---------------------------
# Models
# ---------------------------
mnb = MultinomialNB()
svm = SGDClassifier(loss='hinge', max_iter=50)

print("\n🔹 Multinomial Naive Bayes with Bag of Words")
mnb_bow_predictions = train_predict_evaluate_model(mnb, bow_train_features, y_train, bow_test_features, y_test)

print("\n🔹 SVM with Bag of Words")
svm_bow_predictions = train_predict_evaluate_model(svm, bow_train_features, y_train, bow_test_features, y_test)

print("\n🔹 Multinomial Naive Bayes with TF-IDF")
mnb_tfidf_predictions = train_predict_evaluate_model(mnb, tfidf_train_features, y_train, tfidf_test_features, y_test)

print("\n🔹 SVM with TF-IDF")
svm_tfidf_predictions = train_predict_evaluate_model(svm, tfidf_train_features, y_train, tfidf_test_features, y_test)

# ---------------------------
# Confusion Matrix Example
# ---------------------------
cm = metrics.confusion_matrix(y_test, mnb_tfidf_predictions, labels=categories)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.title("Confusion Matrix (MNB + TF-IDF)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
