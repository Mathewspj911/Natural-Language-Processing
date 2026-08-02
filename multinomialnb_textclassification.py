# Multinomial Naive Bayes Example
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score

# 1. Load dataset (text classification)
data = fetch_20newsgroups(subset='all', categories=['rec.sport.baseball', 'sci.space'])
X, y = data.data, data.target

# 2. Convert text to word counts
vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(X)

# 3. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_counts, y, test_size=0.3, random_state=42)

# 4. Train Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
