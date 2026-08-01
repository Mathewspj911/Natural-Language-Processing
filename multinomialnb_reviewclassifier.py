from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
reviews = [
    "phone is amazing",
    "excellent battery life",
    "good performance",
    "poor camera quality",
    "worst mobile experience",
    "bad battery"
]

# Categories
categories = [
    "Good",
    "Good",
    "Good",
    "Bad",
    "Bad",
    "Bad"
]

# Convert text into numerical features
text_converter = CountVectorizer()
feature_matrix = text_converter.fit_transform(reviews)

# Train the model
classifier = MultinomialNB(alpha=1.0)
classifier.fit(feature_matrix, categories)

# Get input from the user
user_review = input("Enter a product review: ")

# Convert input into vector
new_review = text_converter.transform([user_review])

# Predict
result = classifier.predict(new_review)

# Display result
print("Predicted Category:", result[0])