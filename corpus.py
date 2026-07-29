import nltk
import re
from nltk.corpus import gutenberg, stopwords

# Download required corpora (run once)
nltk.download('gutenberg')
nltk.download('stopwords')

# Display available file IDs
print("Available File IDs:")
print(gutenberg.fileids())

# Load a dataset using fileid
text = gutenberg.raw(gutenberg.fileids()[0])

print("\nOriginal Text Sample:")
print(text[:200])

# Text Cleaning

# Convert to lowercase
text = text.lower()

# Remove special characters and numbers
text = re.sub(r'[^a-z\s]', '', text)

# Tokenization
words = text.split()

# Remove stopwords
stop_words = set(stopwords.words('english'))
clean_words = [word for word in words if word not in stop_words]

print("\nCleaned Words Sample:")
print(clean_words[:50])