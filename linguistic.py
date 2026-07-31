import re
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Input text
text = "John bought 3 laptops in New York on 15/06/2026."

# Regex Transformation using re.findall()
words = re.findall(r'[A-Za-z]+', text)
transformed_text = " ".join(words)

print("Transformed Text:")
print(transformed_text)

# Linguistic Analysis
doc = nlp(transformed_text)

print("\nWord\t\tPOS Tag")
for token in doc:
    print(token.text, "\t\t", token.pos_)