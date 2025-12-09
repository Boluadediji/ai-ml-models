import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import warnings
import re


warnings.filterwarnings('ignore')
def clean_text(text):
    """Clean the text - like washing vegetables before cooking"""
    text = text.lower()  # Make everything lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation !?.,
    return text

# Load data
df = pd.read_csv("bank_intents.csv")

# Clean all the questions
df['text'] = df['text'].apply(clean_text)

# Split data - 80% training, 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Create word features - NOW WITH BIGRAMS!
vectorizer = TfidfVectorizer(
    max_features=5000,  # Only use most common 5000 words/phrases
    ngram_range=(1, 2),  # Use single words AND two-word phrases
    stop_words='english'  # Remove common words like 'the', 'and', 'is'
)

# Convert text to numbers
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model with better settings
model = LogisticRegression(
    C=1.0,  # How strict vs flexible the model should be
    max_iter=1000,  # Give it more time to learn
    class_weight='balanced'  # Pay equal attention to rare and common questions
)

# Train the model
model.fit(X_train_vec, Y_train)

# TEST HOW GOOD IT IS (VERY IMPORTANT!)

# Test on the practice questions it saw
train_predictions = model.predict(X_train_vec)
train_accuracy = accuracy_score(Y_train, train_predictions)
print(f"Accuracy on training questions: {train_accuracy:.2%}")

# Test on NEW questions it hasn't seen
test_predictions = model.predict(X_test_vec)
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f"Accuracy on new questions: {test_accuracy:.2%}")

# Detailed report - shows which categories it's good/bad at
print("\nDetailed Performance Report:")
print(classification_report(Y_test, test_predictions))

# CROSS-VALIDATION - Test multiple times to be sure
print("\nCross-Validation Results (5 different tests):")
cv_scores = cross_val_score(model, X_train_vec, Y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean():.2%}")

# Save the trained model
with open("bank_model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("Model trained and saved successfully!")