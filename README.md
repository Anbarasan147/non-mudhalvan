import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
nltk.download('stopwords')

# Sample dataset (replace with your own dataset)
data = {
    'text': [
        'The government has introduced a new law to fight fake news.',
        'A new species of dinosaur was discovered in Australia yesterday.',
        'Scientists say that regular exercise improves brain function.',
        'Aliens have been found living in the Arctic, according to new reports.',
        'The stock market crashed today due to unforeseen events.'
    ],
    'label': [0, 1, 0, 1, 0] # 0: True news, 1: Fake news
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Preprocessing text: Removing non-alphanumeric characters and lowercasing
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Keep only letters and spaces
    text = text.lower()
    return text

df['text'] = df['text'].apply(preprocess_text)

# Split the data into features and labels
X = df['text']
y = df['label']

# Convert the text data into TF-IDF features
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
X_tfidf = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['True', 'Fake'], yticklabels=['True', 'Fake'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=['True', 'Fake']))
