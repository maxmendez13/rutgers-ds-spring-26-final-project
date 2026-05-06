
# libraries
import pandas as pd
import numpy as np

# text processing
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# evaluation
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

# load data
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 1
true_df["label"] = 0

# combine datasets
df = pd.concat([fake_df, true_df], axis=0)

# shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

print("Dataset shape:", df.shape)
print(df.head())

# data preprocessing
df["content"] = df["title"] + " " + df["text"]

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df["content"] = df["content"].apply(clean_text)

# feature extraction (TF-IDF)
# initializer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X = vectorizer.fit_transform(df["content"])

y = df["label"]

# train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)


# train model (logitic regression)
model = LogisticRegression(max_iter = 1000)
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))