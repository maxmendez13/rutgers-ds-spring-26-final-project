# Max Mendez 
# Section 1
# jam1116
# Fake news detector using TF-IDF and Logistic Regression



# libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
df = df.dropna(subset=["title", "text"])
df["content"] = df["title"] + " " + df["text"]

def clean_text(text):
    text = text.lower()

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # remove numbers
    text = ''.join([char for char in text if not char.isdigit()])

    # remove extra spaces
    text = ' '.join(text.split())

    return text

df["content"] = df["content"].apply(clean_text)

# feature extraction (TF-IDF)
# initializer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X = vectorizer.fit_transform(df["content"])

y = df["label"]

# train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
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


feature_names = vectorizer.get_feature_names_out()

coefficients = model.coef_[0]

top_fake = np.argsort(coefficients)[-10:]
top_real = np.argsort(coefficients)[:10]

print("\nTop words predicting FAKE news:")
for i in reversed(top_fake):
    print(feature_names[i])

print("\nTop words predicting REAL news:")
for i in top_real:
    print(feature_names[i])



cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["Real", "Fake"],
    yticklabels=["Real", "Fake"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix: Fake News Detector Using TF-IDF and Logistic Regression")

plt.show()