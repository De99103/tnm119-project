import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC


# Load CSV files into DataFrames
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

# Display DataFrames
# print("Fake News DataFrame (head):")
# print(df_fake.head())

# print("\nReal News DataFrame (head):")
# print(df_real.head())

# Lägg till en kolumn med etikett (label)
df_fake['label'] = 'Fake'  # Fake news får etikett 0
df_real['label'] = 'True'  # Real news får etikett 1

# Concatenate the fake and real datasets into one combined DataFrame
df = pd.concat([df_fake, df_real], ignore_index=True)

df["processed_text"] = df["title"] + " " + df["text"]
df["processed_text"] = df["processed_text"].str.lower()

# Fix the column names to include the correct columns and the new 'label'
df.columns = ["title", "text", "subject", "date", "label", "processed_text"]

X = df.iloc[:, -1]
Y = df.iloc[:, -2]
# print(X)
# print (Y)

# Split words and explode DataFrame to have each word in its own row
df_words = df.assign(word=df["processed_text"].str.split()).explode("word")

# Visa resultatet
# print(df_words[["word", "label"]].head())

seed = 42
np.random.seed(seed)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)

print(f"Testing sampels: {len(X_train)}, Testing samples : {len(X_test)}")

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# TF_IDF = Varje ord viktas beroende på hur viktigt och ovanligt ordet är.
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

rf_clf = RandomForestClassifier(n_estimators=7)
rf_clf.fit(X_train_tfidf, Y_train)

Y_pred = rf_clf.predict(X_test_tfidf)

# Print Accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")

# Print Classification Report
print(classification_report(Y_test, Y_pred))
