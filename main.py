
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,  roc_curve, auc
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.linear_model import LogisticRegression

 

# Load CSV files into DataFrames
df_fake = pd.read_csv("dataset\Fake.csv")
df_real = pd.read_csv("dataset\True.csv")

 
# Display DataFrames
#print("Fake News DataFrame (head):")
#print(df_fake.head())

 
#print("\nReal News DataFrame (head):")
#print(df_real.head())

# Lägg till en kolumn med etikett (label)
df_fake['label'] = 0 # Fake news får etikett 0
df_real['label'] = 1  # Real news får etikett 1


# Concatenate the fake and real datasets into one combined DataFrame
df = pd.concat([df_fake, df_real], ignore_index=True)
 
df["processed_text"] = df["title"]+ " " + df["text"]
df["processed_text"] = df["processed_text"].str.lower();

 

# Fix the column names to include the correct columns and the new 'label'
df.columns = ["title", "text", "subject", "date", "label", "processed_text"]

X = df.iloc[:,-1]
Y = df.iloc[:,-2]
 
#print(X)
#print (Y)

# Split words and explode DataFrame to have each word in its own row

df_words = df.assign(word=df["processed_text"].str.split()).explode("word")

 

# Visa resultatet

#print(df_words[["word", "label"]].head())

 

seed  = 42

np.random.seed(seed)

X_train,X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.20, random_state=seed)

 

print(f"Testing sampels: {len(X_train)}, Testing samples : {len(X_test)}")

 

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

#TF_IDF = Varje ord viktas beroende på hur viktigt och ovanligt ordet är.

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

 

#print(X_train_tfidf)

 

# ==== Logistic Regression ====
#C: Inverse of regularization strength; smaller values specify stronger regularization.
#solver: Algorithm to use in the optimization problem.
#max_iter: Maximum number of iterations taken for the solvers to converge.

model_LogRed = LogisticRegression(C=0.1 ,solver='liblinear', max_iter=5000)

model_LogRed.fit(X_train_tfidf, Y_train)

 

# make predictions

y_pred_logRed = model_LogRed.predict(X_test_tfidf)

 

accuracy_logReg = accuracy_score(Y_test, y_pred_logRed)

print(f"Logistic Regression Accuracy: {accuracy_logReg:.4f}")

print(classification_report(Y_test, y_pred_logRed))

 

# ==== Random Forest ====
rf_clf = RandomForestClassifier(n_estimators=1)
rf_clf.fit(X_train_tfidf, Y_train)
Y_pred_Rf = rf_clf.predict(X_test_tfidf)

 
# Print Accuracy
accuracy_Rf = accuracy_score(Y_test, Y_pred_Rf)
print(f"Random Forest Accuracy: {accuracy_Rf:.4f}")

 

## Print Classification Report
print(classification_report(Y_test, Y_pred_Rf))

 
# ==== SVM ====
#2 FITTING A MODEL TO THE TRAINING DATA
#linear boundary to separate classes, high C -> complex might not classify unseen data correctly, low C -> better performance
svm_clf = SVC(kernel='linear', C=0.1, probability=True)

# #4 TRANSFORMING THE TEST DATA TO FIND DECISION BOUNDARY
# #finds the best location for the linear boundary
svm_clf.fit(X_train_tfidf, Y_train)
print("working 1 ... ")

 
Y_pred_SVM = svm_clf.predict(X_test_tfidf)[:,1]
print("working 2 ... ")

# Print Accuracy
accuracy_SVM = accuracy_score(Y_test, Y_pred_SVM)
print(f"SVM Accuracy: {accuracy_SVM:.4f}")

#Print Classification Report
print(classification_report(Y_test, Y_pred_SVM))

 

# ==== ROC-kurva ====

#breäkna sannolikheter
y_prob_logReg = model_LogRed.predict_proba(X_test_tfidf)[:,-1]

#fpr= False Positive Rate, tpr Ture Positive Rate
#fpr, tpr, thresholds = roc_curve(...)
#lägger en _ för att vi vill skippa thresholds.

fpr_logReg, tpr_logReg, _ = roc_curve(Y_test, y_prob_logReg)
roc_auc_logReg = auc(fpr_logReg,tpr_logReg)

 
y_prob_rf = rf_clf.predict_proba(X_test_tfidf)[:, -1]
fpr_Rf, tpr_Rf, _ = roc_curve(Y_test, y_prob_rf)
roc_auc_logRf = auc(fpr_Rf, tpr_Rf)
 
fpr_SVM, tpr_SVM, _ = roc_curve(Y_test, Y_pred_SVM)[:, -1]
roc_auc_SVM = auc(fpr_SVM,tpr_SVM)
print(classification_report(Y_test, Y_pred_SVM))


# Plotta ROC
plt.figure(figsize=(8, 6))

plt.plot(fpr_logReg, tpr_logReg, color='blue', label=f"Logistic Regression (AUC = {roc_auc_logReg:.2f})")
plt.plot(fpr_Rf, tpr_Rf, color='red', label=f"Random Forest  (AUC = {roc_auc_logRf:.2f})")
plt.plot(fpr_SVM, tpr_SVM, color='green', label=f"SVM (AUC = {roc_auc_SVM:.2f})")

 
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression / Random Forest / SVM ')
plt.legend(loc='lower right')
plt.grid(True)

plt.show()
