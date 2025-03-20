
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load CSV files into DataFrames
df_fake = pd.read_csv("/Users/deemaabogheda/Desktop/study/Termin6/TNM119/project /dataset/Fake.csv")
df_real = pd.read_csv("/Users/deemaabogheda/Desktop/study/Termin6/TNM119/project /dataset/True.csv")

# Display DataFrames
#print("Fake News DataFrame (head):")
#print(df_fake.head())

#print("\nReal News DataFrame (head):")
#print(df_real.head())

# Lägg till en kolumn med etikett (label)
df_fake['label'] = 'Fake'  # Fake news får etikett 0
df_real['label'] = 'True'  # Real news får etikett 1

# Concatenate the fake and real datasets into one combined DataFrame
df = pd.concat([df_fake, df_real], ignore_index=True)

# Fix the column names to include the correct columns and the new 'label'
df.columns = ["title", "text", "subject", "date", "label"]

X = df.iloc[:,:-1]
Y = df.iloc[:,-1]
#print(X)
#print (Y)

df.columns

# Print first row clearly
#first_row = df.head(1)
#print(first_row)

seed  = 42
np.random.seed(seed)
X_train,X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.20, random_state=seed)




# random forest 
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
rf_classifier.fit(X_train, Y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy and classification report
accuracy = accuracy_score(Y_test, y_pred)
classification_rep = classification_report(Y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)


