#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

labels = ['unrelated', 'agreed', 'disagreed']

# Loading the cleaned data
train_data = pd.read_csv("cleaned_train_data.csv")
test_data = pd.read_csv("cleaned_test_data.csv")

# Double checking the empty rows
train_data = train_data.dropna()
test_data = train_data.dropna()

# Processing the data
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_data["title1_en"] + train_data["title2_en"])
y_train = train_data["label"]
X_test = vectorizer.transform(test_data["title1_en"] + test_data["title2_en"])

# Training the model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Saving the predictions to a submission file
test_data["label"] = y_pred
test_data[["id", "label"]].to_csv("submission.csv", index=False)

# Creating validation data (Splitting from train dataset)
val_data = train_data.sample(frac=0.2, random_state=42)
X_val = vectorizer.transform(val_data["title1_en"] + val_data["title2_en"])
y_val = val_data["label"]
y_val_pred = clf.predict(X_val)

accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred, average="macro")
recall = recall_score(y_val, y_val_pred, average="macro")
f1 = f1_score(y_val, y_val_pred, average="macro")

# Finding scores and confusion matrix for the validation data
cm = confusion_matrix(y_val, y_val_pred)

labels = ['unrelated', 'agreed', 'disagreed']

# Plotting the matrix
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted from Test Data")
plt.ylabel("Actual from Training Data")
plt.title("Confusion Matrix")
plt.show()

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

