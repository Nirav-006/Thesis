# -*- coding: utf-8 -*-
"""ZSJ_252

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pandas as pd

file_path = "/content/sales_data_sample.csv"
df = pd.read_csv(file_path, encoding="latin-1")

# Display basic information
print(df.info())
print(df.describe())
print(df.head())

# Handling missing values
df.dropna(inplace=True)

# Ensure 'SALES' exists
if "SALES" in df.columns:
    df["Total Price"] = df["SALES"]
else:
    print("Warning: SALES column not found in dataset")

# Define a binary target variable (High Sales vs. Low Sales)
threshold = df["Total Price"].median()
df["High_Sales"] = (df["Total Price"] > threshold).astype(int)

# Encode categorical features
encoder = LabelEncoder()
categorical_columns = ["PRODUCTLINE", "PRODUCTCODE", "DEALSIZE"]
for col in categorical_columns:
    if col in df.columns:
        df[col] = encoder.fit_transform(df[col])
    else:
        print(f"Warning: Column '{col}' not found in dataset")

# Data visualization
plt.figure(figsize=(12,6))
sns.histplot(df["Total Price"], bins=30, kde=True, color='blue')
plt.title("Distribution of Total Price")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df["High_Sales"], y=df["Total Price"], palette="coolwarm")
plt.title("Total Price vs. Sales Category")
plt.show()

# 3D Scatter Plot
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["PRODUCTLINE"], df["PRODUCTCODE"], df["Total Price"], c=df["High_Sales"], cmap='coolwarm')
ax.set_xlabel("Product Line")
ax.set_ylabel("Product Code")
ax.set_zlabel("Total Price")
plt.title("3D Visualization of Sales Data")
plt.show()

# Splitting dataset
features = ["PRODUCTLINE", "PRODUCTCODE", "QUANTITYORDERED"]
X = df[features]S
y = df["High_Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))









