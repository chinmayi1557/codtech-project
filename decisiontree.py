import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load Dataset
df = pd.read_csv("anxiety_depression_data.csv")  

# Step 2: Data Preprocessing
df.fillna(df.median(numeric_only=True), inplace=True)  # Handle missing values

# Convert categorical variables to numerical
df = pd.get_dummies(df, drop_first=True)

# Define Features (X) and Target (y)
target_column = 'Anxiety_Score' 
X = df.drop(columns=[target_column])
y = df[target_column]

# Step 3: Split Data into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Decision Tree Model
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Step 5: Visualize the Decision Tree
plt.figure(figsize=(15, 8))
plot_tree(clf, feature_names=X.columns, filled=True, rounded=True)
plt.show()

# Step 6: Model Evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
