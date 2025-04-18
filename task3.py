from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# 1. Load dataset
digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))  # Flatten 8x8 image
y = digits.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# 4. Predict
y_pred = clf.predict(X_test)

# 5. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6. Show sample prediction
plt.imshow(digits.images[0], cmap='gray')
plt.title(f"Prediction: {clf.predict([X[0]])[0]}")
plt.axis('off')
plt.show()
