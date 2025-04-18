import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# 1. Load Dataset
df = pd.read_csv("anxiety_depression_data.csv")

# 2. Create sentiment label (based on anxiety, depression, stress)
df['Mental_State'] = df[['Anxiety_Score', 'Depression_Score', 'Stress_Level']].mean(axis=1)
df['Sentiment'] = df['Mental_State'].apply(lambda x: 1 if x >= 5 else 0)  # 1 = Negative, 0 = Positive

# 3. Drop original target columns
df.drop(columns=['Anxiety_Score', 'Depression_Score', 'Stress_Level', 'Mental_State'], inplace=True)

# 4. Encode categorical columns
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# 5. Features and target
X = df.drop('Sentiment', axis=1)
y = df['Sentiment']

# 6. Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 8. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 9. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 10. Predict on a sample
sample = X_test.iloc[0:1]
print("Sample Prediction:", model.predict(sample))
