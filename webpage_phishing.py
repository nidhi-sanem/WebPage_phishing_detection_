# -------------------------------
# Phishing Website Detection using Random Forest
# -------------------------------

# 1. Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# 2. Load Dataset
# Make sure the path to the CSV or ZIP file is correct
file_path = r"C:\Users\mahas\Downloads\phishing_data.csv.zip"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nColumns:\n", df.columns)

# 3. Check target column
target_col = 'phishing'

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found. Check df.columns output.")

print("\nTarget value counts:\n", df[target_col].value_counts())

# 4. Features & Target
X = df.drop(target_col, axis=1)
y = df[target_col]

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# 8. Make Predictions
y_pred = model.predict(X_test)

# 9. Evaluation
print("\nAccuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 10. Feature Importance
importances = model.feature_importances_

print("\nTop 20 Most Important Features:")
feature_importance = sorted(
    zip(X.columns, importances),
    key=lambda x: x[1],
    reverse=True
)

for feat, score in feature_importance[:20]:
    print(f"{feat:30s} {score:.4f}")
