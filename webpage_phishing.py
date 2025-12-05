
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

file_path = r"C:\Users\mahas\Downloads\phishing_data.csv.zip"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nFirst 5 rows:\n", df.head())


print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget value counts:")
target_col = 'phishing'
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found. Check df.columns output.")
print(df[target_col].value_counts())


sns.countplot(x=target_col, data=df)
plt.title("Target Class Distribution")
plt.show()

X = df.drop(target_col, axis=1)
y = df[target_col]

# Optional: check if categorical columns exist
categorical_cols = X.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print("\nEncoding categorical columns:", categorical_cols)
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)
print("\nBest Hyperparameters:", grid_search.best_params_)

model = grid_search.best_estimator_


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


importances = model.feature_importances_
feature_importance = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)

print("\nTop 20 Most Important Features:")
for feat, score in feature_importance[:20]:
    print(f"{feat:30s} {score:.4f}")


top_features = feature_importance[:20]
features, scores = zip(*top_features)
sns.barplot(x=scores, y=features)
plt.title("Top 20 Feature Importances")
plt.show()


model_file = "phishing_rf_model.pkl"
scaler_file = "scaler.pkl"

joblib.dump(model, model_file)
joblib.dump(scaler, scaler_file)

print(f"\nModel saved as: {model_file}")
print(f"Scaler saved as: {scaler_file}")
