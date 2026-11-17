import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from feature_engineering import create_features


# -----------------------------
# LOAD DATASET
# -----------------------------

file_path = "../data/raw/PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)


# -----------------------------
# REMOVE IDENTIFIERS
# -----------------------------

df = df.drop(columns=["nameOrig", "nameDest"])


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

df = create_features(df)

print("Features created successfully!")
print("Feature-engineered shape:", df.shape)


# -----------------------------
# REMOVE TARGET FROM FEATURES
# -----------------------------

X = df.drop(columns=["isFraud"])
y = df["isFraud"]


# -----------------------------
# REMOVE BALANCE FEATURES
# -----------------------------

balance_columns = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

X = X.drop(columns=balance_columns)


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# -----------------------------
# FEATURE SCALING
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# LOGISTIC REGRESSION
# -----------------------------

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    class_weight="balanced",
    max_iter=100,
    random_state=42
)

model.fit(X_train_scaled, y_train)


print("Model training completed!")


# -----------------------------
# PREDICTIONS
# -----------------------------

y_pred = model.predict(X_test_scaled)

y_probability = model.predict_proba(X_test_scaled)[:, 1]


# -----------------------------
# MODEL EVALUATION
# -----------------------------

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_probability))


# -----------------------------
# SAVE MODEL AND SCALER
# -----------------------------

joblib.dump(model, "../model/fraud_model.pkl")
joblib.dump(scaler, "../model/scaler.pkl")


print("\nModel saved successfully!")
print("Saved: model/fraud_model.pkl")
print("Saved: model/scaler.pkl")