import pandas as pd
from sklearn.model_selection import train_test_split


# -----------------------------
# LOAD DATASET
# -----------------------------

file_path = "../data/raw/PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)


# -----------------------------
# REMOVE IDENTIFIER COLUMNS
# -----------------------------

df = df.drop(columns=["nameOrig", "nameDest"])


# -----------------------------
# ENCODE TRANSACTION TYPE
# -----------------------------

df = pd.get_dummies(
    df,
    columns=["type"],
    drop_first=True
)


# -----------------------------
# SEPARATE FEATURES AND TARGET
# -----------------------------

X = df.drop(columns=["isFraud"])
y = df["isFraud"]


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


# -----------------------------
# DISPLAY RESULTS
# -----------------------------

print("Preprocessing completed successfully!")

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())