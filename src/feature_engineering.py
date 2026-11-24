import pandas as pd
import numpy as np


def create_features(df):
    # -----------------------------
    # TIME FEATURES
    # -----------------------------

    # PaySim 'step' represents an hour
    df["hour"] = df["step"] % 24

    # Convert hours into days
    df["day"] = df["step"] // 24


    # -----------------------------
    # AMOUNT FEATURES
    # -----------------------------

    # Log transformation reduces the effect of extremely large amounts
    df["amount_log"] = np.log1p(df["amount"])


    # -----------------------------
    # TRANSACTION TYPE FEATURES
    # -----------------------------

    # One-hot encode transaction type
    df = pd.get_dummies(
        df,
        columns=["type"],
        drop_first=True
    )


    return df


# -----------------------------
# TEST FEATURE ENGINEERING
# -----------------------------

if __name__ == "__main__":

    file_path = "../data/raw/PS_20174392719_1491204439457_log.csv"

    df = pd.read_csv(file_path)

    # Remove identifier columns
    df = df.drop(columns=["nameOrig", "nameDest"])

    # Create features
    df = create_features(df)

    print("Feature engineering completed successfully!")

    print("\nNew dataset shape:")
    print(df.shape)

    print("\nFeature columns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())