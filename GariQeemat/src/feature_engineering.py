import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_data.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RARE_BRAND_THRESHOLD = 10  # brands with fewer listings than this -> 'Other'
RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = ["Car_Age", "Mileage_km", "Engine_cc", "Mileage_per_year"]
ONE_HOT_FEATURES = ["City", "Fuel Type", "Transmission", "Brand_Grouped"]
TARGET_COLUMN = "log_price"


def load_cleaned_data(path: Path = CLEANED_DATA_PATH):
    df = pd.read_csv(path)
    print(f"Loaded cleaned data: {df.shape}")
    return df


def bucket_rare_brands(df: pd.DataFrame, threshold: int = RARE_BRAND_THRESHOLD):
    """Group brands with fewer than `threshold` listings into 'Other'."""
    brand_counts = df["Brand"].value_counts()
    rare_brands = brand_counts[brand_counts < threshold].index.tolist()

    df["Brand_Grouped"] = df["Brand"].apply(
        lambda b: "Other" if b in rare_brands else b
    )

    print(f"Brands kept individually ({df['Brand_Grouped'].nunique() - 1}): "
          f"{sorted(set(df['Brand_Grouped']) - {'Other'})}")
    print(f"Brands bucketed into 'Other' ({len(rare_brands)}): {rare_brands}")
    return df


def add_mileage_per_year(df: pd.DataFrame):
    """Usage intensity feature — mileage independent of age."""
    df["Mileage_per_year"] = df["Mileage_km"] / df["Car_Age"].clip(lower=1)
    return df


def add_model_frequency_encoding(df: pd.DataFrame):
    """Frequency-encode Model: rarer models get lower values, common ones higher.
    Returns the df plus the lookup dict (needed at inference time for new inputs).
    """
    model_freq = (df["Model"].value_counts() / len(df)).to_dict()
    df["Model_freq"] = df["Model"].map(model_freq)
    return df, model_freq


def build_fair_price_lookup(df: pd.DataFrame):
    """Brand+Model median price lookup — used by the app for the
    'fair price' indicator. Kept separate from the model's training features
    to avoid leaking price information into the feature set.
    """
    lookup = (
        df.groupby(["Brand", "Model"])["Price_PKR"]
        .median()
        .reset_index()
        .rename(columns={"Price_PKR": "group_median_price"})
    )
    return lookup


def one_hot_encode(df: pd.DataFrame, columns: list[str]):
    return pd.get_dummies(df, columns=columns, drop_first=False, dtype=int)


def build_feature_matrix(df: pd.DataFrame):
    """Assembles X (features) and y (target) from the cleaned dataframe."""
    df = df.copy()

    # 1. Engineered features
    df = add_mileage_per_year(df)
    df = bucket_rare_brands(df)
    df, model_freq_lookup = add_model_frequency_encoding(df)

    # 2. Fair-price lookup (business feature, not fed to the model)
    fair_price_lookup = build_fair_price_lookup(df)

    # 3. One-hot encode low-cardinality categoricals
    df_encoded = one_hot_encode(df, ONE_HOT_FEATURES)

    # 4. Assemble final feature set
    one_hot_cols = [
        c for c in df_encoded.columns
        if c.startswith(("City_", "Fuel Type_", "Transmission_", "Brand_Grouped_"))
    ]
    feature_columns = NUMERIC_FEATURES + ["Model_freq"] + one_hot_cols

    X = df_encoded[feature_columns].copy()
    y = df_encoded[TARGET_COLUMN].copy()

    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Features used ({len(feature_columns)}): {feature_columns}")

    return X, y, feature_columns, model_freq_lookup, fair_price_lookup


def scale_numeric_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scale only the numeric columns; one-hot columns are already 0/1."""
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[NUMERIC_FEATURES + ["Model_freq"]] = scaler.fit_transform(
        X_train[NUMERIC_FEATURES + ["Model_freq"]]
    )
    X_test_scaled[NUMERIC_FEATURES + ["Model_freq"]] = scaler.transform(
        X_test[NUMERIC_FEATURES + ["Model_freq"]]
    )
    return X_train_scaled, X_test_scaled, scaler


def main():
    df = load_cleaned_data()

    # Sanity check: required columns present
    required = ["Car_Age", "Mileage_km", "Engine_cc", "Brand", "Model",
                "City", "Fuel Type", "Transmission", "Price_PKR", "log_price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns from cleaned_data.csv: {missing}")

    X, y, feature_columns, model_freq_lookup, fair_price_lookup = build_feature_matrix(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Scale numeric features (fit on train only, to avoid leakage)
    X_train_scaled, X_test_scaled, scaler = scale_numeric_features(X_train, X_test)

    # ---------------------------------------------------------------
    # Save everything train_models.py and the Streamlit app will need
    # ---------------------------------------------------------------
    X_train_scaled.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test_scaled.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    fair_price_lookup.to_csv(PROCESSED_DIR / "fair_price_lookup.csv", index=False)

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    with open(MODELS_DIR / "model_freq_lookup.json", "w") as f:
        json.dump(model_freq_lookup, f, indent=2)

    rare_brands = (
        df["Brand"].value_counts()[df["Brand"].value_counts() < RARE_BRAND_THRESHOLD]
        .index.tolist()
    )
    with open(MODELS_DIR / "rare_brands.json", "w") as f:
        json.dump(rare_brands, f, indent=2)

    print("\nSaved:")
    print(f"  {PROCESSED_DIR / 'X_train.csv'}")
    print(f"  {PROCESSED_DIR / 'X_test.csv'}")
    print(f"  {PROCESSED_DIR / 'y_train.csv'}")
    print(f"  {PROCESSED_DIR / 'y_test.csv'}")
    print(f"  {PROCESSED_DIR / 'fair_price_lookup.csv'}")
    print(f"  {MODELS_DIR / 'scaler.pkl'}")
    print(f"  {MODELS_DIR / 'feature_columns.json'}")
    print(f"  {MODELS_DIR / 'model_freq_lookup.json'}")
    print(f"  {MODELS_DIR / 'rare_brands.json'}")
    print("\nFeature engineering complete.")


if __name__ == "__main__":
    main()