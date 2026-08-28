from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def load_data():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def evaluate_model(name, model, X_test, y_test_log):
    y_pred_log = model.predict(X_test)

    rmse_log = np.sqrt(mean_squared_error(y_test_log, y_pred_log))
    mae_log = mean_absolute_error(y_test_log, y_pred_log)
    r2_log = r2_score(y_test_log, y_pred_log)

    y_test_pkr = np.expm1(y_test_log)
    y_pred_pkr = np.expm1(y_pred_log)

    rmse_pkr = np.sqrt(mean_squared_error(y_test_pkr, y_pred_pkr))
    mae_pkr = mean_absolute_error(y_test_pkr, y_pred_pkr)
    r2_pkr = r2_score(y_test_pkr, y_pred_pkr)

    return {
        "model": name,
        "rmse_log": rmse_log, "mae_log": mae_log, "r2_log": r2_log,
        "rmse_pkr": rmse_pkr, "mae_pkr": mae_pkr, "r2_pkr": r2_pkr,
    }


def print_comparison_table(results):
    df = pd.DataFrame(results).set_index("model")
    display_df = df.copy()
    display_df["rmse_pkr"] = display_df["rmse_pkr"].map(lambda x: f"PKR {x:,.0f}")
    display_df["mae_pkr"] = display_df["mae_pkr"].map(lambda x: f"PKR {x:,.0f}")
    for col in ["rmse_log", "mae_log", "r2_log", "r2_pkr"]:
        display_df[col] = display_df[col].round(4)

    print("\n" + "=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)
    print(display_df.to_string())
    print("=" * 90)
    print("Mertic r2_log used to pick the best model.\n")


def main():
    X_train, X_test, y_train, y_test = load_data()
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}\n")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    results = []
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        results.append(evaluate_model(name, model, X_test, y_test))

    print_comparison_table(results)

    results_df = pd.DataFrame(results).set_index("model")
    best_name = results_df["r2_log"].idxmax()
    best_model = models[best_name]
    print(f"BEST MODEL: {best_name}  (R² log = {results_df.loc[best_name, 'r2_log']:.4f})\n")

    # --- Save only what's needed downstream ---
    joblib.dump(models, MODELS_DIR / "all_models.pkl")      # for the SHAP notebook
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")  # for the Streamlit app
    results_df.to_csv(MODELS_DIR / "model_comparison.csv")

    with open(MODELS_DIR / "best_model_name.txt", "w") as f:
        f.write(best_name)

    print("Saved:")
    print(f"  {MODELS_DIR / 'all_models.pkl'}      (all 3 models, dict — used by the SHAP notebook)")
    print(f"  {MODELS_DIR / 'best_model.pkl'}      (used by the Streamlit app)")
    print(f"  {MODELS_DIR / 'model_comparison.csv'}")
    print(f"  {MODELS_DIR / 'best_model_name.txt'}")
    print("\nTraining Completed!")


if __name__ == "__main__":
    main()