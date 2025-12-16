import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "data/advertising.csv"
MODEL_PATH = "models/lr_pipeline.joblib"

# Mapping konsep untuk laporan & UI:
# TV        -> Meta Ads Spend
# Radio     -> Google Ads Spend
# Newspaper -> TikTok Ads Spend
FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"  # dianggap sebagai proxy "GMV"


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}\n"
            "Pastikan file advertising.csv ada."
        )

    df = pd.read_csv(path)

 
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])


    rename_map = {
        "TV Ad Budget ($)": "TV",
        "Radio Ad Budget ($)": "Radio",
        "Newspaper Ad Budget ($)": "Newspaper",
        "Sales ($)": "Sales",
    }

    df = df.rename(columns=rename_map)

    required = {"TV", "Radio", "Newspaper", "Sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ada: {missing}\n"
            f"Kolom yang tersedia: {list(df.columns)}"
        )


    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).reset_index(drop=True)

    return df


def main():
    df = load_dataset(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_names": FEATURES,
            "target_name": TARGET,
            "metrics": {"rmse": float(rmse), "r2": float(r2)},
            "channel_mapping": {
                "TV": "Meta Ads Spend",
                "Radio": "Google Ads Spend",
                "Newspaper": "TikTok Ads Spend",
            },
        },
        MODEL_PATH,
    )

    print("Training selesai")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")
    print(f"Model tersimpan: {MODEL_PATH}")


if __name__ == "__main__":
    main()
