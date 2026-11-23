from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart_disease_uci.csv"

NUM = ["age", "trestbps", "chol", "thalch", "oldpeak", "HR_reserve"]
BIN = ["sex", "exang", "fbs"]
CAT = ["cp", "restecg", "slope"]
FEATURES = NUM + BIN + CAT


def load_clean():
    raw = pd.read_csv(DATA_PATH)
    df = raw[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
              "thalch", "exang", "oldpeak", "slope", "num"]].copy()
    df["HeartDisease"] = (df["num"] > 0).astype(int)
    df["chol"] = df["chol"].replace(0, np.nan)
    df["trestbps"] = df["trestbps"].replace(0, np.nan)
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
    df["exang"] = df["exang"].map({True: 1, False: 0, "True": 1, "False": 0})
    df["fbs"] = df["fbs"].map({True: 1, False: 0, "True": 1, "False": 0})
    for c in CAT:
        df[c] = df[c].fillna("Unknown")
    df["HR_reserve"] = (220 - df["age"]) - df["thalch"]
    return df[FEATURES + ["HeartDisease"]]


def split_data(df, test_size=0.2, random_state=RANDOM_STATE):
    X = df[FEATURES]
    y = df["HeartDisease"]
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def make_preprocessor(scale):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM),
        ("bin", SimpleImputer(strategy="most_frequent"), BIN),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
