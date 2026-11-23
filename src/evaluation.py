import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from preprocessing import RANDOM_STATE, load_clean, make_preprocessor, split_data


def evaluate(name, estimator, scale, out_dir):
    df = load_clean()
    X_train, X_test, y_train, y_test = split_data(df)

    pipe = Pipeline([("pre", make_preprocessor(scale)), ("clf", estimator)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cvres = cross_validate(pipe, X_train, y_train, cv=cv,
                           scoring=["accuracy", "f1", "roc_auc"])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "cv_roc_auc_mean": float(cvres["test_roc_auc"].mean()),
        "cv_roc_auc_std": float(cvres["test_roc_auc"].std()),
        "cv_f1_mean": float(cvres["test_f1"].mean()),
        "cv_accuracy_mean": float(cvres["test_accuracy"].mean()),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_f1": float(f1_score(y_test, pred)),
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(pipe, out_dir / "model.joblib")
    return metrics
