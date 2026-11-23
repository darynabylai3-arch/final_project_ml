# Heart Disease Prediction: final project

Binary classification of heart disease (`HeartDisease` = 1 if `num` > 0) on the UCI Heart Disease data (920 patients).

## Structure

- `data/` - raw dataset `heart_disease_uci.csv`
- `eda/` - EDA and preprocessing notebook (Week 2)
- `src/` - shared code: cleaning, split, preprocessing pipeline, evaluation
- `models/logistic_regression/`, `models/random_forest/`, `models/catboost/` - one folder per model with its notebook, `metrics.json` and fitted `model.joblib`
- `models/comparison.ipynb` - side-by-side comparison of all models
- `reports/` - PDF exports

## Run order

1. `eda/16_Week2_EDA_Preprocessing.ipynb`
2. each notebook in `models/*/model.ipynb`
3. `models/comparison.ipynb`
