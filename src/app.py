from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


# Rutas, objetivo y variables elegidas en el notebook para mantener coherencia.
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RAW_PATH = RAW_DIR / "demographic_health_data.csv"
DATA_URL = "https://breathecode.herokuapp.com/asset/internal-link?id=733&path=demographic_health_data.csv"
TARGET = "diabetes_prevalence"
FEATURES = [
    "Percent of adults with less than a high school diploma 2014-18",
    "Percent of adults with a high school diploma only 2014-18",
    "Percent of adults completing some college or associate's degree 2014-18",
    "Percent of adults with a bachelor's degree or higher 2014-18",
    "PCTPOVALL_2018",
    "Unemployment_rate_2018",
    "Median_Household_Income_2018",
    "% Black-alone",
    "% White-alone",
    "Percent of Population Aged 60+",
    "Active Physicians per 100000 Population 2018 (AAMC)",
    "Active Primary Care Physicians per 100000 Population 2018 (AAMC)",
    "Total nurse practitioners (2019)",
    "Total Hospitals (2019)",
    "ICU Beds_x",
    "Urban_rural_code",
]


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_dataset() -> pd.DataFrame:
    """Carga el dataset desde data/raw o lo descarga si aun no existe localmente."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists():
        return pd.read_csv(RAW_PATH)

    data = pd.read_csv(DATA_URL)
    data.to_csv(RAW_PATH, index=False)
    return data


def prepare_model_data(data: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo las variables seleccionadas y la variable objetivo."""
    return data[FEATURES + [TARGET]].copy()


def split_and_scale(model_data: pd.DataFrame):
    """Divide el dataset en train/test y escala las variables predictoras."""
    X = model_data.drop(columns=[TARGET])
    y = model_data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled


def evaluate_regression(y_true: pd.Series, predictions: np.ndarray) -> dict:
    """Resume el rendimiento del modelo con MAE, RMSE y R2."""
    return {
        "mae": round(mean_absolute_error(y_true, predictions), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_true, predictions)), 4),
        "r2": round(r2_score(y_true, predictions), 4),
    }


def save_processed_data(model_data, X_train, X_test, y_train, y_test, comparison):
    """Guarda el dataset usado, los subconjuntos train/test y la comparacion final."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    model_data.to_csv(PROCESSED_DIR / "demographic_health_model_data.csv", index=False)
    pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1).to_csv(
        PROCESSED_DIR / "demographic_health_train.csv", index=False
    )
    pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv(
        PROCESSED_DIR / "demographic_health_test.csv", index=False
    )
    comparison.to_csv(PROCESSED_DIR / "regression_model_comparison.csv", index=False)


def main():
    """Ejecuta el flujo principal de preparacion, modelado y guardado de resultados."""
    total_data = load_dataset()
    model_data = prepare_model_data(total_data)

    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = split_and_scale(model_data)

    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    linear_metrics = evaluate_regression(y_test, linear_model.predict(X_test_scaled))

    lasso_model = Lasso(alpha=0.1, max_iter=20000)
    lasso_model.fit(X_train_scaled, y_train)
    lasso_metrics = evaluate_regression(y_test, lasso_model.predict(X_test_scaled))

    grid_search = GridSearchCV(
        estimator=Lasso(max_iter=20000),
        param_grid={"alpha": [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5]},
        scoring="r2",
        cv=5,
    )
    grid_search.fit(X_train_scaled, y_train)
    optimized_model = grid_search.best_estimator_
    optimized_metrics = evaluate_regression(y_test, optimized_model.predict(X_test_scaled))

    comparison = pd.DataFrame(
        [
            {"model": "linear_regression", **linear_metrics},
            {"model": "lasso_alpha_0.1", **lasso_metrics},
            {"model": f"lasso_optimized_alpha_{grid_search.best_params_['alpha']}", **optimized_metrics},
        ]
    )

    save_processed_data(model_data, X_train, X_test, y_train, y_test, comparison)
    print(comparison)


if __name__ == "__main__":
    main()
