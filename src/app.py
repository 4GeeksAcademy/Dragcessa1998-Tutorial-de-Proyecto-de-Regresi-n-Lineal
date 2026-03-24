from pathlib import Path
import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "model": name,
        "r2": r2_score(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "mae": mean_absolute_error(y_test, predictions),
    }


project_dir = Path(__file__).resolve().parent.parent
dataset_path = project_dir / "data" / "raw" / "medical_insurance_cost.csv"

df = pd.read_csv(dataset_path)

# Para el modelo base me quedo con las variables que mejor explican el coste.
selected_features = ["age", "bmi", "children", "smoker"]
X = df[selected_features].copy()

# En la parte optimizada convierto smoker a binaria para poder generar
# interacciones polinómicas de forma más cómoda.
X_optimized = X.copy()
X_optimized["smoker_numeric"] = (X_optimized["smoker"] == "yes").astype(int)
X_optimized = X_optimized.drop(columns=["smoker"])
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(
    X_optimized, y, test_size=0.2, random_state=42
)

categorical_features = ["smoker"]
numeric_features = ["age", "bmi", "children"]

baseline_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

baseline_model = Pipeline(
    steps=[
        ("preprocessor", baseline_preprocessor),
        ("model", LinearRegression()),
    ]
)

optimized_pipeline = Pipeline(
    steps=[
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", Ridge()),
    ]
)

search = GridSearchCV(
    estimator=optimized_pipeline,
    param_grid={
        "poly__degree": [1, 2, 3],
        "model__alpha": [0.01, 0.1, 1, 10, 100],
    },
    cv=5,
    scoring="r2",
)

baseline_results = evaluate_model(
    "LinearRegression", baseline_model, X_train, X_test, y_train, y_test
)
optimized_results = evaluate_model(
    "Tuned Ridge", search, X_train_opt, X_test_opt, y_train_opt, y_test_opt
)

results = pd.DataFrame([baseline_results, optimized_results]).sort_values(
    by="r2", ascending=False
)

print("Model comparison:")
print(results.to_string(index=False))

print("\nBest hyperparameters:")
print(search.best_params_)
