from pathlib import Path

import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "crop_yield_cleaned.csv"
CATEGORY_CODES_PATH = PROJECT_DIR / "crop_yield_category_codes.json"
MODEL_PATH = PROJECT_DIR / "best_crop_yield_model.joblib"
TEST_SET_PATH = PROJECT_DIR / "crop_yield_test_set.csv"
COMPARISON_PATH = PROJECT_DIR / "model_comparison.csv"
COMPARISON_PLOT_PATH = PROJECT_DIR / "model_comparison_validation.png"
TARGET_COLUMN = "crop_yield_tons"
CV_FOLDS = 5

if not DATASET_PATH.is_file():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATASET_PATH}. Run clean.py first."
    )

df = pd.read_csv(DATASET_PATH)
category_codes = json.loads(CATEGORY_CODES_PATH.read_text(encoding="utf-8"))
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column {TARGET_COLUMN!r} is missing from the dataset.")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]
feature_columns = list(X.columns)
categorical_columns = [
    column for column in category_codes if column in feature_columns
]
preprocessor = ColumnTransformer(
    [
        (
            "categorical",
            OneHotEncoder(
                categories=[
                    np.arange(len(category_codes[column]))
                    for column in categorical_columns
                ],
                drop="first",
                handle_unknown="error",
                sparse_output=False,
            ),
            categorical_columns,
        ),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
test_set = X_test.copy()
test_set[TARGET_COLUMN] = y_test
test_set.to_csv(TEST_SET_PATH, index=False)

candidate_factories = {
    "Linear Regression": LinearRegression,
    "Gradient Boosting": lambda: HistGradientBoostingRegressor(
        max_iter=100, max_leaf_nodes=31, random_state=42
    ),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=30, max_depth=12, min_samples_leaf=5,
        max_samples=0.2, random_state=42, n_jobs=1
    ),
    "Decision Tree": lambda: DecisionTreeRegressor(random_state=42),
}

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=43)
validation_results = []
best_model = None
best_model_name = None
best_validation_r2 = float("-inf")
for index, (model_name, create_model) in enumerate(candidate_factories.items(), start=1):
    print(
        f"[{index}/{len(candidate_factories)}] Running {CV_FOLDS}-fold CV for "
        f"{model_name}...",
        flush=True,
    )
    candidate = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("regressor", create_model()),
    ])
    scores = {
        "train_r2": [],
        "test_r2": [],
        "train_mae": [],
        "test_mae": [],
        "train_rmse": [],
        "test_rmse": [],
    }
    for train_indices, validation_indices in cv.split(X_train, y_train):
        fold_model = clone(candidate)
        X_fold_train = X_train.iloc[train_indices]
        X_fold_validation = X_train.iloc[validation_indices]
        y_fold_train = y_train.iloc[train_indices]
        y_fold_validation = y_train.iloc[validation_indices]
        fold_model.fit(X_fold_train, y_fold_train)
        train_predictions = fold_model.predict(X_fold_train)
        validation_predictions = fold_model.predict(X_fold_validation)
        scores["train_r2"].append(r2_score(y_fold_train, train_predictions))
        scores["test_r2"].append(
            r2_score(y_fold_validation, validation_predictions)
        )
        scores["train_mae"].append(
            mean_absolute_error(y_fold_train, train_predictions)
        )
        scores["test_mae"].append(
            mean_absolute_error(y_fold_validation, validation_predictions)
        )
        scores["train_rmse"].append(
            np.sqrt(mean_squared_error(y_fold_train, train_predictions))
        )
        scores["test_rmse"].append(
            np.sqrt(mean_squared_error(y_fold_validation, validation_predictions))
        )
    mean_cv_r2 = np.mean(scores["test_r2"])
    model_result = {
        "model": model_name,
        "train_r2_mean": np.mean(scores["train_r2"]),
        "cv_r2_mean": mean_cv_r2,
        "cv_r2_std": np.std(scores["test_r2"], ddof=1),
        "r2_gap_mean": np.mean(scores["train_r2"]) - mean_cv_r2,
        "train_mae_mean": np.mean(scores["train_mae"]),
        "cv_mae_mean": np.mean(scores["test_mae"]),
        "cv_mae_std": np.std(scores["test_mae"], ddof=1),
        "train_rmse_mean": np.mean(scores["train_rmse"]),
        "cv_rmse_mean": np.mean(scores["test_rmse"]),
        "cv_rmse_std": np.std(scores["test_rmse"], ddof=1),
    }
    validation_results.append(model_result)
    print(
        f"    Mean train R2: {model_result['train_r2_mean']:.4f}\n"
        f"    CV R2: {mean_cv_r2:.4f} +/- {model_result['cv_r2_std']:.4f}, "
        f"MAE: {model_result['cv_mae_mean']:.4f} +/- "
        f"{model_result['cv_mae_std']:.4f}, RMSE: "
        f"{model_result['cv_rmse_mean']:.4f} +/- "
        f"{model_result['cv_rmse_std']:.4f}",
        flush=True,
    )
    if model_name != "Linear Regression" and mean_cv_r2 > best_validation_r2:
        best_model = create_model
        best_model_name = model_name
        best_validation_r2 = mean_cv_r2

comparison = pd.DataFrame(validation_results).sort_values(
    "cv_r2_mean", ascending=False
)
comparison.to_csv(COMPARISON_PATH, index=False)

figure, axis = plt.subplots(figsize=(9, 5))
positions = range(len(comparison))
bar_height = 0.38
axis.barh(
    [position - bar_height / 2 for position in positions],
    comparison["train_r2_mean"],
    height=bar_height,
    label="Mean training R2",
    color="#7ba6c9",
)
axis.barh(
    [position + bar_height / 2 for position in positions],
    comparison["cv_r2_mean"],
    height=bar_height,
    xerr=comparison["cv_r2_std"],
    label=f"{CV_FOLDS}-fold CV mean R2 (+/- 1 SD)",
    color="#3977a8",
)
axis.set_yticks(list(positions), comparison["model"])
axis.invert_yaxis()
axis.set_xlabel("R-squared")
axis.set_title(f"Training vs. {CV_FOLDS}-fold cross-validation comparison")
axis.legend()
axis.grid(axis="x", alpha=0.25)
figure.tight_layout()
figure.savefig(COMPARISON_PLOT_PATH, dpi=160)
plt.close(figure)

model = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("regressor", best_model()),
])
model.fit(X_train, y_train)

joblib.dump({
    "model": model,
    "feature_columns": feature_columns,
    "categorical_columns": categorical_columns,
    "categorical_values": category_codes,
    "model_name": best_model_name,
    "validation_results": validation_results,
    "cross_validation_results": validation_results,
    "cv_folds": CV_FOLDS,
    "cv_shuffle_seed": 43,
}, MODEL_PATH)

print(f"{CV_FOLDS}-fold comparison (higher R2, lower MAE/RMSE are better):")
print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print("Selected model:", best_model_name)
print("Cross-validation rows per fold:", f"approximately {len(X_train) * 4 // 5}")
print("Selected model refit rows (80% development data):", len(X_train))
print("Reserved test rows (20%):", len(X_test))
print("Saved model:", MODEL_PATH)
print("Saved holdout set:", TEST_SET_PATH)
print("Saved validation metrics:", COMPARISON_PATH)
print("Saved validation comparison chart:", COMPARISON_PLOT_PATH)
