from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "best_crop_yield_model.joblib"
TEST_SET_PATH = PROJECT_DIR / "crop_yield_test_set.csv"
PREDICTIONS_PATH = PROJECT_DIR / "crop_yield_test_predictions.csv"
TARGET_COLUMN = "crop_yield_tons"

for required_path in (MODEL_PATH, TEST_SET_PATH):
    if not required_path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {required_path}. Run train.py first."
        )

saved = joblib.load(MODEL_PATH)
model = saved["model"]
feature_columns = saved["feature_columns"]
test_set = pd.read_csv(TEST_SET_PATH)
missing_columns = [column for column in feature_columns if column not in test_set.columns]
if missing_columns:
    raise ValueError(f"Test set is missing feature columns: {missing_columns}")

X_test = test_set[feature_columns]
y_test = test_set[TARGET_COLUMN]
predictions = model.predict(X_test)

results = pd.DataFrame({
    "actual_yield_tons": y_test,
    "predicted_yield_tons": predictions,
    "prediction_error_tons": y_test - predictions,
})
results.to_csv(PREDICTIONS_PATH, index=False)

plot_min = min(y_test.min(), predictions.min())
plot_max = max(y_test.max(), predictions.max())
figure, axis = plt.subplots(figsize=(7, 6))
axis.scatter(y_test, predictions, alpha=0.25, s=12, color="#3977a8")
axis.plot([plot_min, plot_max], [plot_min, plot_max], "--", color="#c24e35")
axis.set_xlabel("Actual crop yield (tons)")
axis.set_ylabel("Predicted crop yield (tons)")
axis.set_title("Held-out test: actual vs. predicted")
axis.grid(alpha=0.2)
figure.tight_layout()

residuals = y_test - predictions
figure, axis = plt.subplots(figsize=(8, 5))
axis.hist(residuals, bins=40, color="#3977a8", edgecolor="white")
axis.axvline(0, color="#c24e35", linestyle="--", linewidth=1.5)
axis.set_xlabel("Prediction error (actual - predicted, tons)")
axis.set_ylabel("Number of test rows")
axis.set_title("Held-out test: residual distribution")
axis.grid(axis="y", alpha=0.2)
figure.tight_layout()

print("Model:", saved.get("model_name", type(model).__name__))
print("Test rows:", len(y_test))
print(f"R2: {r2_score(y_test, predictions):.4f}")
print(f"MAE: {mean_absolute_error(y_test, predictions):.4f} tons")
print(f"RMSE: {mean_squared_error(y_test, predictions) ** 0.5:.4f} tons")
print("Saved row-by-row predictions:", PREDICTIONS_PATH)
print("Displaying evaluation charts.")
plt.show()