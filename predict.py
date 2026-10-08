import argparse
from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "crop_yield_model_final.joblib"

parser = argparse.ArgumentParser(description="Predict crop yield manually or from a CSV.")
parser.add_argument(
    "--input-csv",
    type=Path,
    help="CSV file with crop feature values (omit to enter values manually)",
)
parser.add_argument(
    "--output",
    type=Path,
    help="Output CSV path (default depends on prediction mode)",
)
args = parser.parse_args()

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Final model not found: {MODEL_PATH}. Run train.py, test.py, "
        "then finalize.py first."
    )

saved = joblib.load(MODEL_PATH)
feature_columns = saved["feature_columns"]
categorical_columns = saved.get("categorical_columns", [])
if args.input_csv is not None:
    if not args.input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    input_data = pd.read_csv(args.input_csv)
    missing_columns = [column for column in feature_columns if column not in input_data.columns]
    if missing_columns:
        raise ValueError(f"Input CSV is missing feature columns: {missing_columns}")
    features = input_data.loc[:, feature_columns]
    for column in categorical_columns:
        categories = saved["categorical_values"][column]
        category_codes = {category: code for code, category in enumerate(categories)}
        values = features[column]
        numeric_values = pd.to_numeric(values, errors="coerce")
        text_values = values.astype("string").str.strip().map(category_codes)
        use_text = values.notna() & numeric_values.isna()
        invalid_text = use_text & text_values.isna()
        if invalid_text.any():
            unknown_values = values.loc[invalid_text].unique().tolist()
            raise ValueError(
                f"Unknown values for {column}: {unknown_values}. "
                f"Expected one of {categories} or a numeric category code."
            )
        features[column] = numeric_values.where(~use_text, text_values)
        if features[column].isna().any():
            raise ValueError(f"Input CSV has missing values in {column}.")
        allowed_codes = set(range(len(categories)))
        invalid_codes = ~features[column].isin(allowed_codes)
        if invalid_codes.any():
            raise ValueError(
                f"Invalid category code(s) for {column}: "
                f"{features.loc[invalid_codes, column].unique().tolist()}"
            )
    output_path = args.output or args.input_csv.with_name(
        f"{args.input_csv.stem}_predictions.csv"
    )
    predictions = saved["model"].predict(features)
    results = input_data.copy()
    results["predicted_yield_tons"] = predictions
    results.to_csv(output_path, index=False)
    print("Predictions:")
    print(results["predicted_yield_tons"].head(5).to_string(index=False))
    if len(results) > 5:
        print(f"... {len(results) - 5} more predictions in the CSV.")
    print("Saved predictions:", output_path)
else:
    category_values = saved.get("categorical_values", {})
    print("Enter a category name or its numeric code as shown in the prompt.")
    values = {}
    for column in feature_columns:
        categories = category_values.get(column)
        options = (
            ", ".join(f"{index}={value}" for index, value in enumerate(categories))
            if categories
            else None
        )
        prompt = f"{column} ({options}): " if options else f"{column}: "
        raw_value = input(prompt).strip()
        if column in categorical_columns:
            if raw_value in categories:
                values[column] = categories.index(raw_value)
            else:
                try:
                    category_code = int(raw_value)
                except ValueError as error:
                    raise ValueError(
                        f"Unknown value for {column}: {raw_value!r}. "
                        f"Expected one of {categories} or a numeric code from "
                        f"0 to {len(categories) - 1}."
                    ) from error
                if not 0 <= category_code < len(categories):
                    raise ValueError(
                        f"Invalid category code for {column}: {category_code}. "
                        f"Expected a number from 0 to {len(categories) - 1}."
                    )
                values[column] = category_code
        else:
            try:
                values[column] = float(raw_value)
            except ValueError as error:
                raise ValueError(
                    f"{column} must be a number; received {raw_value!r}."
                ) from error

    features = pd.DataFrame([values], columns=feature_columns)
    prediction = saved["model"].predict(features)[0]
    results = features.copy()
    results["predicted_yield_tons"] = prediction
    output_path = args.output or PROJECT_DIR / "manual_prediction.csv"
    results.to_csv(output_path, index=False)
    print(f"Predicted crop yield: {prediction:.3f} tons")
    print("Saved input and prediction:", output_path)