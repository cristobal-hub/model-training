import json
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATASET_PATH = PROJECT_DIR / "crop_yield_data.csv"
ARCHIVED_DATASET_PATH = Path(r"C:\Users\27637\Downloads\r")
LEGACY_RAW_DATASET_PATH = Path(r"C:\Users\27637\Downloads\our crop dataset\crop_yield_data.csv")
OUTPUT_PATH = PROJECT_DIR / "crop_yield_cleaned.csv"
CATEGORY_CODES_PATH = PROJECT_DIR / "crop_yield_category_codes.json"
CATEGORICAL_COLUMNS = (
    "weather_zone",
    "irrigation_method",
    "seed_quality",
    "soil_type",
    "fertilized",
    "season",
    "crop_type",
)
SPARSE_COLUMNS = ("soil_ph", "phosphorus_kg_ha", "potassium_kg_ha")
NON_NEGATIVE_COLUMNS = (
    "irrigation_mm",
    "days_from_last_harvest",
    "nitrogen_kg_ha",
    "rainfall_mm",
    "humidity_pct",
    "pest_index",
    "crop_yield_tons",
)
NUMERIC_COLUMNS = (
    "irrigation_mm",
    "days_from_last_harvest",
    "nitrogen_kg_ha",
    "rainfall_mm",
    "avg_temp_c",
    "humidity_pct",
    "pest_index",
    "crop_yield_tons",
)


def main():
    input_path = None
    for candidate in (
        RAW_DATASET_PATH,
        ARCHIVED_DATASET_PATH,
        LEGACY_RAW_DATASET_PATH,
        OUTPUT_PATH,
    ):
        if candidate.is_file():
            input_path = candidate
            break

    if input_path is None:
        raise FileNotFoundError(
            "Dataset not found. Checked the project raw dataset at "
            f"{RAW_DATASET_PATH}, archived dataset at {ARCHIVED_DATASET_PATH}, "
            f"legacy dataset at {LEGACY_RAW_DATASET_PATH}, and cleaned dataset "
            f"at {OUTPUT_PATH}."
        )

    df = pd.read_csv(input_path)
    print("Loaded dataset from:", input_path)

    initial_rows, initial_cols = df.shape
    print("Initial shape:", df.shape)
    print("Duplicate rows:", df.duplicated().sum())
    print("Missing values per column:\n", df.isna().sum())
    print("Missing values total:", df.isna().sum().sum())

    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\s+', '_', regex=True)
    )

    df = df.drop_duplicates()
    print("Rows after dropping duplicates:", df.shape[0])

    columns_to_drop = [column for column in SPARSE_COLUMNS if column in df.columns]
    df = df.drop(columns=columns_to_drop)
    print("Dropped mostly-missing columns:", columns_to_drop)

    repaired_numeric_values = 0
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            values = df[column].astype("string").str.strip()
            bracket_artifacts = values.str.match(r"^\]\s*-?\d", na=False)
            repaired_numeric_values += int(bracket_artifacts.sum())
            values = values.str.replace(r"^\]\s*(?=-?\d)", "", regex=True)
            df[column] = pd.to_numeric(values, errors="coerce")
    print("Removed stray leading brackets from numeric values:", repaired_numeric_values)

    df = df.dropna(axis=0)
    print("Rows after dropping missing rows:", df.shape[0])

    non_negative_columns = [
        column for column in NON_NEGATIVE_COLUMNS if column in df.columns
    ]
    invalid_rows = df[non_negative_columns].lt(0).any(axis=1)
    print("Rows with negative nonnegative measurements:", int(invalid_rows.sum()))
    df = df.loc[~invalid_rows]

    category_codes = {}
    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            categories = sorted(df[column].astype(str).unique().tolist())
            category_codes[column] = categories
            df[column] = df[column].astype("category").cat.set_categories(
                categories
            ).cat.codes

    df.to_csv(OUTPUT_PATH, index=False)
    CATEGORY_CODES_PATH.write_text(
        json.dumps(category_codes, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFinal data quality summary:")
    print("Rows before cleaning:", initial_rows)
    print("Rows after cleaning:", df.shape[0])
    print("Columns before cleaning:", initial_cols)
    print("Columns after cleaning:", df.shape[1])
    print("Encoded columns:", list(df.columns))
    print("Category codes:", category_codes)
    print("\nFinal cleaned shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    main()