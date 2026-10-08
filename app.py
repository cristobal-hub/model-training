from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "crop_yield_model_final.joblib"
UNITS = {
    "irrigation_mm": "mm",
    "days_from_last_harvest": "days",
    "nitrogen_kg_ha": "kg/ha",
    "rainfall_mm": "mm",
    "avg_temp_c": "°C",
    "humidity_pct": "%",
}
NON_NEGATIVE_FEATURES = {
    "irrigation_mm",
    "days_from_last_harvest",
    "nitrogen_kg_ha",
    "rainfall_mm",
    "humidity_pct",
    "pest_index",
}

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌱")
st.title("Crop Yield Predictor")
st.write("Enter crop and growing-condition details to estimate crop yield.")

if not MODEL_PATH.is_file():
    st.error(
        f"Final model not found: {MODEL_PATH}. Run train.py, test.py, "
        "and finalize.py before launching the app."
    )
    st.stop()


@st.cache_resource
def load_saved_model(path: Path):
    return joblib.load(path)


saved = load_saved_model(MODEL_PATH)
feature_columns = saved["feature_columns"]
categorical_columns = saved.get("categorical_columns", [])
category_values = saved.get("categorical_values", {})
st.caption(f"Prediction model: {saved['model_name']}")

with st.form("crop-inputs"):
    st.subheader("Growing conditions")
    left, right = st.columns(2)
    values = {}
    for index, column in enumerate(feature_columns):
        container = left if index % 2 == 0 else right
        label = column.replace("_", " ").title()
        with container:
            if column in categorical_columns:
                categories = category_values.get(column)
                if not categories:
                    st.error(f"No saved category options are available for {column}.")
                    st.stop()
                values[column] = st.selectbox(label, categories)
            else:
                unit = UNITS.get(column)
                field_label = f"{label} ({unit})" if unit else label
                minimum = 0.0 if column in NON_NEGATIVE_FEATURES else None
                values[column] = st.number_input(
                    field_label,
                    min_value=minimum,
                    value=None,
                    step=0.1,
                    format="%.2f",
                )

    submitted = st.form_submit_button("Predict crop yield", type="primary")

if submitted:
    missing_values = [column for column, value in values.items() if value is None]
    if missing_values:
        st.error(f"Enter a value for each numeric input: {', '.join(missing_values)}")
        st.stop()

    input_features = pd.DataFrame([values], columns=feature_columns)
    model_features = input_features.copy()
    for column in categorical_columns:
        model_features[column] = model_features[column].map(
            {category: code for code, category in enumerate(category_values[column])}
        )
    prediction = float(saved["model"].predict(model_features)[0])
    st.success(f"Estimated crop yield: **{prediction:.3f} tons**")

    results = input_features.copy()
    results["predicted_yield_tons"] = prediction
    st.download_button(
        "Download prediction as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="crop_yield_prediction.csv",
        mime="text/csv",
    )
