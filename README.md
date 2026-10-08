
# Crop Yield Prediction

## Project overview

This project trains regression models to estimate crop yield from crop and farming inputs. The target is `crop_yield_tons`; model predictions are in tons.

The workflow cleans the raw dataset into a numeric 15-column CSV with categorical values represented by documented integer codes. Model pipelines one-hot encode those categorical codes for comparison, evaluation, and prediction.

The workflow reserves a deterministic 20% test partition and uses shuffled five-fold cross-validation on the remaining 80% for model comparison. The test partition is kept out of model selection.

## Dataset and model inputs

`crop_yield_data.csv` is the raw, unencoded dataset copied from `C:\Users\27637\Downloads\archive\CROP_DATASET.csv`. It contains 247,321 rows and 18 columns, including categorical columns. The project also includes:

- `crop_yield_cleaned.csv` — cleaned numeric dataset used by training; regenerated from the raw CSV when `clean.py` runs. It has 247,298 rows and exactly 15 columns.
- `crop_yield_category_codes.json` — mapping from categorical labels in the raw file to their integer codes in the cleaned CSV.

The target is `crop_yield_tons`. The archived data contains weather zone, irrigation method and amount, soil measurements and type, harvest timing, seed quality, fertilizer and nutrient values, season, rainfall, temperature, humidity, pest index, crop type, and yield. `clean.py` drops the highly incomplete `soil_ph`, `phosphorus_kg_ha`, and `potassium_kg_ha` columns, removes duplicate rows and rows missing any retained values, rejects negative values in nonnegative measurements and yield (negative temperatures remain valid), and encodes each categorical column as one numeric code. This preserves exactly 15 columns after the three sparse columns are removed. Code-to-category mappings are saved in `crop_yield_category_codes.json`. Models one-hot encode those category-code features internally so they are not treated as ordered numeric values. The raw `crop_yield_data.csv` remains unencoded.

## Python and package setup

Install Python 3.11 or newer from [python.org/downloads](https://www.python.org/downloads/) if Python is not already installed. On Windows, the Python launcher `py` is useful for selecting a specific installed version.

Open PowerShell in this project folder. For a fresh setup, create and activate a virtual environment, then install the listed packages:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, allow it for this terminal only, then activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

The project already contains a `.venv` in this workspace. If it exists, activate it and install/refresh dependencies with:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The dependency manifest is `requirements.txt`. It installs:

| Package | Use |
| --- | --- |
| `pandas` | Loading, cleaning, and writing CSV data |
| `numpy` | Numeric operations used by the notebook and scientific Python stack |
| `scikit-learn` | Regression models, data splitting, and evaluation metrics |
| `joblib` | Saving and loading trained models |
| `matplotlib` | Saving the training comparison chart and displaying test charts |
| `jupyter` | Running the interactive notebook |
| `ipykernel` | Registering/selecting the virtual environment as a notebook kernel |

Installing from `requirements.txt` downloads the packages from the configured Python package index. The raw `crop_yield_data.csv` is included in the project. If it is missing, `clean.py` falls back to the archived file at `C:\Users\27637\Downloads\archive\CROP_DATASET.csv`, then a legacy machine-specific path, then `crop_yield_cleaned.csv`.

## Run the workflow

Run these commands in PowerShell from the project folder with `.venv` activated.

### 1. Clean the data

```powershell
python clean.py
```

`clean.py` reads the raw `crop_yield_data.csv` first (or the documented fallback sources), normalizes column names, removes the three highly incomplete measurement columns, converts measurement columns to numeric, removes duplicate rows and rows missing any retained values, drops rows with negative values in nonnegative measurements or yield, and replaces category labels with integer codes. It writes the numeric `crop_yield_cleaned.csv` with 247,298 rows and exactly 15 columns, plus `crop_yield_category_codes.json`. The script does not modify the raw input file or cap the row count. **The cleaned CSV is overwritten when this script runs.**

### 2. Compare models and train the selected model

```powershell
python train.py
```

The script:

1. Reserves 20% of the data as the final test set (`random_state=42`).
2. Runs shuffled five-fold cross-validation on the remaining 80% development partition (`random_state=43`), fitting each candidate on four folds and scoring it on the fifth.
3. Compares Linear Regression, Gradient Boosting, Random Forest, and Decision Tree using the same folds. Linear Regression is reported but remains ineligible for selection. Model settings are specified in `train.py`.
4. Reports mean and standard deviation across folds for R-squared, MAE, and RMSE, along with mean training R-squared.
5. Selects the eligible model with the highest mean cross-validation R-squared and refits it on the full 80% development partition.

The separate 20% test partition remains untouched during cross-validation and model selection and is used only by `test.py`. Elapsed time is not measured or used as an evaluation metric.

The current candidate configurations are:

| Candidate | Configuration |
| --- | --- |
| Linear Regression | Scikit-learn defaults. |
| Gradient Boosting | Histogram-based gradient boosting, 100 boosting iterations, maximum 31 leaf nodes, random seed 42. |
| Random Forest | 30 trees, maximum depth 12, minimum 5 samples per leaf, 20% maximum sample fraction, random seed 42, one worker. |
| Decision Tree | Scikit-learn defaults with random seed 42. |

Candidates are ranked by mean cross-validation R-squared (higher is better). `model_comparison.csv` includes mean and standard deviation of fold R-squared, MAE, and RMSE, plus mean training scores. It contains no timing measurements.

### 3. Evaluate on the held-out test set

```powershell
python test.py
```

This reports R-squared, MAE, and RMSE on the untouched 20% test set, writes row-by-row predictions to `crop_yield_test_predictions.csv`, and displays actual-versus-predicted and residual charts. It does not save those two test charts as PNG files.

### 4. Refit a final prediction model

After reviewing the test metrics, run:

```powershell
python finalize.py
```

This refits the selected estimator on all cleaned labeled rows and saves `crop_yield_model_final.joblib`. The holdout score estimates performance before this final refit. Do not report predictions from the all-data model on its training rows as an independent test score.

### 5. Predict manually

```powershell
python predict.py
```

Enter numeric measurements and choose category labels from the prompts. Category labels are converted to the matching saved codes before prediction. Batch input can use either raw category labels or the numeric codes from the cleaned CSV. The script writes the inputs and prediction to `manual_prediction.csv` by default. Use `--output` to choose a different destination:

```powershell
python predict.py --output .\my_manual_prediction.csv
```

### 6. Use the browser prediction interface

Install the project dependencies if needed, then launch the local web app:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens in a browser. Choose the crop categories, enter numeric growing conditions, and select **Predict crop yield**. Use **Download prediction as CSV** to save the entered features and prediction.

### 7. Predict from an input CSV (optional)

Prepare a CSV containing all 15 model feature columns. A target column is not needed; any additional input columns are retained in the output.

```powershell
python predict.py --input-csv .\my_crop_inputs.csv
```

By default, predictions are saved beside the input as `my_crop_inputs_predictions.csv`. Set a custom output path with:

```powershell
python predict.py --input-csv .\my_crop_inputs.csv --output .\my_predictions.csv
```

### Interactive notebook

Open `crop_yield_modeling.ipynb` in VS Code or Jupyter, select the project's `.venv` Python kernel, and run the notebook cells in order. It performs data inspection and basic cleanup, compares the four models using shuffled five-fold cross-validation, evaluates on the held-out test set, and displays plots.

## Model comparison and current results

The saved comparison is regenerated by `train.py`. The table reports the latest five-fold cross-validation averages. Linear Regression is included for comparison but is not eligible for selection; the eligible model with the highest mean cross-validation R-squared is selected:

| Model | CV mean R-squared (± SD) | CV mean MAE (tons) | CV mean RMSE (tons) |
| --- | ---: | ---: | ---: |
| Gradient Boosting | 0.8014 ± 0.0011 | 1.5585 | 1.9668 |
| Random Forest | 0.7255 ± 0.0041 | 1.8279 | 2.3122 |
| Decision Tree | 0.5225 ± 0.0072 | 2.4132 | 3.0494 |
| Linear Regression | 0.4196 ± 0.0020 | 2.6982 | 3.3622 |

Gradient Boosting is selected because it has the highest mean cross-validation R-squared among eligible models. Its current held-out test results are:

- Test rows: 49,460
- R-squared: 0.8037
- MAE: 1.5534 tons
- RMSE: 1.9625 tons

These metrics describe this dataset and this fixed split; they are not a guarantee of performance on other farms or future data. R-squared measures explained variation (higher is better). MAE is the average absolute prediction error; RMSE gives larger errors more weight. Both error values are measured in tons and lower is better.

## Imports and code dependencies

The scripts use standard-library `pathlib` and `time`, plus:

- `pandas` for tabular data.
- `joblib` for model serialization.
- `matplotlib` for charts.
- `streamlit` for the local browser prediction interface.
- `scikit-learn` Linear Regression, Random Forest, Histogram Gradient Boosting, and Decision Tree estimators.
- `scikit-learn` train/test splitting and R-squared, MAE, and RMSE metrics.
- The notebook also imports `numpy`.

The required third-party packages are listed in `requirements.txt`; install them as described above. No separate downloads are required for these imports beyond that package installation.

## Project files and generated outputs

### Source code and configuration

| File or folder | Purpose |
| --- | --- |
| `README.md` | Project overview, setup and package installation, run instructions, outputs, and presentation limitations. |
| `clean.py` | Cleans the raw dataset and writes a numeric 15-column CSV with categorical codes. |
| `train.py` | Compares four estimators with five-fold cross-validation, selects by mean CV R-squared, and saves the selected model and holdout split. |
| `test.py` | Reports held-out test metrics, writes test predictions, and displays evaluation charts. |
| `finalize.py` | Refits the selected model on all labeled data for production-style predictions. |
| `predict.py` | Runs interactive prediction by default, or batch prediction with `--input-csv`. |
| `crop_yield_modeling.ipynb` | Interactive analysis covering inspection, validation comparison, and test evaluation. |
| `requirements.txt` | Third-party Python packages needed by scripts, notebook, and browser app. |
| `app.py` | Local Streamlit browser interface for interactive crop-yield prediction. |
| `.vscode/settings.json` | Workspace Python interpreter and terminal activation settings; the interpreter path is specific to this machine. |
| `.gitignore` | Ignore rules for selected generated/dependency files. |
| `.env.example`, `.env` | Configuration files for a separate application; they are not used by the crop-yield scripts. Never share or publish secret values from `.env`. |

### Data and model artifacts

| File | Created/used by | Purpose |
| --- | --- | --- |
| `crop_yield_data.csv` | Supplied raw input, copied from the archive | Unencoded source data; not modified by cleaning. |
| `C:\Users\27637\Downloads\archive\CROP_DATASET.csv` | Supplied archive copy | Fallback raw source if the project CSV is unavailable. |
| `crop_yield_cleaned.csv` | `clean.py` | Clean 15-column numeric dataset generated from the raw input; overwritten by cleaning. |
| `crop_yield_category_codes.json` | `clean.py` | Categorical label-to-code mapping used by model training and prediction. |
| `crop_yield_test_set.csv` | `train.py` | Fixed 20% held-out test rows, including actual target values. |
| `model_comparison.csv` | `train.py` | Per-model mean training and cross-validation metrics; no timing measurements. |
| `best_crop_yield_model.joblib` | `train.py` | Validation-selected estimator, feature order, and validation comparison data. |
| `crop_yield_test_predictions.csv` | `test.py` | Actual yield, predicted yield, and prediction error for each held-out row. |
| `crop_yield_model_final.joblib` | `finalize.py` | Estimator refit on all labeled rows, used by `predict.py`. |
| `manual_prediction.csv` | `predict.py` | Most recent manually entered values and predicted yield. |
| `*_predictions.csv` | `predict.py --input-csv` | Input rows with a predicted-yield column. |

### Charts and local environment

- `model_comparison_validation.png` — training versus mean five-fold CV R-squared for the four compared estimators; regenerated by `train.py`.
- Test charts are displayed by `test.py` and are not saved as image files.
- `.venv/` — local virtual environment; recreate it from `requirements.txt` on another machine.
- `__pycache__/` — automatically generated Python bytecode cache.

Training, cleaning, and prediction scripts overwrite their corresponding generated outputs when rerun. Keep backups of any generated files that need to be retained.

## Presentation notes and limitations

- This is a supervised regression task: the model learns a numeric yield from labeled historical examples.
- The 20% test set is separated before cross-validation and is not used to select a candidate.
- Four configured models are compared using shuffled five-fold cross-validation; this is model comparison, not hyperparameter search.
- Cleaning removes duplicates and incomplete rows/columns but intentionally does not reject unusual ranges or outliers.
- Manual prediction accepts numeric input without physical-range or category-combination validation; prediction quality for unusual inputs is not established.
- `crop_yield_model_final.joblib` is trained on all cleaned labeled data, so its predictions should not be evaluated against those same training rows as if they were unseen.
- Package versions are not pinned in `requirements.txt`; recreating the environment later may install newer compatible releases.
