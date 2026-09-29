# Calories Burnt Prediction Model

An exploratory machine-learning project that estimates calories burned during an exercise session. The project trains an **XGBoost regressor** on physiological and activity measurements, then evaluates the model with mean absolute error (MAE).

> **Important:** This is an educational prediction project, not a medical, dietary, or fitness prescription. Calorie estimates can vary substantially between people and activities. Do not use its output as the sole basis for health decisions.

## Contents

- [Project overview](#project-overview)
- [Repository layout](#repository-layout)
- [Dataset](#dataset)
- [Approach](#approach)
- [Quick start](#quick-start)
- [Run the notebook](#run-the-notebook)
- [Results](#results)
- [Reproducibility and limitations](#reproducibility-and-limitations)
- [Troubleshooting](#troubleshooting)

## Project overview

The workflow is contained in a single Jupyter notebook, [`calories_burnt_prediction.ipynb`](calories_burnt_prediction.ipynb). It:

1. Loads the training data from `sample_data/train.csv`.
2. Inspects data shape, types, null values, and descriptive statistics.
3. Visualizes the sex count and distributions of the numeric variables.
4. Encodes `Sex` as a numeric feature (`male` → `1`, `female` → `0`) and plots a correlation heatmap.
5. Splits the data into training and test sets.
6. Fits an `xgboost.XGBRegressor`.
7. Predicts calories for the held-out test data and reports MAE.

## Repository layout

```text
.
├── calories_burnt_prediction.ipynb  # End-to-end exploration, training, and evaluation
├── sample_data/
│   └── train.csv                    # Bundled labeled exercise-session data
└── README.md
```

## Dataset

The bundled CSV has **750,000 rows** and nine columns. Each row represents an exercise-session observation.

| Column | Role | Description |
| --- | --- | --- |
| `id` | Identifier | Row identifier. It is excluded from model features. |
| `Sex` | Feature | Categorical sex value, encoded in the notebook as `male = 1` and `female = 0`. |
| `Age` | Feature | Participant age. |
| `Height` | Feature | Participant height. |
| `Weight` | Feature | Participant weight. |
| `Duration` | Feature | Exercise duration. |
| `Heart_Rate` | Feature | Recorded heart rate. |
| `Body_Temp` | Feature | Recorded body temperature. |
| `Calories` | Target | Calories burned; the value the model predicts. |

The notebook assumes the CSV is available at `./sample_data/train.csv`, relative to the repository root. Keep the header names unchanged if you replace the data. The notebook itself does not document units for the measurements, so confirm the source-data units before interpreting or reusing predictions.

## Approach

### Feature preparation

`Calories` is separated as the target, while `id` is removed to avoid treating a row identifier as a predictive signal. The remaining seven fields are used as features. Before splitting, the notebook converts the `Sex` labels to integers.

### Train/test split

The data is divided using scikit-learn's `train_test_split` with:

- `test_size=0.2` — 80% training data and 20% test data;
- `random_state=2` — a repeatable split when the same input data and library behavior are used.

With the bundled dataset, this produces 600,000 training records and 150,000 test records.

### Model and metric

The notebook initializes `XGBRegressor()` with its library defaults, fits it on the training split, and calculates:

\[
\operatorname{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
\]

MAE is expressed in the same units as the `Calories` target. Lower values indicate predictions that are, on average, closer to the observed values.

## Quick start

### 1. Clone the repository

```bash
git clone <repository-url>
cd Calories-Burnt-Prediction-Model
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install jupyter matplotlib numpy pandas scikit-learn seaborn xgboost
```

### 4. Verify the data file

Run this from the repository root:

```bash
head -n 2 sample_data/train.csv
```

The first line should contain the expected column headers, beginning with `id,Sex,Age` and ending with `Calories`.

## Run the notebook

Start Jupyter from the repository root so the notebook can resolve its relative data path:

```bash
jupyter notebook calories_burnt_prediction.ipynb
```

Then select **Run All** in the Jupyter interface. Training uses 600,000 examples and may take time depending on your CPU, memory, and installed XGBoost version.

For a non-interactive execution, install the dependencies above and run:

```bash
jupyter nbconvert --to notebook --execute --inplace calories_burnt_prediction.ipynb
```

This command updates the notebook's stored cell outputs. Use a copy of the notebook if you do not want to modify the working tree.

## Results

The notebook's currently saved output reports a test-set MAE of:

```text
Mean Absolute Error: 2.3433596724996963
```

Treat this as a recorded run rather than a guaranteed benchmark. Results can change with dependency versions, hardware behavior, XGBoost defaults, or changes to the CSV.

## Reproducibility and limitations

- The split seed is fixed, but the notebook does not pin Python package versions or explicitly set every XGBoost training parameter.
- The model uses XGBoost defaults; it does not include hyperparameter search, cross-validation, feature-importance analysis, or calibration.
- Encoding a binary categorical value as `0`/`1` matches the notebook's implementation. New data must use the same encoding before calling the trained model.
- The notebook trains and evaluates a model in memory; it does not save a fitted model artifact or expose an API/command-line prediction interface.
- The bundled data is large (approximately 34 MB). Ensure adequate disk space and memory before running all cells.
- Correlation and model accuracy do not establish causation or guarantee accuracy for populations, activities, sensors, or conditions absent from the training data.

## Troubleshooting

| Problem | Suggested resolution |
| --- | --- |
| `FileNotFoundError` for `./sample_data/train.csv` | Start Jupyter from the repository root, or adjust the `pd.read_csv(...)` path in the notebook. |
| `ModuleNotFoundError` | Activate the virtual environment and reinstall the packages listed in [Install dependencies](#3-install-dependencies). |
| Training is slow or runs out of memory | Close other memory-intensive programs, use a machine with more memory, or prototype with a smaller CSV sample. |
| Plotting warnings/errors from Seaborn | The notebook uses older plotting calls such as `sns.distplot`; use a Seaborn version that supports them or update the notebook to a current equivalent such as `sns.histplot`. |
| Different MAE after rerunning | Verify that the same CSV and split seed are used, then compare Python, scikit-learn, and XGBoost versions. |

## Next steps

Useful extensions include pinning dependency versions, adding cross-validation and tuned XGBoost parameters, persisting the trained model, adding automated tests, and creating a small prediction interface that validates incoming features.
