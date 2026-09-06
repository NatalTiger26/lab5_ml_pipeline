# Titanic Survival Prediction — Leak-Free ML Pipeline

A two-day exploration-to-modeling pipeline on the Titanic dataset: data quality
auditing and cleaning, a leak-free `ColumnTransformer` + `Pipeline`, hypothesis-driven
feature engineering (tested via ablation, not assumed), a cross-validated model
comparison, and a single held-out test evaluation.

## Project structure

```
.
├── data.py                              # Dataset loader (OpenML, falls back to a matched synthetic dataset offline)
├── pipeline.py                          # engineer(), build_preprocessor(), build_pipeline(), get_feature_columns()
├── day1_exploration_starter.ipynb       # Data audit, cleaning decisions, train/test split
├── day2_pipeline_modeling_starter.ipynb # Preprocessing pipeline, feature ablation, model comparison, tuning, test eval
├── split.joblib                         # Saved train/test split (produced by Day 1, consumed by Day 2)
└── requirements.txt
```

## Setup

```bash
uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
jupyter notebook
```

`data.py` tries to fetch the real Titanic dataset from OpenML first. With no network
access it falls back to a synthetic dataset built to reproduce the same schema,
missingness pattern, and leakage-relevant columns (`boat`, `body`, `home.dest`), so
both notebooks run either way.

## How to run

1. Open `day1_exploration_starter.ipynb` and run top to bottom. It profiles the raw
   data, documents a cleaning decision per problem column, splits the data
   (`stratify=y`, fixed `random_state`), and saves the split to `split.joblib`.
2. Open `day2_pipeline_modeling_starter.ipynb` and run top to bottom. It reloads
   `split.joblib`, builds the preprocessing pipeline, engineers and ablates
   features, cross-validates four model types, tunes the best one, and evaluates
   the tuned model on the test set exactly once.

## Data cleaning decisions (Day 1)

| Column | Decision | Reason |
|---|---|---|
| `boat`, `body` | Dropped | Assigned only once survival outcome is known — near-perfect / perfect target leakage |
| `home.dest` | Dropped | ~43% missing, 369 unique values, known only after the fact |
| `Cabin` | Raw string dropped, `has_cabin` flag kept | ~77% missing, but presence of a recorded cabin correlates strongly with survival (~65% vs ~30%) |
| `Name` | Raw text dropped, `title` extracted | Title (Mr/Mrs/Miss/Master/Rare) is a more compact status/age/sex signal than the raw name |
| `Ticket` | Dropped | High cardinality (929 unique), little standalone signal beyond `Fare`/`Pclass` |
| `Age` | Kept, median-imputed | ~20% missing; missingness correlates with survival, so the column is kept rather than dropped, with a robust (median) imputer |
| `Embarked` | Kept, most-frequent imputed | Only 2 missing rows |
| `Fare` | Kept, median-imputed; zeros left as-is | 1 missing value; the ~17 zero fares are treated as plausible (crew/complimentary), not placeholders |
| `Pclass` | Kept, treated as categorical | Ordinal label, not a continuous quantity — must not be scaled |

## Pipeline design (`pipeline.py`)

- **`engineer(df)`** — adds `family_size`, `is_alone`, `title` (extracted from `Name`,
  rare titles bucketed), and `has_cabin`, each with its hypothesis stated up front.
- **`build_preprocessor(num_cols, cat_cols)`** — numeric columns are median-imputed
  then scaled; categorical columns are most-frequent-imputed then one-hot encoded
  with `handle_unknown="ignore"` so a category unseen at fit time doesn't crash
  prediction.
- **`get_feature_columns(X)`** — splits columns by dtype, with explicit overrides so
  `Pclass`, `title`, `Sex`, `Embarked`, `is_alone`, and `has_cabin` are always treated
  as categorical regardless of their numeric encoding.
- Everything is fit inside a single `sklearn.Pipeline`, refit from scratch on each
  cross-validation fold, so no preprocessing step ever sees held-out data.

## Results (Day 2, 5-fold stratified CV on the training set)

**Feature ablation** (Logistic Regression, F1):

| Feature set | F1 |
|---|---|
| All engineered features | 0.754 ± 0.029 |
| Without `family_size` / `is_alone` | 0.745 ± 0.029 |
| Without `title` | 0.713 ± 0.026 |
| Without `has_cabin` | 0.749 ± 0.033 |
| Raw columns only (no engineered features) | 0.707 ± 0.033 |

`title` gives the clearest lift; `family_size`/`is_alone` and `has_cabin` help
marginally, within noise of each other.

**Model comparison** (same preprocessing, all engineered features):

| Model | F1 | AUC |
|---|---|---|
| Logistic Regression | 0.754 ± 0.029 | 0.853 ± 0.012 |
| Gradient Boosting | 0.746 ± 0.052 | 0.866 ± 0.018 |
| Random Forest | 0.730 ± 0.031 | 0.846 ± 0.024 |
| KNN | 0.719 ± 0.039 | 0.850 ± 0.021 |

The spread across models is smaller than each model's own fold-to-fold standard
deviation, so this is not strong evidence that any one model is genuinely better;
Logistic Regression was carried forward for tuning as the highest mean.

**Tuning and final test evaluation** (test set touched exactly once):

- Tuned Logistic Regression (`C=5.0`): CV F1 = 0.756, held-out test F1 = 0.758, test AUC = 0.865
- Untuned Logistic Regression baseline: held-out test F1 = 0.754

Tuning produced a small (~0.004) improvement, on the order of the cross-validation
noise — reported honestly rather than treated as a decisive win.

## What we'd try next

- Interaction features (`Sex × Pclass`), or binning `Age` into groups.
- A feature derived from `home.dest`'s country of origin, with missing values
  bucketed as `"Unknown"` and one-hot encoded, rather than dropping the column
  outright.
- Probability calibration if the model's output is used as a score rather than a
  hard prediction.
- Re-running the ablation with a leaner feature set if a feature is later confirmed
  to be pure noise.

## Requirements

See `requirements.txt`. Core dependencies: `pandas`, `numpy`, `scikit-learn`,
`matplotlib`, `seaborn`, `joblib`.