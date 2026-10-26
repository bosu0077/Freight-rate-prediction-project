# Freight Rate Prediction — Spotter ML Engineer Assessment

End-to-end machine-learning solution that predicts the **posted rate (USD)** of truckload loads.
The work is organised as a capstone notebook (data audit → EDA → cleaning → time-based validation →
feature engineering → baselines → production pipeline → tuning → diagnostics → predictions).

## Results at a glance

Train on Jan–Aug 2025, score the untouched **Sep–Oct 2025 holdout** (a 2-month-ahead forecast, like Nov–Dec):

| Metric (holdout, clean labels) | Value |
|---|---|
| MAPE | **1.68%** |
| MAE | **\$38.44** per load |
| RMSE | \$55.56 |
| R² | 0.9984 |
| Loads within ±5% | 97.9% |

Including the 677 flagged corrupted labels the holdout MAPE is 4.20% — the
difference comes from labels that are 2–6× wrong, not from the model.

## Approach

**Key findings from the data**
- The validation loads (Nov–Dec) are later than every labelled load (Jan–Oct) → this is a **forecasting** problem.
- **677 labels (1.41%) are corrupted** (2–6× too high or too low) — removed from training.
- **`quote_signal` flips regime every month**: equal to the rate per mile in some months, its mirror image in others,
  pure noise in August — and (checked without labels) noise in Nov–Dec. **Excluded.**
- **`market_index`** is noisy at load level and drifts away from rates in Sep–Oct → used only as a *daily* average
  with a small linear elasticity next to a trend.
- Rates rise **~4% over the last ~25–30 days of every quarter** — December is a quarter end.
- **8 cities exist only in validation** (1,447 loads) → coordinates instead of city IDs.

**Validation / split**
- Development = Jan 1–Aug 31, holdout = Sep 1–Oct 31 (scored once), tuning on expanding-window folds
  (Jan–Apr → May–Jun, Jan–Jun → Jul–Aug), final refit on Jan–Oct.
- Random K-fold would have picked the wrong model:

| Model | Random 5-fold CV MAPE | Time holdout MAPE |
|---|---|---|
| Naive HistGB (all raw columns) | 1.56% | 3.52% |
| HistGB + raw quote & market signals | 1.22% | 1.95% |
| **Two-stage production model** | 1.53% | **1.70%** |

**Model** — `log(rate) = lane price + market level`
- *Lane price*: `HistGradientBoostingRegressor` on distance, coordinates, circuity, equipment and weight.
- *Market level*: linear model on day index (trend), log daily market index and a quarter-end ramp, so the level
  **extrapolates** into Nov–Dec (trees cannot).
- Both parts are fitted jointly by back-fitting inside a scikit-learn `Pipeline` with a custom `FreightFeatureEngineer`.
- Tuned with `RandomizedSearchCV` on the time-series folds: min_samples_leaf=10, max_leaf_nodes=63, learning_rate=0.03, l2_regularization=0, ramp_days=30.

**All models on the Sep–Oct holdout**

| Model | MAE (\$) | MAPE (%) | R² |
|---|---|---|---|
| Two-stage production model | 38.90 | 1.70 | 0.9983 |
| HistGradientBoosting (log) | 44.07 | 1.89 | 0.9979 |
| HistGB + raw quote & market signals | 46.84 | 1.95 | 0.9976 |
| Gradient Boosting (log) | 46.37 | 1.97 | 0.9977 |
| Ridge (log, alpha=1) | 47.62 | 2.10 | 0.9976 |
| Linear Regression (log) | 47.62 | 2.10 | 0.9976 |
| Random Forest (log) | 62.70 | 2.62 | 0.9956 |
| Decision Tree (log) | 77.02 | 3.37 | 0.9936 |
| Naive HistGB (all raw columns) | 84.88 | 3.52 | 0.9933 |
| Rate-per-mile heuristic | 83.39 | 3.74 | 0.9929 |

**December chart** (fixed Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb): \$818.38–\$883.59,
mid-week peaks / Sunday–Monday dips from the market index and a ~4.0% quarter-end surge into Dec 31.

## Repository structure

```
freight-rate-prediction/
├── Freight_Rate_Prediction.ipynb      # main solution - run top to bottom
├── score.py                           # provided scorer (unchanged)
├── requirements.txt
├── validation_predictions.csv         # submission file: load_id,predicted_rate
├── data/
│   ├── train_test.csv
│   ├── validation.csv
│   ├── validation_predictions_template.csv
│   └── december_chart_inputs.csv      # predicted_rate filled by the notebook
├── scorer_results/
│   └── candidate_december.png         # produced by score.py
├── outputs/
│   ├── figures/                       # EDA and evaluation figures
│   ├── models/freight_rate_model.joblib
│   └── reports/                       # metrics.json, baseline_results.csv, data_quality_log.csv, ...
└── report/
    ├── Freight_Rate_Report.pdf        # validation approach + December chart
    └── Freight_Rate_Report.docx
```

## How to run

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

# run the whole notebook (about 7-10 minutes on a laptop)
jupyter nbconvert --to notebook --execute --inplace Freight_Rate_Prediction.ipynb
#   or open it interactively:  jupyter notebook Freight_Rate_Prediction.ipynb  ->  Run All
```

The notebook writes `validation_predictions.csv`, fills `data/december_chart_inputs.csv`, runs `score.py`
(which validates both files and creates `scorer_results/candidate_december.png`) and saves the model,
figures and metrics under `outputs/`.

To run the scorer on its own:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The notebook looks for the data in `data/` (official names) and also accepts the files in the project root.

**Re-using the saved model:** `outputs/models/freight_rate_model.joblib` holds the whole fitted pipeline. It relies on the
two custom classes defined in the notebook (`FreightFeatureEngineer`, `TwoStageFreightRegressor`), so load it in a session
where those cells have been run, with the same scikit-learn version that saved it (re-running the notebook re-creates it).

## Tested with

- Python 3.9 · pandas 2.1 · numpy 1.26 · scikit-learn 1.0.2 · matplotlib 3.8
- Python 3.9 · pandas 2.3 · numpy 2.0 · scikit-learn 1.6 · matplotlib 3.9 (what `pip install -r requirements.txt` resolves to)
- Python 3.12 · pandas 2.3 · numpy 2.x · scikit-learn 1.9 · matplotlib 3.11 (the run saved in this repository)

Fixed random seed (42). Different library versions give the same model choice and holdout MAPE within ±0.01
percentage points; individual predictions differ by about 0.2% on average.
