# AI-NIDS: Complete Network Intrusion Detection Project

A complete individual B.Tech CSE (AI/ML) project using the UNSW-NB15 dataset.

## Dataset download (not included in this ZIP)
Download the UNSW-NB15 dataset from:
- Kaggle: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
- Official UNSW page: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Put these two CSV files directly inside `data/`:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

The dataset is not bundled because it is a separate, large dataset. This ZIP contains the complete project code and folder structure.

## Windows + VS Code setup

Open this extracted folder in VS Code. In PowerShell terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run everything in this order

```powershell
python -m src.train
python -m src.evaluate
python -m src.explain
streamlit run app.py
```

Training can take several minutes depending on laptop CPU/RAM. For a quick first test, open `src/config.py` and set `SAMPLE_FRAC = 0.20`. Set it back to `1.0` for final experiments.

## What is included

- Binary classification: Normal vs Attack
- Models: Logistic Regression, Decision Tree, Random Forest, XGBoost (if installed)
- Multiclass classification: Normal and UNSW-NB15 attack categories using Random Forest
- Evaluation: Accuracy, precision, recall, F1, false-positive rate, classification reports and confusion matrices
- Explainable AI: SHAP global feature-importance summary for the binary Random Forest
- Streamlit dashboard: CSV upload, binary/multiclass predictions, metrics and confusion-matrix visualizations

## Methodology notes

- Binary target: `label` (0=Normal, 1=Attack).
- `id` and `attack_cat` are excluded from binary model inputs to avoid identifier and target leakage.
- For multiclass training, the target is `attack_cat`; `id` and `label` are excluded from features.
- Missing `attack_cat` values on normal rows are mapped to `Normal`.
- Imputers, scaling and one-hot encoding are fitted inside a scikit-learn Pipeline on training data only.
- The official test CSV is kept separate for evaluation.
- Report macro-F1 and per-class recall as well as accuracy; class imbalance can make accuracy misleading.
- This is an offline benchmark/educational prototype, not a production network security appliance.

## Output files

Generated files are saved in `models/` and `reports/`. These generated model/report files are not included in the ZIP; create them by running the commands above.
