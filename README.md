# Diabetes classification experiment

Compares scikit-learn Decision Tree, Random Forest and Gradient Boosting on the included CSV. This is an exploratory portfolio experiment.

## Run

Use Python 3.11 or 3.12 in a virtual environment, from the repository root:

```shell
pip install -r requirements.txt
python src/train_all.py
python -m unittest discover -s tests
```

## Evaluation

Zero values in Glucose, BloodPressure, SkinThickness, Insulin and BMI are converted to missing values. Median imputation, scaling and SMOTE are fitted inside each training fold. A stratified 20% test set is reserved before fitting preprocessing. Models are selected by five-fold ROC-AUC on the training split before viewing test outcomes. Held-out comparisons for all candidates are descriptive; do not repeatedly tune against this test set.

`outputs/metrics.json` records dataset checksum, split sizes, seed, threshold, training CV scores and test metrics. `models/selected_pipeline.pkl` includes preprocessing and the selected classifier. Only load artifacts you trained yourself. Earlier chart and accuracy claims were removed because they did not match the committed code. No EDA notebook, Power BI dashboard or SHAP explanation is included.

## Data and limits

**Dataset provenance is unverified.** The included CSV has 768 rows and 188 positive outcomes (24.5%). This differs from the commonly cited Pima dataset prevalence, so this repository does not claim to reproduce canonical Pima results. The file is retained unchanged to avoid silently substituting another dataset. Results describe this CSV only; verify its original source before presenting it as a Pima analysis.
