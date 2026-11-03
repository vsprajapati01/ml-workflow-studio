# ML Workflow Studio

A Streamlit application for exploring tabular data, training baseline machine-learning models, comparing results, and testing predictions.

## What it does

1. **Upload data** — CSV and Excel files, or a direct public file URL.
2. **Check data quality** — inspect missing values, data types, and duplicate rows; optionally remove duplicates.
3. **Explore data** — descriptive statistics, distributions, and numeric correlation heatmap.
4. **Train models** — choose a regression or classification task and compare baseline models.
5. **Evaluate** — view task-appropriate metrics and a confusion matrix for classification.
6. **Export** — download model comparison results as CSV and a workflow summary as Markdown.
7. **Predict** — enter one record or upload a batch file with the required feature columns.

## Models

**Regression:** Linear Regression, Ridge, Random Forest Regressor, Gradient Boosting Regressor.

**Classification:** Logistic Regression, Random Forest Classifier, Gradient Boosting Classifier.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Deploy the repository with Streamlit Community Cloud and set the app entry point to `app.py`. Test the deployment with a small, non-sensitive sample dataset before sharing it.

## Data handling and modeling notes

- The original uploaded file is not modified.
- Rows with missing target values are excluded from training; target values are not imputed.
- Numeric and categorical feature imputers are part of the scikit-learn pipeline. They are fitted on training data only, reducing preprocessing leakage into the test set.
- Classification uses a stratified split when class counts and split sizes allow it; otherwise a regular random split is used with a warning.
- Unknown categories at prediction time are handled by the one-hot encoder.
- Model scores are estimates from a held-out split, not guarantees of performance on future data. Small or imbalanced datasets can produce unstable metrics.
- Public URL ingestion is intended for direct public CSV/Excel files. Do not use private data or untrusted links.

## Current limitations

This is a baseline tabular ML tool, not an AutoML platform. It does not yet provide nested cross-validation, hyperparameter search, model persistence, SHAP explanations, or a production prediction API. Those are possible future improvements.
