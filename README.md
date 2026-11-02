# ML Workflow Studio

Upload a CSV or Excel file (or paste a link to one) and the app runs a full
machine-learning workflow, step by step:

1. **Upload** — CSV / Excel file, or a direct link to one.
2. **Clean, or skip cleaning** — if your data is already clean, skip ahead to
   insights. Otherwise the app removes duplicates, fills missing values
   (median for numbers, most common value for categories), and caps outliers
   with a conservative 3x IQR clip, logging every change.
3. **Insights (EDA)** — summary stats, distributions, correlation heatmap,
   and missing-value chart.
4. **Train and compare** — pick a target column. Regression models: Linear
   Regression, Ridge, Random Forest, Gradient Boosting. Classification
   models: Logistic Regression, Random Forest, Gradient Boosting. All are
   trained on the same 80/20 split so the comparison is fair.
5. **Report** — comparison table with MAE / RMSE / R² (regression) or
   accuracy / precision / recall / F1 (classification), downloadable as
   Markdown and CSV.
6. **Test the best model** — enter feature values one by one and predict,
   or upload a test file and download predictions for every row.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free

Push this repo to GitHub, then go to
[share.streamlit.io](https://share.streamlit.io), pick this repo, set the main
file to `app.py`, and deploy. No code changes needed.

## Notes

- The app never changes your original file; cleaning happens on a copy.
- Model quality depends on the data. A high score on synthetic or leaky data
  does not mean it will work the same on new real data.
- Start with a small file (a few thousand rows) to check the flow first.
