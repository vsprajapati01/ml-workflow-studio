"""ML Workflow Studio — upload data, clean (or skip), explore, train,
compare models, get a report, and test the best model.

Run:  pip install -r requirements.txt
       streamlit run app.py
"""
import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="ML Workflow Studio", layout="wide")
st.title("ML Workflow Studio")
st.write(
    "Upload a CSV or Excel file (or paste a link to one). The app cleans it "
    "— or skips cleaning if your data is already clean — explores it, trains "
    "several models, compares them, and lets you test the best one."
)

# ---------------------------------------------------------------- load data
st.header("1. Upload your data")
source = st.radio("Data source", ["Upload a file", "Link to a CSV / Excel file"])
df = None
if source == "Upload a file":
    uploaded = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        except Exception as exc:  # noqa: BLE001 - show the real error to the user
            st.error(f"Could not read that file: {exc}")
else:
    url = st.text_input("Paste a direct link to a CSV or Excel file")
    if url:
        try:
            df = pd.read_csv(url) if url.lower().endswith(".csv") else pd.read_excel(url)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read that link: {exc}")

if df is None:
    st.info("Upload a file or paste a link to get started. Try a small CSV first.")
    st.stop()

st.write(f"**Rows:** {len(df):,} &nbsp;|&nbsp; **Columns:** {len(df.columns)}")
st.dataframe(df.head(10), use_container_width=True)

# ------------------------------------------------------- clean or skip step
st.header("2. Clean, or skip cleaning")
already_clean = st.checkbox(
    "My data is already cleaned — skip to insights",
    help="No values are changed. The app still checks for problems and warns you.",
)
clean_log = []
work = df.copy()

if already_clean:
    missing = work.isna().sum()
    missing = missing[missing > 0]
    dupes = int(work.duplicated().sum())
    if len(missing) or dupes:
        st.warning(
            "Heads-up: this data still has "
            + ", ".join([f"{c} ({n} missing)" for c, n in missing.items()])
            + (f" and {dupes} duplicate rows." if dupes else ".")
            + " Training may fail — untick the box above to run cleaning, or fix the file."
        )
    else:
        st.success("No missing values or duplicate rows found. Moving on unchanged.")
    clean_log.append("Cleaning skipped by user (data declared already clean).")
else:
    before = len(work)
    work = work.drop_duplicates()
    if before - len(work):
        clean_log.append(f"Removed {before - len(work):,} duplicate rows.")
    for col in work.columns:
        n_missing = int(work[col].isna().sum())
        if not n_missing:
            continue
        if pd.api.types.is_numeric_dtype(work[col]):
            work[col] = work[col].fillna(work[col].median())
            clean_log.append(f"{col}: filled {n_missing:,} missing values with the median.")
        else:
            mode = work[col].mode()
            fill = mode.iloc[0] if len(mode) else "Unknown"
            work[col] = work[col].fillna(fill)
            clean_log.append(f"{col}: filled {n_missing:,} missing values with the most common value.")
    # IQR cap (conservative 3x, like a careful analyst): clip, don't delete.
    for col in work.select_dtypes(include="number").columns:
        q1, q3 = work[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        n_capped = int(((work[col] < lo) | (work[col] > hi)).sum())
        if n_capped:
            work[col] = work[col].clip(lo, hi)
            clean_log.append(f"{col}: capped {n_capped:,} outlier values to the 3x IQR range.")
    st.subheader("Cleaning log")
    for line in clean_log or ["No cleaning changes were needed."]:
        st.write("- " + line)
    st.dataframe(work.head(10), use_container_width=True)

# ------------------------------------------------------------------- EDA
st.header("3. Insights (EDA)")
st.subheader("Summary")
st.dataframe(work.describe(include="all").T, use_container_width=True)

numeric_cols = work.select_dtypes(include="number").columns.tolist()
if numeric_cols:
    col = st.selectbox("Distribution chart", numeric_cols)
    st.plotly_chart(px.histogram(work, x=col, title=f"Distribution of {col}"), use_container_width=True)
    if len(numeric_cols) > 1:
        st.plotly_chart(
            px.imshow(work[numeric_cols].corr(), text_auto=".2f", title="Correlation heatmap"),
            use_container_width=True,
        )
missing_chart = work.isna().sum()
missing_chart = missing_chart[missing_chart > 0]
if len(missing_chart):
    st.plotly_chart(px.bar(missing_chart, title="Missing values by column"), use_container_width=True)
else:
    st.write("No missing values remain in this data.")

# ------------------------------------------------------------- train models
st.header("4. Train and compare models")
target = st.selectbox("Pick the column to predict (target)", work.columns.tolist())
if target is None:
    st.stop()
features = [c for c in work.columns if c != target]
if not features:
    st.error("Need at least one column besides the target.")
    st.stop()

y = work[target]
is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > 10
problem = st.radio(
    "Problem type (auto-detected, change it if wrong)",
    ["Regression", "Classification"],
    index=0 if is_regression else 1,
)
is_regression = problem == "Regression"

X = work[features]
num_feats = X.select_dtypes(include="number").columns.tolist()
cat_feats = [c for c in features if c not in num_feats]
pre = ColumnTransformer(
    [
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_feats),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
    ]
)

if is_regression:
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
else:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

if st.button("Train all models"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if not is_regression and y.nunique() > 1 else None
    )
    rows, fitted = [], {}
    for name, model in models.items():
        pipe = Pipeline([("preprocess", pre), ("model", model)])
        try:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            fitted[name] = pipe
            if is_regression:
                rows.append({
                    "Model": name,
                    "MAE": round(mean_absolute_error(y_test, pred), 4),
                    "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
                    "R²": round(r2_score(y_test, pred), 4),
                })
            else:
                rows.append({
                    "Model": name,
                    "Accuracy": round(accuracy_score(y_test, pred), 4),
                    "Precision": round(precision_score(y_test, pred, average="weighted", zero_division=0), 4),
                    "Recall": round(recall_score(y_test, pred, average="weighted", zero_division=0), 4),
                    "F1": round(f1_score(y_test, pred, average="weighted", zero_division=0), 4),
                })
        except Exception as exc:  # noqa: BLE001
            rows.append({"Model": name, "Error": str(exc)})
    results = pd.DataFrame(rows)
    st.session_state["results"] = results
    st.session_state["fitted"] = fitted
    st.session_state["features"] = features
    st.session_state["target"] = target
    st.session_state["is_regression"] = is_regression
    st.session_state["work"] = work
    st.subheader("Results — lower MAE/RMSE is better, higher R² / accuracy / F1 is better")
    st.dataframe(results, use_container_width=True)
    if "R²" in results.columns:
        best = results.loc[results["R²"].idxmax(), "Model"]
    elif "F1" in results.columns:
        best = results.loc[results["F1"].idxmax(), "Model"]
    else:
        best = None
    if best:
        st.success(f"Best model on this test split: **{best}**")
        st.session_state["best"] = best
        score_col = "R²" if "R²" in results.columns else "F1"
        st.plotly_chart(px.bar(results, x="Model", y=score_col, title=f"Model comparison ({score_col})"), use_container_width=True)

# ------------------------------------------------------------------ report
st.header("5. Report")
if "results" in st.session_state:
    report_lines = [
        "# ML Workflow Report",
        "",
        f"Target: {st.session_state['target']} | Problem: {'Regression' if st.session_state['is_regression'] else 'Classification'}",
        f"Rows: {len(st.session_state['work']):,} | Features: {', '.join(st.session_state['features'])}",
        "",
        "## Cleaning",
        *[f"- {line}" for line in clean_log],
        "",
        "## Model comparison",
        st.session_state["results"].to_markdown(index=False),
        "",
        f"Best model: {st.session_state.get('best', 'n/a')}",
    ]
    report = "\n".join(report_lines)
    st.download_button("Download report (Markdown)", report, file_name="ml-workflow-report.md")
    st.download_button(
        "Download results (CSV)",
        st.session_state["results"].to_csv(index=False),
        file_name="model-results.csv",
    )
else:
    st.info("Train the models above first, then the report download appears here.")

# ------------------------------------------------------------ test platform
st.header("6. Test the best model")
if "best" in st.session_state and st.session_state["best"] in st.session_state["fitted"]:
    pipe = st.session_state["fitted"][st.session_state["best"]]
    st.write(f"Using **{st.session_state['best']}**. Enter values for each feature and predict.")
    inputs = {}
    for col in st.session_state["features"]:
        series = st.session_state["work"][col]
        if pd.api.types.is_numeric_dtype(series):
            inputs[col] = st.number_input(col, value=float(series.median()))
        else:
            options = sorted(series.dropna().astype(str).unique().tolist())
            inputs[col] = st.selectbox(col, options)
    if st.button("Predict"):
        row = pd.DataFrame([inputs])
        pred = pipe.predict(row)[0]
        st.success(f"Prediction for {st.session_state['target']}: **{pred}**")
        st.write("How to read this: change one input at a time to see how the prediction moves. "
                 "This is a test of the trained model on your inputs, not a guarantee for new real data.")
    st.subheader("Batch test: upload a file with the same feature columns")
    test_file = st.file_uploader("Test file (CSV or Excel)", type=["csv", "xlsx", "xls"], key="test_file")
    if test_file is not None:
        try:
            test_df = pd.read_csv(test_file) if test_file.name.endswith(".csv") else pd.read_excel(test_file)
            test_df["prediction"] = pipe.predict(test_df[st.session_state["features"]])
            st.dataframe(test_df.head(20), use_container_width=True)
            st.download_button("Download predictions (CSV)", test_df.to_csv(index=False), file_name="predictions.csv")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not run batch predictions: {exc}")
else:
    st.info("Train the models above first, then the testing platform appears here.")
