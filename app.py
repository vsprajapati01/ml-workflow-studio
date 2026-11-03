"""ML Workflow Studio — reliable end-to-end tabular ML workflow."""
import hashlib
import io
import ipaddress
import socket
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="ML Workflow Studio", page_icon="🧪", layout="wide")


def markdown_table(frame):
    """Create a Markdown table without an optional dependency."""
    safe = frame.fillna("").astype(str)
    lines = ["| " + " | ".join(map(str, safe.columns)) + " |",
             "| " + " | ".join(["---"] * len(safe.columns)) + " |"]
    for row in safe.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def read_uploaded_file(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def validate_public_url(raw_url):
    """Allow public HTTP(S) URLs only; reject localhost/private destinations."""
    parsed = urlparse(raw_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Enter a complete public http:// or https:// URL.")
    host = parsed.hostname
    try:
        addresses = socket.getaddrinfo(host, None)
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if not ip.is_global:
                raise ValueError("Private or local network URLs are not allowed.")
    except socket.gaierror as exc:
        raise ValueError("The URL hostname could not be resolved.") from exc
    if not parsed.path.lower().endswith((".csv", ".xlsx", ".xls")):
        raise ValueError("The link must point directly to a .csv, .xlsx, or .xls file.")
    return raw_url.strip()


def build_preprocessor(numeric_features, categorical_features):
    transformers = []
    if numeric_features:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ))
    return ColumnTransformer(transformers, remainder="drop")


def make_models(regression):
    if regression:
        return {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }
    return {
        "Logistic Regression": LogisticRegression(max_iter=1500),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def reset_training_state():
    for key in ("results", "fitted", "features", "target", "is_regression",
                "work", "best", "clean_log", "report"):
        st.session_state.pop(key, None)


st.title("🧪 ML Workflow Studio")
st.caption("Explore data, prepare features, compare models, and make predictions.")
st.write(
    "A guided workspace for tabular machine learning. Your original uploaded "
    "data is never overwritten."
)

# 1. Data
st.header("1 · Upload data")
source = st.radio("Data source", ["Upload a file", "Public file URL"], horizontal=True)
df = None
if source == "Upload a file":
    uploaded = st.file_uploader(
        "Choose a CSV or Excel file", type=["csv", "xlsx", "xls"]
    )
    if uploaded:
        try:
            if uploaded.size > 25 * 1024 * 1024:
                st.error("Please upload a file smaller than 25 MB.")
            else:
                df = read_uploaded_file(uploaded)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
else:
    url = st.text_input("Direct link to a public CSV or Excel file")
    if url:
        try:
            safe_url = validate_public_url(url)
            st.info("Loading the public file URL…")
            df = pd.read_csv(safe_url) if safe_url.lower().endswith(".csv") else pd.read_excel(safe_url)
        except Exception as exc:
            st.error(f"Could not load that link: {exc}")

if df is None:
    st.info("Upload a dataset to begin. For a first run, try a small CSV file.")
    st.stop()

if df.empty or len(df.columns) == 0:
    st.error("This file has no usable rows or columns.")
    st.stop()
if len(df.columns) < 2:
    st.error("At least two columns are required: one target and one feature.")
    st.stop()

# Avoid ambiguous duplicate column labels and unsupported all-empty columns.
df.columns = [str(c).strip() for c in df.columns]
if pd.Index(df.columns).duplicated().any():
    st.error("Your dataset has duplicate column names. Rename them and upload again.")
    st.stop()
empty_cols = df.columns[df.isna().all()].tolist()
if empty_cols:
    df = df.drop(columns=empty_cols)
    st.warning("Removed completely empty columns: " + ", ".join(empty_cols))
if len(df.columns) < 2:
    st.error("After removing empty columns, at least two columns are required.")
    st.stop()

data_signature = hashlib.sha256(
    pd.util.hash_pandas_object(df, index=True).values.tobytes()
    + "|".join(df.columns).encode()
).hexdigest()
if st.session_state.get("data_signature") != data_signature:
    reset_training_state()
    st.session_state["data_signature"] = data_signature

m1, m2, m3 = st.columns(3)
m1.metric("Rows", f"{len(df):,}")
m2.metric("Columns", f"{len(df.columns):,}")
m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
with st.expander("Preview uploaded data", expanded=True):
    st.dataframe(df.head(15), use_container_width=True)

# 2. Data checks / optional cleaning
st.header("2 · Data quality")
clean_option = st.radio(
    "Preparation option",
    ["Remove duplicate rows", "Keep rows as uploaded"],
    horizontal=True,
    help="Missing feature values are imputed inside the training pipeline using training data only.",
)
work = df.copy()
clean_log = []
if clean_option == "Remove duplicate rows":
    duplicate_count = int(work.duplicated().sum())
    if duplicate_count:
        work = work.drop_duplicates().copy()
        clean_log.append(f"Removed {duplicate_count:,} duplicate rows.")
    else:
        clean_log.append("No duplicate rows found.")
else:
    clean_log.append("Duplicate rows were retained as requested.")

quality = pd.DataFrame({
    "Missing values": work.isna().sum(),
    "Missing %": (work.isna().mean() * 100).round(2),
    "Unique values": work.nunique(dropna=True),
    "Data type": work.dtypes.astype(str),
})
st.dataframe(quality, use_container_width=True)
st.caption("Feature imputation is fitted only on the training split to avoid test-data leakage.")

# 3. EDA
st.header("3 · Explore the data")
with st.expander("Descriptive statistics", expanded=True):
    st.dataframe(work.describe(include="all").T, use_container_width=True)
numeric_cols = work.select_dtypes(include=np.number).columns.tolist()
if numeric_cols:
    chart_col = st.selectbox("Explore a numeric column", numeric_cols)
    st.plotly_chart(
        px.histogram(work, x=chart_col, title=f"Distribution of {chart_col}"),
        use_container_width=True,
    )
    if len(numeric_cols) > 1:
        st.plotly_chart(
            px.imshow(work[numeric_cols].corr(), text_auto=".2f",
                      title="Numeric feature correlations"),
            use_container_width=True,
        )
else:
    st.info("No numeric columns are available for numeric charts.")

# 4. Model training
st.header("4 · Train and compare models")
target = st.selectbox("Select the column to predict", work.columns.tolist())
features = [c for c in work.columns if c != target]
target_data = work.loc[work[target].notna()].copy()
if target_data.empty:
    st.error("The selected target column contains no non-missing values.")
    st.stop()
if len(target_data) != len(work):
    st.warning(
        f"{len(work) - len(target_data):,} rows with missing target values "
        "will be excluded from training. Target values are never imputed."
    )
if target_data[target].nunique(dropna=True) < 2:
    st.error("The target must contain at least two distinct values.")
    st.stop()

auto_regression = (
    pd.api.types.is_numeric_dtype(target_data[target])
    and target_data[target].nunique() > 10
)
problem = st.radio(
    "Prediction task",
    ["Regression", "Classification"],
    index=0 if auto_regression else 1,
    horizontal=True,
    help="Automatic selection is a suggestion. Choose the task that matches your problem.",
)
is_regression = problem == "Regression"
if is_regression and not pd.api.types.is_numeric_dtype(target_data[target]):
    st.error("Regression requires a numeric target. Choose classification or change the target.")
    st.stop()

if len(target_data) < 5:
    st.error("At least 5 rows with a valid target are required for a train/test split.")
    st.stop()
if not is_regression:
    class_counts = target_data[target].value_counts()
    if len(class_counts) < 2:
        st.error("Classification requires at least two target classes.")
        st.stop()
    if (class_counts < 2).any():
        st.warning(
            "At least one class has only one row. Stratified splitting is not possible; "
            "the app will use a regular random split. Results may be unstable."
        )

test_fraction = st.slider("Test data (%)", min_value=10, max_value=40, value=20, step=5)
if st.button("Train and compare models", type="primary"):
    reset_training_state()
    X = target_data[features]
    y = target_data[target]
    stratify = None
    if not is_regression and (y.value_counts() >= 2).all():
        # Stratification also requires enough test and train slots for every class.
        n_test = int(np.ceil(len(y) * test_fraction / 100))
        n_train = len(y) - n_test
        if n_test >= y.nunique() and n_train >= y.nunique():
            stratify = y
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_fraction / 100, random_state=42, stratify=stratify
        )
    except ValueError as exc:
        st.error(f"Could not create a train/test split: {exc}")
        st.stop()

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [c for c in features if c not in numeric_features]
    rows, fitted, predictions = [], {}, {}
    progress = st.progress(0, text="Training models…")
    model_items = list(make_models(is_regression).items())
    for index, (name, estimator) in enumerate(model_items, start=1):
        try:
            pipeline = Pipeline([
                ("preprocess", build_preprocessor(numeric_features, categorical_features)),
                ("model", estimator),
            ])
            pipeline.fit(X_train, y_train)
            predicted = pipeline.predict(X_test)
            fitted[name] = pipeline
            predictions[name] = predicted
            if is_regression:
                rmse = float(np.sqrt(mean_squared_error(y_test, predicted)))
                score = r2_score(y_test, predicted) if len(y_test) > 1 else np.nan
                rows.append({
                    "Model": name,
                    "MAE": mean_absolute_error(y_test, predicted),
                    "RMSE": rmse,
                    "R²": score,
                    "Status": "Success",
                })
            else:
                rows.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, predicted),
                    "Precision": precision_score(y_test, predicted, average="weighted", zero_division=0),
                    "Recall": recall_score(y_test, predicted, average="weighted", zero_division=0),
                    "F1": f1_score(y_test, predicted, average="weighted", zero_division=0),
                    "Status": "Success",
                })
        except Exception as exc:
            rows.append({"Model": name, "Status": f"Failed: {exc}"})
        progress.progress(index / len(model_items), text=f"Processed {name}")
    results = pd.DataFrame(rows)
    successful = results[results["Status"] == "Success"].copy()
    st.session_state["results"] = results
    st.session_state["fitted"] = fitted
    st.session_state["predictions"] = predictions
    st.session_state["features"] = features
    st.session_state["target"] = target
    st.session_state["is_regression"] = is_regression
    st.session_state["work"] = target_data
    st.session_state["clean_log"] = clean_log
    st.session_state["test_actual"] = y_test
    if successful.empty:
        st.error("None of the models trained successfully. Check the data types and target values.")
        st.session_state.pop("best", None)
    else:
        metric = "R²" if is_regression else "F1"
        ranked = successful.dropna(subset=[metric])
        if ranked.empty:
            st.warning("Models trained, but there are not enough test samples to rank them reliably.")
            best = successful.iloc[0]["Model"]
        else:
            best = ranked.loc[ranked[metric].idxmax(), "Model"]
        st.session_state["best"] = best
        st.success(f"Best model by {metric}: {best}")
    progress.empty()

if "results" in st.session_state:
    st.subheader("Model comparison")
    st.dataframe(st.session_state["results"], use_container_width=True)
    successful_names = list(st.session_state["fitted"].keys())
    if successful_names:
        metric = "R²" if st.session_state["is_regression"] else "F1"
        chart_data = st.session_state["results"]
        chart_data = chart_data[chart_data["Status"] == "Success"]
        st.plotly_chart(
            px.bar(chart_data, x="Model", y=metric, title=f"Model comparison by {metric}"),
            use_container_width=True,
        )
        chosen = st.selectbox("Inspect model", successful_names)
        if not st.session_state["is_regression"]:
            actual = st.session_state["test_actual"]
            predicted = st.session_state["predictions"][chosen]
            labels = sorted(pd.Series(actual).astype(str).unique().tolist())
            cm = confusion_matrix(actual.astype(str), pd.Series(predicted).astype(str), labels=labels)
            st.plotly_chart(
                px.imshow(cm, x=labels, y=labels, text_auto=True,
                          labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                          title=f"Confusion matrix — {chosen}"),
                use_container_width=True,
            )

# 5. Report
st.header("5 · Download report")
if "results" in st.session_state:
    result_frame = st.session_state["results"]
    report_lines = [
        "# ML Workflow Studio Report", "",
        f"- Target: {st.session_state['target']}",
        f"- Task: {'Regression' if st.session_state['is_regression'] else 'Classification'}",
        f"- Rows used: {len(st.session_state['work']):,}",
        f"- Features: {', '.join(st.session_state['features'])}",
        "", "## Data preparation",
        *[f"- {line}" for line in st.session_state["clean_log"]],
        "- Feature missing values are imputed inside each training pipeline.",
        "- Rows with missing target values are excluded from model training.",
        "", "## Model comparison", markdown_table(result_frame),
        "", f"Best model: {st.session_state.get('best', 'No successful model')}",
    ]
    report = "\n".join(report_lines)
    st.download_button(
        "Download Markdown report", report,
        file_name="ml-workflow-report.md", mime="text/markdown",
    )
    st.download_button(
        "Download model results (CSV)", result_frame.to_csv(index=False),
        file_name="model-results.csv", mime="text/csv",
    )
else:
    st.info("Train models to generate a report.")

# 6. Predictions
st.header("6 · Test predictions")
best = st.session_state.get("best")
if best and best in st.session_state.get("fitted", {}):
    pipeline = st.session_state["fitted"][best]
    st.caption(f"Using {best} for predictions.")
    inputs = {}
    for column in st.session_state["features"]:
        series = st.session_state["work"][column]
        if pd.api.types.is_numeric_dtype(series):
            median = series.median()
            default = float(median) if pd.notna(median) else 0.0
            inputs[column] = st.number_input(column, value=default, key=f"single_{column}")
        else:
            options = sorted(series.dropna().astype(str).unique().tolist())
            inputs[column] = st.selectbox(
                column, options if options else ["Unknown"], key=f"single_{column}"
            )
    if st.button("Predict", key="single_predict"):
        try:
            prediction = pipeline.predict(pd.DataFrame([inputs]))[0]
            st.success(f"Predicted {st.session_state['target']}: {prediction}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    st.subheader("Batch predictions")
    batch_file = st.file_uploader(
        "Upload a CSV or Excel file with the same feature columns",
        type=["csv", "xlsx", "xls"], key="batch_file",
    )
    if batch_file:
        try:
            batch = read_uploaded_file(batch_file)
            missing = [c for c in st.session_state["features"] if c not in batch.columns]
            if missing:
                st.error("Missing required feature columns: " + ", ".join(missing))
            else:
                output = batch.copy()
                output["prediction"] = pipeline.predict(batch[st.session_state["features"]])
                st.dataframe(output.head(25), use_container_width=True)
                st.download_button(
                    "Download predictions (CSV)", output.to_csv(index=False),
                    file_name="predictions.csv", mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Could not process the batch file: {exc}")
else:
    st.info("Train at least one model successfully to enable predictions.")

st.divider()
st.caption("ML Workflow Studio · Review model metrics before using predictions in real decisions.")
