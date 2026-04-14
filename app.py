import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

st.set_page_config(
    page_title="Kidney Disease Classification Model Trainer",
    page_icon="🩺",
    layout="wide",
)

st.title("Kidney Disease Classification Model Trainer")
st.caption(
    "Upload your dataset, select the target column, train the model, and download the best trained model."
)


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def load_dataset(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if file_name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Please upload a CSV or XLSX file.")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [str(col).strip() for col in df.columns]

    missing_tokens = ["?", "NA", "N/A", "na", "null", "None", "none", ""]
    df.replace(missing_tokens, np.nan, inplace=True)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(
                {
                    "nan": np.nan,
                    "None": np.nan,
                    "none": np.nan,
                    "": np.nan,
                }
            )

    return df


def convert_numeric_like_columns(df: pd.DataFrame, min_ratio: float = 0.80) -> pd.DataFrame:
    """
    Convert object columns to numeric if most non-null values look numeric.
    This helps columns like pcv, wc, rc, sg, al, su when they are read as text.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            non_null = df[col].dropna().astype(str).str.strip()
            if non_null.empty:
                continue

            converted = pd.to_numeric(non_null, errors="coerce")
            numeric_ratio = converted.notna().mean()

            if numeric_ratio >= min_ratio:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def normalize_target(series: pd.Series) -> pd.Series:
    """
    Normalize target labels so values like 'ckd', 'ckd\\t', ' CKD ' become the same.
    """
    series = (
        series.astype(str)
        .str.lower()
        .str.replace(r"\s+", "", regex=True)
        .replace(
            {
                "nan": np.nan,
                "none": np.nan,
                "": np.nan,
            }
        )
    )
    return series


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def render_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return acc, pre, rec, f1


# -------------------------------------------------
# UI
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    try:
        raw_df = load_dataset(uploaded_file)
        df = clean_dataframe(raw_df)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.write(f"Rows: **{df.shape[0]}**")
    st.write(f"Columns: **{df.shape[1]}**")

    if df.shape[1] < 2:
        st.error("Your dataset must contain at least 2 columns.")
        st.stop()

    possible_target_index = 0
    for i, col in enumerate(df.columns):
        if col.strip().lower() in ["classification", "class", "target", "label", "kidney_disease"]:
            possible_target_index = i
            break

    col1, col2 = st.columns(2)

    with col1:
        target_column = st.selectbox(
            "Select the target column",
            options=df.columns.tolist(),
            index=possible_target_index,
        )

    auto_default_drop = [
        c for c in df.columns
        if c != target_column and c.strip().lower() in ["id", "index", "unnamed: 0"]
    ]

    with col2:
        drop_columns = st.multiselect(
            "Optional: columns to drop before training",
            options=[c for c in df.columns if c != target_column],
            default=auto_default_drop,
        )

    col3, col4 = st.columns(2)

    with col3:
        test_size = st.slider(
            "Test size",
            min_value=0.10,
            max_value=0.40,
            value=0.20,
            step=0.05,
        )

    with col4:
        random_state = st.number_input(
            "Random state",
            min_value=1,
            max_value=9999,
            value=42,
            step=1,
        )

    train_button = st.button("Train Model")

    if train_button:
        try:
            work_df = df.copy()

            # Drop user-selected columns
            if drop_columns:
                work_df = work_df.drop(columns=drop_columns)

            # Auto-drop leakage columns if still present
            leakage_columns = [
                c for c in work_df.columns
                if c != target_column and c.strip().lower() in ["id", "index", "unnamed: 0"]
            ]
            if leakage_columns:
                work_df = work_df.drop(columns=leakage_columns)

            if target_column not in work_df.columns:
                st.error("The selected target column is missing after dropping columns.")
                st.stop()

            # Normalize target labels
            work_df[target_column] = normalize_target(work_df[target_column])

            # Remove rows with missing target
            work_df = work_df[work_df[target_column].notna()].copy()

            if work_df.empty:
                st.error("No rows remain after removing missing target values.")
                st.stop()

            X = work_df.drop(columns=[target_column]).copy()
            y_raw = work_df[target_column].copy()

            if X.shape[1] == 0:
                st.error("No feature columns remain.")
                st.stop()

            # Clean feature values again
            for col in X.columns:
                if X[col].dtype == "object":
                    X[col] = X[col].astype(str).str.strip()
                    X[col] = X[col].replace(
                        {
                            "nan": np.nan,
                            "None": np.nan,
                            "none": np.nan,
                            "": np.nan,
                            "?": np.nan,
                        }
                    )

            # Convert numeric-like text columns to numeric
            X = convert_numeric_like_columns(X)

            # Encode target labels
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_raw)
            class_names = label_encoder.classes_.tolist()
            all_labels = list(range(len(class_names)))

            if len(class_names) < 2:
                st.error("The target column must contain at least 2 classes for classification.")
                st.stop()

            if len(class_names) > 20:
                st.warning(
                    "The target column has many unique classes. "
                    "Please make sure you selected the correct classification target."
                )

            st.write("Detected target classes:", class_names)

            preprocessor, numeric_features, categorical_features = build_preprocessor(X)

            stratify_value = y if len(np.unique(y)) > 1 else None

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=int(random_state),
                stratify=stratify_value,
            )

            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced"
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=300,
                    random_state=int(random_state),
                    class_weight="balanced",
                ),
            }

            results = []
            trained_pipelines = {}

            for model_name, classifier in models.items():
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("classifier", classifier),
                    ]
                )

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                acc, pre, rec, f1 = render_metrics(y_test, y_pred)

                results.append(
                    {
                        "Model": model_name,
                        "Accuracy": round(acc, 4),
                        "Precision (weighted)": round(pre, 4),
                        "Recall (weighted)": round(rec, 4),
                        "F1 Score (weighted)": round(f1, 4),
                    }
                )

                trained_pipelines[model_name] = pipeline

            results_df = pd.DataFrame(results).sort_values(
                by="F1 Score (weighted)",
                ascending=False
            ).reset_index(drop=True)

            best_model_name = results_df.iloc[0]["Model"]
            best_model = trained_pipelines[best_model_name]
            best_y_pred = best_model.predict(X_test)

            st.success(f"Training completed. Best model: {best_model_name}")

            st.subheader("Model Comparison")
            st.dataframe(results_df, use_container_width=True)

            st.subheader("Classification Report - Best Model")
            report_dict = classification_report(
                y_test,
                best_y_pred,
                labels=all_labels,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df, use_container_width=True)

            st.subheader("Confusion Matrix - Best Model")
            cm = confusion_matrix(
                y_test,
                best_y_pred,
                labels=all_labels,
            )
            cm_df = pd.DataFrame(
                cm,
                index=[f"Actual: {c}" for c in class_names],
                columns=[f"Predicted: {c}" for c in class_names],
            )
            st.dataframe(cm_df, use_container_width=True)

            st.subheader("Train/Test Shapes")
            st.write(f"X_train: {X_train.shape}")
            st.write(f"X_test: {X_test.shape}")

            # Save one sample input row for later prediction testing
            sample_input = X.head(1).copy()

            # Save model package
            model_package = {
                "model": best_model,
                "target_column": target_column,
                "feature_columns": X.columns.tolist(),
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "class_names": class_names,
                "label_encoder": label_encoder,
                "best_model_name": best_model_name,
                "metrics": results_df.to_dict(orient="records"),
            }

            model_buffer = io.BytesIO()
            joblib.dump(model_package, model_buffer)
            model_buffer.seek(0)

            sample_buffer = io.StringIO()
            sample_input.to_csv(sample_buffer, index=False)

            st.subheader("Download Files")

            st.download_button(
                label="Download trained model (.joblib)",
                data=model_buffer,
                file_name="kidney_model.joblib",
                mime="application/octet-stream",
            )

            st.download_button(
                label="Download sample input (.csv)",
                data=sample_buffer.getvalue(),
                file_name="sample_input.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Training failed: {e}")

else:
    st.info("Upload your dataset to begin.")