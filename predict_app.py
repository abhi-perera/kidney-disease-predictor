import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="Kidney Disease Prediction",
    page_icon="🩺",
    layout="wide",
)

MODEL_PATH = "kidney_model.joblib"


@st.cache_resource
def load_model_from_path(path):
    return joblib.load(path)


@st.cache_resource
def load_model_from_bytes(file_bytes):
    return joblib.load(io.BytesIO(file_bytes))


def parse_float(value: str):
    value = str(value).strip()
    if value == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def parse_category(value: str):
    value = str(value).strip()
    if value == "":
        return np.nan
    return value


st.title("Kidney Disease Prediction")
st.caption("Enter patient details to predict whether the patient has CKD or not.")

# -------------------------------------------------
# Load model package
# -------------------------------------------------
model_package = None

if os.path.exists(MODEL_PATH):
    model_package = load_model_from_path(MODEL_PATH)
else:
    uploaded_model = st.file_uploader(
        "Upload trained model (.joblib)",
        type=["joblib"]
    )
    if uploaded_model is not None:
        model_package = load_model_from_bytes(uploaded_model.read())

if model_package is None:
    st.error(
        "Model file not found. Either place 'kidney_model.joblib' in the same folder as this app or upload it here."
    )
    st.stop()

model = model_package["model"]
label_encoder = model_package["label_encoder"]

# -------------------------------------------------
# Input Form
# -------------------------------------------------
st.subheader("Patient Input Form")

with st.form("prediction_form"):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        age = st.text_input("Age", "")
        bp = st.text_input("Blood Pressure (bp)", "")
        sg = st.selectbox("Specific Gravity (sg)", ["", "1.005", "1.010", "1.015", "1.020", "1.025"])
        al = st.selectbox("Albumin (al)", ["", "0", "1", "2", "3", "4", "5"])
        su = st.selectbox("Sugar (su)", ["", "0", "1", "2", "3", "4", "5"])
        rbc = st.selectbox("Red Blood Cells (rbc)", ["", "normal", "abnormal"])

    with c2:
        pc = st.selectbox("Pus Cell (pc)", ["", "normal", "abnormal"])
        pcc = st.selectbox("Pus Cell Clumps (pcc)", ["", "present", "notpresent"])
        ba = st.selectbox("Bacteria (ba)", ["", "present", "notpresent"])
        bgr = st.text_input("Blood Glucose Random (bgr)", "")
        bu = st.text_input("Blood Urea (bu)", "")
        sc = st.text_input("Serum Creatinine (sc)", "")

    with c3:
        sod = st.text_input("Sodium (sod)", "")
        pot = st.text_input("Potassium (pot)", "")
        hemo = st.text_input("Hemoglobin (hemo)", "")
        pcv = st.text_input("Packed Cell Volume (pcv)", "")
        wc = st.text_input("White Blood Cell Count (wc)", "")
        rc = st.text_input("Red Blood Cell Count (rc)", "")

    with c4:
        htn = st.selectbox("Hypertension (htn)", ["", "yes", "no"])
        dm = st.selectbox("Diabetes Mellitus (dm)", ["", "yes", "no"])
        cad = st.selectbox("Coronary Artery Disease (cad)", ["", "yes", "no"])
        appet = st.selectbox("Appetite (appet)", ["", "good", "poor"])
        pe = st.selectbox("Pedal Edema (pe)", ["", "yes", "no"])
        ane = st.selectbox("Anemia (ane)", ["", "yes", "no"])

    submitted = st.form_submit_button("Predict CKD")

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if submitted:
    input_data = pd.DataFrame(
        [
            {
                "age": parse_float(age),
                "bp": parse_float(bp),
                "sg": parse_float(sg),
                "al": parse_float(al),
                "su": parse_float(su),
                "rbc": parse_category(rbc),
                "pc": parse_category(pc),
                "pcc": parse_category(pcc),
                "ba": parse_category(ba),
                "bgr": parse_float(bgr),
                "bu": parse_float(bu),
                "sc": parse_float(sc),
                "sod": parse_float(sod),
                "pot": parse_float(pot),
                "hemo": parse_float(hemo),
                "pcv": parse_float(pcv),
                "wc": parse_float(wc),
                "rc": parse_float(rc),
                "htn": parse_category(htn),
                "dm": parse_category(dm),
                "cad": parse_category(cad),
                "appet": parse_category(appet),
                "pe": parse_category(pe),
                "ane": parse_category(ane),
            }
        ]
    )

    st.subheader("Entered Values")
    st.dataframe(input_data, use_container_width=True)

    pred_encoded = model.predict(input_data)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    st.subheader("Prediction Result")

    normalized_pred = str(pred_label).strip().lower()

    if normalized_pred == "ckd":
        st.error("Prediction: The patient is likely to have CKD.")
    elif normalized_pred == "notckd":
        st.success("Prediction: The patient is likely NOT to have CKD.")
    else:
        st.info(f"Prediction: {pred_label}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]

        prob_data = []
        for i, prob in enumerate(probabilities):
            class_name = label_encoder.inverse_transform([i])[0]
            prob_data.append(
                {
                    "Class": class_name,
                    "Probability": round(float(prob), 4),
                    "Percentage": f"{float(prob) * 100:.2f}%",
                }
            )

        prob_df = pd.DataFrame(prob_data)

        st.subheader("Prediction Probabilities")
        st.dataframe(prob_df, use_container_width=True)

        best_idx = int(np.argmax(probabilities))
        best_class = label_encoder.inverse_transform([best_idx])[0]
        best_prob = float(probabilities[best_idx]) * 100

        st.write(f"Most likely class: **{best_class}**")
        st.write(f"Confidence: **{best_prob:.2f}%**")