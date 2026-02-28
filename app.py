import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Steel Column Base Surrogate", layout="centered")

@st.cache_resource
def load_assets():
    base = Path(__file__).parent
    model_M = joblib.load(base / "models" / "model_M.joblib")
    model_logS = joblib.load(base / "models" / "model_logS.joblib")
    with open(base / "models" / "meta.json", "r") as f:
        meta = json.load(f)
    return model_M, model_logS, meta

model_M, model_logS, meta = load_assets()
feature_cols = meta["feature_cols"]

st.title("Steel Column Base — Surrogate Predictor")
st.write("Inputs in mm. Outputs: Mj,Rd (kNm) and Sj,ini (MNm/rad).")

st.sidebar.header("Inputs")

h = st.sidebar.number_input("h (mm)", value=200.0)
b = st.sidebar.number_input("b (mm)", value=200.0)
tw = st.sidebar.number_input("tw (mm)", value=8.0)
tf = st.sidebar.number_input("tf (mm)", value=12.0)

steel = st.sidebar.selectbox("Steel grade (MPa)", [235, 275, 355])
bolt = st.sidebar.selectbox("Bolt grade", [8.8, 10.9])

t_bp = st.sidebar.number_input("t_bp (mm)", value=25.0)
l_bp = st.sidebar.number_input("l_bp (mm)", value=300.0)
b_bp = st.sidebar.number_input("b_bp (mm)", value=200.0)

x_values = {
    "h_mm": float(h),
    "b_mm": float(b),
    "tw_mm": float(tw),
    "tf_mm": float(tf),
    "Steel_numeric": float(steel),
    "Bolt_grade_numeric": float(bolt),
    "t_bp_mm": float(t_bp),
    "l_bp_mm": float(l_bp),
    "b_bp_mm": float(b_bp),
}

X_new = pd.DataFrame([x_values], columns=feature_cols)

if st.button("Predict"):
    M_pred = float(model_M.predict(X_new)[0])
    logS_pred = float(model_logS.predict(X_new)[0])
    S_pred = float(np.exp(logS_pred))

    col1, col2 = st.columns(2)
    col1.metric("Mj,Rd (kNm)", f"{M_pred:.1f}")
    col2.metric("Sj,ini (MNm/rad)", f"{S_pred:.2f}")
    
