import streamlit as st
import pandas as pd
import joblib

from preprocess import build_feature_df_v2
from classifier import SeranggaClassifier

st.title("Insect Sound Classification")

# =========================
# Load model (cached)
# =========================
@st.cache_resource
def load_model():
    return SeranggaClassifier({
        'rf' : joblib.load('./random_forest.pkl'),
        'svc' : joblib.load('./svc.pkl')
    })

classifier = load_model()

# =========================
# Upload data
# =========================
uploaded_file = st.file_uploader(
    "Upload CSV (1 baris = 1 sinyal audio)",
    type=["csv"]
)

if uploaded_file is not None:
    # Load raw signal
    df_raw = pd.read_csv(uploaded_file)

    st.subheader("Raw Signal (Preview)")
    st.dataframe(df_raw.head())

    # =========================
    # Preprocess (PAKAI PUNYA KAMU)
    # =========================
    X = build_feature_df_v2(df_raw)

    st.subheader("Extracted Features (Preview)")
    st.dataframe(X.head())

    # =========================
    # Prediction
    # =========================
    y_pred = classifier.predict(X)

    df_result = pd.DataFrame({
        "prediction": y_pred
    })

    st.subheader("Prediction Result")
    st.dataframe(df_result)
