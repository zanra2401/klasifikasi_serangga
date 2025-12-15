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
    # Try to get probabilities
    y_proba = None
    try:
        y_proba = classifier.predict_proba(X)
    except Exception:
        y_proba = None

    df_result = pd.DataFrame({
        "prediction": y_pred
    })

    st.subheader("Prediction Result")
    st.dataframe(df_result)

    # =========================
    # Prediction Statistics
    # =========================
    st.subheader("Prediction Statistics")
    # Summary counts and percentages
    import numpy as np
    unique_pred, counts = np.unique(y_pred, return_counts=True)
    summary_df = pd.DataFrame({
        "class": unique_pred,
        "count": counts,
        "percent": (counts / len(y_pred) * 100).round(2)
    }).sort_values("count", ascending=False)
    st.dataframe(summary_df, use_container_width=True)
    st.bar_chart(summary_df.set_index("class")["percent"])
    st.markdown("Grafik di atas menunjukkan persentase prediksi per kelas dari seluruh data yang diunggah.")

    # If probabilities available, show top-3 for first 5 samples
    if y_proba is not None:
        st.markdown("Top-3 class probabilities (first 5 samples)")
        top_rows = min(5, len(y_pred))
        topn = []
        # Derive class order if available from classifier
        classes = None
        try:
            classes = classifier.classes_
        except Exception:
            classes = unique_pred
        for i in range(top_rows):
            probs = np.array(y_proba[i]).ravel()
            idx = np.argsort(probs)[::-1][:3]
            topn.append({
                "sample": i,
                "top1": f"{classes[idx[0]]}: {probs[idx[0]]:.2f}",
                "top2": f"{classes[idx[1]]}: {probs[idx[1]]:.2f}",
                "top3": f"{classes[idx[2]]}: {probs[idx[2]]:.2f}",
            })
        st.dataframe(pd.DataFrame(topn), use_container_width=True)

    # Evaluation/accuracy section removed per request; focusing on class percentage chart.
