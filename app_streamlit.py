
import json
from joblib import load
import streamlit as st

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

@st.cache_resource
def load_artifacts(path="artifacts"):
    vectorizer = load(f"{path}/vectorizer.joblib")
    model = load(f"{path}/model.joblib")
    with open(f"{path}/label_map.json","r") as f:
        label_map = json.load(f)
    inv_label = {int(v): k for k,v in label_map.items()}
    return vectorizer, model, inv_label

st.title("📰 Fake News Detector (TF‑IDF + LogReg)")
st.write("Type or paste an article. This is a quick demo — not production truth verification.")

vectorizer, model, inv_label = load_artifacts()

text = st.text_area("Article text (you can paste title + text):", height=220)
btn = st.button("Classify")

if btn and text.strip():
    X = vectorizer.transform([text])
    pred = int(model.predict(X)[0])
    label = inv_label.get(pred, str(pred))
    st.subheader("Prediction")
    st.write(f"**{label}**  (class {pred})")
