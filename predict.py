
import argparse, json
from joblib import load
import numpy as np

def load_artifacts(path="artifacts"):
    vectorizer = load(f"{path}/vectorizer.joblib")
    model = load(f"{path}/model.joblib")
    with open(f"{path}/label_map.json","r") as f:
        label_map = json.load(f)
    inv_label = {int(v): k for k,v in label_map.items()}
    return vectorizer, model, inv_label

def predict_one(text, vectorizer, model, inv_label):
    X = vectorizer.transform([text])
    pred = int(model.predict(X)[0])
    label = inv_label.get(pred, str(pred))
    return pred, label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default=None, help="Text to classify. If omitted, use --batch.")
    ap.add_argument("--batch", default=None, help="Path to a text file: one example per line.")
    ap.add_argument("--artifacts", default="artifacts", help="Artifacts folder.")
    args = ap.parse_args()

    vectorizer, model, inv_label = load_artifacts(args.artifacts)

    if args.batch:
        with open(args.batch, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for i, line in enumerate(lines, 1):
            pred, label = predict_one(line, vectorizer, model, inv_label)
            print(f"[{i}] -> {label} ({pred}) : {line[:80]}{'...' if len(line)>80 else ''}")
    else:
        if not args.text:
            print("Provide text or --batch file.")
            return
        pred, label = predict_one(args.text, vectorizer, model, inv_label)
        print(f"Prediction: {label} ({pred})")

if __name__ == "__main__":
    main()
