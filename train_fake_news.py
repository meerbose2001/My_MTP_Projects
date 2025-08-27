
import argparse, json, os, sys, re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch
from joblib import dump
import matplotlib.pyplot as plt

def read_csv_any(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        sys.exit(1)

def infer_label_mapping(series: pd.Series):
    # Map labels to {0,1}. Try to infer which is "fake".
    unique = sorted(series.dropna().unique().tolist())
    mapping = {}

    # If numeric 0/1
    if set(unique).issubset({0, 1}):
        mapping = {0: 0, 1: 1}
    elif set(unique).issubset({"fake","real","FAKE","REAL","Fake","Real"}):
        mapping = {"fake":1,"FAKE":1,"Fake":1, "real":0,"REAL":0,"Real":0}
    elif set(unique).issubset({"true","false","True","False"}):
        mapping = {"true":0,"True":0,"false":1,"False":1}
    else:
        # Default: first distinct == 0, second == 1 (documented in label_map.json)
        mapping = {unique[0]: 0}
        if len(unique) > 1:
            mapping[unique[1]] = 1
        # If more than 2 classes, try to collapse by string match
        if len(unique) > 2:
            # Heuristic: if a value contains "fake" or "false" -> 1
            mapping = {}
            for u in unique:
                s = str(u).lower()
                mapping[u] = 1 if ("fake" in s or "false" in s) else 0

    return mapping

def combine_text(df: pd.DataFrame, text_cols):
    parts = []
    for col in text_cols:
        if col not in df.columns:
            print(f"[WARN] Text column '{col}' not in data; skipping.")
            continue
        parts.append(df[col].astype(str).fillna(""))
    if not parts:
        raise ValueError("No valid text columns found.")
    combined = parts[0]
    for p in parts[1:]:
        combined = combined.str.cat(p, sep=" ")
    # Basic normalization
    combined = combined.str.replace(r"\s+", " ", regex=True).str.strip()
    return combined

def plot_confusion(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig = plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.xticks([0,1], ['real(0)','fake(1)'])
    plt.yticks([0,1], ['real(0)','fake(1)'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV dataset")
    ap.add_argument("--text-cols", nargs="+", default=["text"], help="Text columns to concatenate (space-separated)")
    ap.add_argument("--label-col", default="label", help="Label column name")
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--max-features", type=int, default=50000)
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument("--model", choices=["logreg","linearsvc"], default="logreg")
    args = ap.parse_args()

    os.makedirs("artifacts", exist_ok=True)

    df = read_csv_any(args.data)
    if args.label_col not in df.columns:
        print(f"[ERROR] Label column '{args.label_col}' not found. Available: {list(df.columns)[:20]}")
        sys.exit(1)

    X_text = combine_text(df, args.text_cols)
    y_raw = df[args.label_col]

    label_map = infer_label_mapping(y_raw)
    y = y_raw.map(label_map)
    if y.isna().any():
        print("[WARN] Some labels could not be mapped; dropping those rows.")
    mask = ~y.isna() & X_text.notna() & (X_text.str.len() > 0)
    X_text = X_text[mask]
    y = y[mask].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X_text, y, test_size=args.val_size, random_state=args.random_state, stratify=y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        strip_accents="unicode"
    )

    if args.model == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None)
    else:
        clf = LinearSVC(class_weight="balanced")

    # Fit
    Xtr = vectorizer.fit_transform(X_train)
    clf.fit(Xtr, y_train)

    # Validate
    Xva = vectorizer.transform(X_val)
    y_pred = clf.predict(Xva)

    acc = accuracy_score(y_val, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)

    # Try ROC-AUC only if clf supports decision_function / predict_proba
    try:
        if hasattr(clf, "predict_proba"):
            auc = roc_auc_score(y_val, clf.predict_proba(Xva)[:,1])
        elif hasattr(clf, "decision_function"):
            auc = roc_auc_score(y_val, clf.decision_function(Xva))
        else:
            auc = None
    except Exception:
        auc = None

    report_str = classification_report(y_val, y_pred, target_names=["real(0)","fake(1)"])
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    if auc is not None:
        print("ROC-AUC:", auc)
    print(report_str)

    # Save artifacts
    dump(vectorizer, "artifacts/vectorizer.joblib")
    dump(clf, "artifacts/model.joblib")
    with open("artifacts/label_map.json","w") as f:
        json.dump({str(k): int(v) for k,v in label_map.items()}, f, indent=2)

    with open("artifacts/report.txt","w") as f:
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n")
        if auc is not None:
            f.write(f"ROC-AUC: {auc:.4f}\n")
        f.write("\n")
        f.write(report_str)

    plot_confusion(y_val, y_pred, labels=[0,1], out_path="artifacts/confusion_matrix.png")

    print("\nSaved artifacts to ./artifacts")
    print(" - artifacts/vectorizer.joblib")
    print(" - artifacts/model.joblib")
    print(" - artifacts/label_map.json")
    print(" - artifacts/report.txt")
    print(" - artifacts/confusion_matrix.png")

if __name__ == "__main__":
    main()
