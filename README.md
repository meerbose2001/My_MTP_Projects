
# Fake News Detection — Starter (TF‑IDF + Logistic Regression)

This starter is designed to get you from zero → a working **text-only** fake-news detector **today**.

## What you’ll build
- A fast baseline model using **TF‑IDF** features and **Logistic Regression**.
- Clean training script with evaluation + saved artifacts.
- Simple CLI predictor and an optional Streamlit mini‑app.

---

## 1) Install dependencies (Python 3.9+ recommended)

```bash
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install pandas scikit-learn matplotlib joblib streamlit
```

> If you see errors, try: `pip install --upgrade scikit-learn==1.5.1` (or latest).

---

## 2) Prepare your dataset

- CSV file with at least a **text** column and a **label** column.
- Common label names: `label`, `target`, `is_fake`, `y`. Values can be `fake/real`, `0/1`, etc.
- If you have both `title` and `text`, this script will concatenate them for better accuracy.

**Example:** See `data_sample.csv` (toy only). Replace it with your real data, e.g. `data/train.csv`.

---

## 3) Train the model

```bash
python train_fake_news.py --data data_sample.csv --text-cols title text --label-col label --val-size 0.2 --random-state 42
```

Key flags:
- `--data`: path to CSV.
- `--text-cols`: which text columns to combine (space‑separated).
- `--label-col`: the label column.
- `--val-size`: validation split size (0.2 = 20%).
- `--max-features`: TF‑IDF vocabulary size (default 50000).
- `--ngram-max`: use n‑grams up to this size (default 2).
- `--model`: `logreg` (default) or `linearsvc`.

Outputs (saved in `artifacts/`):
- `vectorizer.joblib` and `model.joblib`
- `label_map.json` (how labels were mapped to `0/1`)
- `report.txt` with metrics
- `confusion_matrix.png`

---

## 4) Predict from the saved model (CLI)

```bash
python predict.py "This just in: scientists discover water is wet."
python predict.py --batch texts.txt              # one example per line
```

---

## 5) Optional: run the mini app

```bash
streamlit run app_streamlit.py
```
Then open the local URL it prints.

---

## Tips to finish **today**
- Start with the baseline (default flags) — it trains in seconds on small/medium datasets.
- If classes are imbalanced, the scripts already use `class_weight='balanced'`.
- Bigger `--max-features` and `--ngram-max 3` can help on larger datasets.
- Clean noisy rows (empty text, duplicates) for quick gains.

---

## What to submit
- `artifacts/report.txt` (metrics)
- A screenshot of `confusion_matrix.png`
- Brief notes: features (TF‑IDF), model (LogReg), and key metrics.

Good luck — you’ve got this!
