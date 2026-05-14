# Breast Cancer Classifier (Streamlit)

This project provides a Streamlit app to predict **Benign vs Malignant** using a simple **DecisionTreeClassifier** trained on scikit-learn’s breast cancer dataset.

## Files
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- The app trains the model on first run and caches it via `st.cache_resource`.
- The dataset used is `sklearn.datasets.load_breast_cancer()` (29 numerical features).

