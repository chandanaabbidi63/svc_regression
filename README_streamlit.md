# Streamlit App

This repo contains a Streamlit UI for the breast cancer classifier.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL shown in the terminal (typically http://localhost:8501).

## Notes
- Model training is cached via `st.cache_resource` on first run.
- UI lets you input the 29 numeric features and get a prediction.

