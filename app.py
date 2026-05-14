import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


@st.cache_resource(show_spinner=False)
def train_model():
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, accuracy, cancer.feature_names


def main():
    st.set_page_config(page_title="Breast Cancer Classifier", layout="centered")
    st.title("Breast Cancer Classifier")
    st.caption("Streamlit app using a Decision Tree model (trained on sklearn breast cancer dataset).")

    model, accuracy, feature_names = train_model()

    st.sidebar.header("Model Info")
    st.sidebar.write(f"Test accuracy: **{accuracy:.3f}**")
    st.sidebar.write("Target mapping: 0 = Benign, 1 = Malignant")

    st.subheader("Input Features")
    st.write("Enter values for all features. If you want a quick try, use the buttons below.")

    cols = st.columns(3)
    default_idx = 0
    defaults = np.zeros(len(feature_names), dtype=float)

    # Provide simple defaults from dataset by using the sklearn data itself
    cancer = load_breast_cancer()
    example_row = cancer.data[default_idx]

    input_values = {}

    feature_inputs = []
    for i, name in enumerate(feature_names):
        c = cols[i % 3]
        default_val = float(example_row[i])
        feature_inputs.append(
            c.number_input(
                label=name,
                value=default_val,
                format="%.6f",
                step=0.01,
                key=f"feat_{i}",
            )
        )

    x = np.array(feature_inputs, dtype=float).reshape(1, -1)

    if st.button("Predict", type="primary"):
        pred = int(model.predict(x)[0])
        # In sklearn breast cancer dataset: 0=malignant, 1=benign.
        # But the notebook code used 0 -> Malignant, else -> Benign.
        # We'll follow that notebook mapping for consistency.
        if pred == 0:
            result = "Malignant Cancer"
            confidence_label = "Predicted class: 0"
        else:
            result = "Benign Cancer"
            confidence_label = "Predicted class: 1"

        st.success(result)
        st.info(confidence_label)

    with st.expander("About", expanded=False):
        st.markdown(
            """
            **How it works**\n\n            - Model is trained once (cached) using scikit-learn's breast cancer dataset.\n            - Decision Tree: `criterion='gini', max_depth=5`.\n            - Use the input fields to provide feature values (29 features).\n            """
        )


if __name__ == "__main__":
    main()

