import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn import tree
 
# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
 
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)
 
# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
 
st.markdown("""
<style>
 
.main {
    background-color: #f5f7fa;
}
 
.title {
    font-size: 42px;
    font-weight: bold;
    color: #0e4c92;
    text-align: center;
}
 
.subtitle {
    font-size: 18px;
    text-align: center;
    color: gray;
}
 
.stButton>button {
    background-color: #0e4c92;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}
 
</style>
""", unsafe_allow_html=True)
 
# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
 
st.markdown(
    '<p class="title">🩺 Breast Cancer Prediction System</p>',
    unsafe_allow_html=True
)
 
st.markdown(
    '<p class="subtitle">Decision Tree Classification using Machine Learning</p>',
    unsafe_allow_html=True
)
 
st.markdown("---")
 
# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
 
cancer = load_breast_cancer()
 
X = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names
)
 
y = pd.Series(cancer.target)
 
# ---------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------
 
st.header("📂 Dataset Overview")
 
col1, col2 = st.columns(2)
 
with col1:
    st.subheader("Features Dataset (First 5 Rows)")
    st.dataframe(X.head())
 
with col2:
    st.subheader("Target Dataset (First 5 Rows)")
    st.dataframe(y.head().rename("diagnosis"))
 
st.write("### Dataset Shape")
st.write("Features Shape:", X.shape)
st.write("Target Shape:", y.shape)
 
# ---------------------------------------------------
# DATA VISUALIZATION - BOXPLOT
# ---------------------------------------------------
 
st.header("📊 Data Visualization")
 
st.subheader("Boxplot of First 10 Features")
 
fig1, ax1 = plt.subplots(figsize=(15, 6))
sns.boxplot(data=X.iloc[:, :10], ax=ax1)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)
 
# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------
 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
 
# ---------------------------------------------------
# MODEL CREATION & TRAINING
# ---------------------------------------------------
 
st.header("🤖 Decision Tree Model Training")
 
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    random_state=42
)
 
model.fit(X_train, y_train)
 
st.success("✅ Model trained successfully!")
 
# ---------------------------------------------------
# MODEL PREDICTIONS
# ---------------------------------------------------
 
y_pred = model.predict(X_test)
 
accuracy = accuracy_score(y_test, y_pred)
 
# ---------------------------------------------------
# MODEL PERFORMANCE METRICS
# ---------------------------------------------------
 
st.header("📈 Model Performance")
 
col1, col2, col3 = st.columns(3)
 
with col1:
    st.metric("Accuracy", f"{accuracy * 100:.2f}%")
 
with col2:
    st.metric("Training Samples", len(X_train))
 
with col3:
    st.metric("Testing Samples", len(X_test))
 
# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------
 
st.subheader("Confusion Matrix")
 
cm = confusion_matrix(y_test, y_pred)
 
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Malignant', 'Benign'],
    yticklabels=['Malignant', 'Benign'],
    ax=ax2
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)
 
# ---------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------
 
st.subheader("Classification Report")
 
report = classification_report(
    y_test,
    y_pred,
    target_names=['Malignant', 'Benign'],
    output_dict=True
)
 
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format("{:.2f}"))
 
# ---------------------------------------------------
# DECISION TREE VISUALIZATION
# ---------------------------------------------------
 
st.header("🌳 Decision Tree Visualization")
 
fig3, ax3 = plt.subplots(figsize=(20, 10))
 
tree.plot_tree(
    model,
    filled=True,
    feature_names=list(cancer.feature_names),
    class_names=list(cancer.target_names),
    fontsize=8,
    ax=ax3
)
 
plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)
 
# ---------------------------------------------------
# USER INPUT SECTION
# ---------------------------------------------------
 
st.header("🔍 Predict Breast Cancer")
 
st.write("Enter the feature values below (default values are dataset averages):")
 
input_data = []
 
col1, col2 = st.columns(2)
 
for i, feature in enumerate(cancer.feature_names):
 
    mean_value = float(X[feature].mean())
 
    if i % 2 == 0:
        with col1:
            value = st.number_input(
                label=str(feature),
                value=mean_value,
                format="%.4f",
                key=f"feature_{i}"
            )
    else:
        with col2:
            value = st.number_input(
                label=str(feature),
                value=mean_value,
                format="%.4f",
                key=f"feature_{i}"
            )
 
    input_data.append(value)
 
# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------
 
if st.button("🔬 Predict Cancer Type"):
 
    input_array = np.array(input_data).reshape(1, -1)
 
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)
 
    st.subheader("🎯 Prediction Result")
 
    if prediction[0] == 0:
        st.error("⚠️ Malignant Cancer Detected")
        st.write("The model predicts this sample is **Malignant** (Cancerous).")
    else:
        st.success("✅ Benign Cancer Detected")
        st.write("The model predicts this sample is **Benign** (Non-Cancerous).")
 
    st.write("### Prediction Probability")
 
    prob_df = pd.DataFrame({
        "Class": ["Malignant", "Benign"],
        "Probability": [f"{p:.4f}" for p in probability[0]],
        "Percentage": [f"{p * 100:.2f}%" for p in probability[0]]
    })
 
    st.dataframe(prob_df, use_container_width=True)
 
    # Probability bar chart
    fig4, ax4 = plt.subplots(figsize=(6, 3))
    bars = ax4.bar(
        ['Malignant', 'Benign'],
        probability[0],
        color=['#e74c3c', '#2ecc71'],
        edgecolor='white',
        linewidth=1.5
    )
    ax4.set_ylim(0, 1)
    ax4.set_ylabel("Probability")
    ax4.set_title("Prediction Probability")
    for bar, prob in zip(bars, probability[0]):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{prob:.2%}",
            ha='center',
            fontweight='bold'
        )
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)
 
# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
 
st.markdown("---")
 
st.markdown(
    """
    <center>
        <h4>Developed using Streamlit &amp; Scikit-Learn | Breast Cancer Dataset</h4>
    </center>
    """,
    unsafe_allow_html=True
)
 