import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

st.set_page_config(page_title="ML Classifier", layout="wide")
st.title("Breast Cancer Classification App")

st.markdown("Upload the ⁠ test data.csv ⁠ file to evaluate the models.")

# Upload CSV
uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head(3))
    
    if 'target' not in df.columns:
        st.error("The uploaded CSV must contain a 'target' column.")
    else:
        X_test = df.drop('target', axis=1)
        y_test = df['target']
        
        # Model Selection
        model_name = st.selectbox("Select a Model to Evaluate", [
            'Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Support Vector Machine'
        ])
        
        model_path = f"model/{model_name.replace(' ', '_').lower()}.pkl"
        
        if st.button("Run Evaluation"):
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                
                st.write(f"### Evaluation Metrics for {model_name}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 4))
                col2.metric("AUC Score", round(roc_auc_score(y_test, y_proba), 4))
                col3.metric("Precision", round(precision_score(y_test, y_pred), 4))
                
                col4, col5, col6 = st.columns(3)
                col4.metric("Recall", round(recall_score(y_test, y_pred), 4))
                col5.metric("F1 Score", round(f1_score(y_test, y_pred), 4))
                col6.metric("MCC Score", round(matthews_corrcoef(y_test, y_pred), 4))
                
                st.write("### Confusion Matrix")
                st.dataframe(confusion_matrix(y_test, y_pred))
                
                st.write("### Classification Report")
                st.text(classification_report(y_test, y_pred))
            else:
                st.error("Model file not found. Ensure models are trained and saved in the 'model/' directory.")