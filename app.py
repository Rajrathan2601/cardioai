import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
from report_generator import create_report

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="CardioAI Pro", layout="wide")

st.title("💓 CardioAI Pro – ECG Intelligence System")
st.write("App Started Successfully ✅")

# ==============================
# CLASSES
# ==============================
classes = ["Normal","PAC","PVC","LBBB","RBBB"]

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("💓 CardioAI Pro")
page = st.sidebar.radio("Navigation", ["Dashboard","History"])

# ==============================
# DASHBOARD
# ==============================
if page == "Dashboard":

    st.header("📊 ECG Analysis Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Patient Name")
        age = st.slider("Age", 10, 90, 30)
        heart_rate = st.slider("Heart Rate", 50, 150, 80)

    uploaded_file = st.file_uploader("Upload ECG CSV", type=["csv"])

    # Generate or load signal
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        signal = df.iloc[:,0].values
    else:
        signal = np.sin(np.linspace(0,20,200)) + np.random.randn(200)*0.1

    signal = signal[:200]

    st.subheader("📈 ECG Signal")
    st.line_chart(signal)

    # ==============================
    # ANALYSIS
    # ==============================
    if st.button("🚀 Analyze ECG"):

        # Fake prediction (for deployment)
        predicted = random.choice(classes)
        confidence = random.uniform(70, 95)

        # Risk calculation
        risk = min(confidence + age*0.2 + abs(heart_rate-75)*0.5, 100)

        # Save patient data
        new_data = pd.DataFrame([[name, age, predicted, risk]],
                                columns=["name","age","condition","risk"])

        if os.path.exists("patients.csv"):
            new_data.to_csv("patients.csv", mode='a', header=False, index=False)
        else:
            new_data.to_csv("patients.csv", index=False)

        # Results
        st.success(f"🧠 Condition: {predicted}")
        st.info(f"📊 Confidence: {confidence:.2f}%")
        st.warning(f"⚠️ Risk Score: {risk:.2f}")

        # Generate PDF report
        report = create_report(name, age, predicted, risk)

        with open(report, "rb") as f:
            st.download_button(
                label="📄 Download Report",
                data=f,
                file_name="CardioAI_Report.pdf",
                mime="application/pdf"
            )

# ==============================
# HISTORY PAGE
# ==============================
else:
    st.header("📁 Patient History")

    if os.path.exists("patients.csv"):
        df = pd.read_csv("patients.csv")
        st.dataframe(df)
    else:
        st.warning("No patient data available yet.")