import streamlit as st

st.write("App Started Successfully")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from report_generator import create_report
import os

st.set_page_config(page_title="CardioAI Pro", layout="wide")

@st.cache_resource
def load_my_model():
    return load_model("ecg_model.h5")

model = load_my_model()

classes = ["Normal","PAC","PVC","LBBB","RBBB"]

# SIDEBAR
st.sidebar.title("💓 CardioAI Pro")
page = st.sidebar.radio("Navigation", ["Dashboard","History"])

# DASHBOARD
if page == "Dashboard":

    st.title("CardioAI Pro Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Patient Name")
        age = st.slider("Age", 10, 90, 30)
        heart_rate = st.slider("Heart Rate", 50, 150, 80)

    uploaded_file = st.file_uploader("Upload ECG CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        signal = df.iloc[:,0].values
    else:
        signal = np.sin(np.linspace(0,20,200))

    signal = signal[:200]

    st.line_chart(signal)

    if st.button("Analyze ECG"):

        sample = signal.reshape(1,200,1)

        pred = model.predict(sample)
        idx = np.argmax(pred)

        predicted = classes[idx]
        confidence = np.max(pred)*100

        risk = min(confidence + age*0.2 + abs(heart_rate-75)*0.5, 100)

        # Save data
        new_data = pd.DataFrame([[name, age, predicted, risk]],
                                columns=["name","age","condition","risk"])

        if os.path.exists("patients.csv"):
            new_data.to_csv("patients.csv", mode='a', header=False, index=False)
        else:
            new_data.to_csv("patients.csv", index=False)

        st.success(f"Condition: {predicted}")
        st.warning(f"Risk Score: {risk:.2f}")

        # Generate report
        report = create_report(name, age, predicted, risk)

        with open(report, "rb") as f:
         st.download_button(
            label="📄 Download Report",
            data=f,
            file_name="CardioAI_Report.pdf",
            mime="application/pdf"
         )
# HISTORY PAGE
else:
    st.title("Patient History")

    if os.path.exists("patients.csv"):
        df = pd.read_csv("patients.csv")
        st.dataframe(df)
    else:
        st.warning("No data yet")