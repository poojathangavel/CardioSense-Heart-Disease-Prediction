import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set Streamlit page configuration at the very top
st.set_page_config(page_title="CardioSense: Heart Disease Prediction", layout="wide")

# Load dataset
DATA_PATH = "D:\heart\heart.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# Prepare Data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# App UI
st.title("💖 CardioSense: Predictive Insights for Heart Health")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔍 Input Features")

    age = st.slider("Age", min_value=20, max_value=80, value=60)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.slider("Chest Pain Type (CP)", min_value=0, max_value=3, value=1)
    trestbps = st.slider("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.slider("Serum Cholesterol (mg/dl)", min_value=100, max_value=400, value=230)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.slider("Resting ECG Results", min_value=0, max_value=2, value=1)
    thalach = st.slider("Max Heart Rate Achieved", min_value=60, max_value=200, value=140)
    exang = st.selectbox("Exercise-Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.slider("ST Depression Induced", min_value=0.0, max_value=5.0, value=1.5)
    slope = st.slider("Slope of ST Segment", min_value=0, max_value=2, value=1)
    ca = st.slider("Number of Major Vessels", min_value=0, max_value=4, value=0)
    thal = st.slider("Thalassemia Type", min_value=0, max_value=3, value=2)

    new_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                            columns=X.columns)

    if st.button("Predict"):
        prediction = clf.predict(new_data)[0]
        st.session_state["prediction"] = "🔴 Heart Disease: Positive" if prediction == 1 else "🟢 No Heart Disease Detected"

with col2:
    st.subheader("📝 User Input Features")
    st.dataframe(new_data)

    st.subheader("📢 Prediction")
    if "prediction" in st.session_state:
        st.write(f"### {st.session_state['prediction']}")




