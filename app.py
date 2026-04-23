
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Heart Risk AI", page_icon="❤️", layout="wide")

# ------------------ STYLE ------------------
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Cards / containers */
.block-container {
    padding: 2rem;
    border-radius: 15px;
    background-color: rgba(0,0,0,0.4);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

/* Inputs */
label, .stSlider, .stSelectbox {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.title("❤️ Heart Disease Risk AI System")
st.markdown("### Smart AI-based Health Prediction & Analysis")

# ------------------ LOAD + TRAIN ------------------
df = pd.read_csv('heart_hci.csv')

df = df.drop(['id', 'ca', 'thal', 'slope'], axis=1)
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df = df.drop('dataset', axis=1)
df = pd.get_dummies(df, drop_first=True)
df = df.fillna(df.mean())

X = df.drop('num', axis=1)
y = df['num']

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# ------------------ INPUT UI ------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧾 Patient Details")
    age = st.slider("Age", 20, 100, 50)
    trestbps = st.slider("Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 400, 200)
    thalch = st.slider("Max Heart Rate", 60, 220, 150)
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)

with col2:
    st.subheader("🩺 Medical Info")
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain", ["typical angina", "atypical angina", "non-anginal"])
    fbs = st.selectbox("High Blood Sugar", [True, False])
    restecg = st.selectbox("ECG", ["normal", "st-t abnormality"])
    exang = st.selectbox("Exercise Angina", [True, False])

# ------------------ PREP INPUT ------------------
input_dict = {
    'age': age,
    'trestbps': trestbps,
    'chol': chol,
    'thalch': thalch,
    'oldpeak': oldpeak,
    'sex_Male': 1 if sex == "Male" else 0,
    'cp_atypical angina': 1 if cp == "atypical angina" else 0,
    'cp_non-anginal': 1 if cp == "non-anginal" else 0,
    'cp_typical angina': 1 if cp == "typical angina" else 0,
    'fbs_True': 1 if fbs else 0,
    'restecg_normal': 1 if restecg == "normal" else 0,
    'restecg_st-t abnormality': 1 if restecg == "st-t abnormality" else 0,
    'exang_True': 1 if exang else 0
}

input_df = pd.DataFrame([input_dict])

# ------------------ PREDICTION ------------------
if st.button("🔍 Predict Risk"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.markdown("---")

    # Result
    if prediction == 1:
        st.error(f"⚠️ High Risk ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk ({prob*100:.2f}%)")

    # Risk meter
    st.subheader("📊 Risk Level")
    st.progress(int(prob * 100))

    # Smart insights
    st.subheader("🧠 AI Insight")
    if prob > 0.7:
        st.warning("High probability of heart disease. Immediate medical consultation recommended.")
    elif prob > 0.4:
        st.info("Moderate risk. Maintain diet, exercise, and monitor health.")
    else:
        st.success("Low risk. Keep maintaining a healthy lifestyle!")

    # ------------------ FEATURE IMPORTANCE ------------------
    st.subheader("📊 Important Factors")

    importances = model.feature_importances_
    features = X.columns

    feat_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(8)

    fig, ax = plt.subplots()
    ax.barh(feat_df['Feature'], feat_df['Importance'])
    ax.invert_yaxis()
    st.pyplot(fig)

    # ------------------ DOWNLOAD REPORT ------------------
    st.subheader("💾 Download Report")

    report = f"""
    Heart Disease Risk Report

    Age: {age}
    Blood Pressure: {trestbps}
    Cholesterol: {chol}

    Risk Probability: {prob*100:.2f}%
    Result: {"High Risk" if prediction==1 else "Low Risk"}
    """

    st.download_button("Download Report", report, file_name="heart_report.txt")
