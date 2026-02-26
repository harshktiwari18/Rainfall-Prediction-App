import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rainfall Dashboard", page_icon="🌧️", layout="wide")

# Premium UI Styling
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg,#667eea,#764ba2);
}

.block-container {
    background:white;
    padding:40px;
    border-radius:15px;
    margin-top:20px;
    box-shadow:0px 8px 20px rgba(0,0,0,0.25);
}

.result-rain {
    background:#d4edda;
    padding:20px;
    border-radius:10px;
    font-size:22px;
    text-align:center;
    color:#155724;
}

.result-sun {
    background:#fff3cd;
    padding:20px;
    border-radius:10px;
    font-size:22px;
    text-align:center;
    color:#856404;
}

.stButton>button {
    background: linear-gradient(to right,#667eea,#764ba2);
    color:white;
    border-radius:8px;
    height:3em;
    width:100%;
}
</style>
""", unsafe_allow_html=True)

# Load ML model
with open("model.pkl","rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_names = model_data["feature_names"]

st.title("🌧️ Rainfall Prediction Dashboard")
st.write("Enter weather parameters to predict rainfall.")

# Input layout
col1, col2 = st.columns(2)

with col1:
    pressure = st.number_input("Pressure",1015.9)
    dewpoint = st.number_input("Dew Point",19.9)
    humidity = st.number_input("Humidity",95)
    cloud = st.number_input("Cloud",81)

with col2:
    sunshine = st.number_input("Sunshine",0.0)
    winddirection = st.number_input("Wind Direction",40)
    windspeed = st.number_input("Wind Speed",13.7)

# Prediction
if st.button("Predict Rainfall"):

    input_df = pd.DataFrame(
        [[pressure, dewpoint, humidity, cloud,
          sunshine, winddirection, windspeed]],
        columns=feature_names
    )

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.markdown(
            "<div class='result-rain'>🌧️ Rainfall Expected</div>",
            unsafe_allow_html=True
        )
        st.balloons()
    else:
        st.markdown(
            "<div class='result-sun'>☀️ No Rainfall Expected</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ⭐ Prediction Probability Chart
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]

        fig = plt.figure()
        plt.bar(["No Rain", "Rain"], probs)
        plt.title("Prediction Confidence")
        plt.ylabel("Probability")
        st.pyplot(fig)

    # ⭐ Feature Importance Chart (RandomForest only)
    if hasattr(model, "feature_importances_"):
        fig2 = plt.figure()
        plt.barh(feature_names, model.feature_importances_)
        plt.title("Feature Importance")
        st.pyplot(fig2)