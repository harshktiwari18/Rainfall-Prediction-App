import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Rainfall Dashboard", page_icon="🌧️", layout="wide")

# ================== CSS ==================
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#567eea,#764ba2);
}

.block-container {
    background: transparent;
    padding:40px;
    border-radius:20px;
}

</style>
""", unsafe_allow_html=True)

# ================== LOAD MODEL ==================
with open("model.pkl","rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_names = model_data["feature_names"]

# ================== TITLE ==================
st.title("🌧️ Rainfall Prediction Dashboard")
st.write("Enter weather parameters to predict rainfall.")

# ================== INPUT ==================
col1, col2 = st.columns(2)

with col1:
    pressure = st.number_input("Pressure", value=1015.9)
    dewpoint = st.number_input("Dew Point", value=19.9)
    humidity = st.number_input("Humidity", value=95)
    cloud = st.number_input("Cloud", value=81)

with col2:
    sunshine = st.number_input("Sunshine", value=0.0)
    winddirection = st.number_input("Wind Direction", value=40)
    windspeed = st.number_input("Wind Speed", value=13.7)

st.markdown("")

# ================== BUTTON ==================
col1,col2,col3 = st.columns([1,2,1])

with col2:
    predict = st.button("Predict Rainfall")

# ================== PREDICTION ==================
if predict:

    input_df = pd.DataFrame(
        [[pressure, dewpoint, humidity, cloud,
        sunshine, winddirection, windspeed]],
        columns=feature_names
    )

    prediction = model.predict(input_df)

    st.markdown("---")

    if prediction[0] == 1:
        st.success("🌧️ Rainfall Expected")
        st.balloons()
    else:
        st.warning("☀️ No Rainfall Expected")

# ================== PROBABILITY CHART (PLOTLY) ==================
    if hasattr(model,"predict_proba"):

        probs = model.predict_proba(input_df)[0]

        fig = go.Figure(
            data=[go.Bar(
                x=["No Rain","Rain"],
                y=probs,
                marker_color=["#F00C98","#11b2cb"]
            )]
        )

        fig.update_layout(
            title="Prediction Confidence",
            yaxis_title="Probability",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

# ================== FEATURE IMPORTANCE ==================
    if hasattr(model,"feature_importances_"):

        fig2, ax2 = plt.subplots()

        importance = model.feature_importances_

        sorted_idx = importance.argsort()

        ax2.barh(
            [feature_names[i] for i in sorted_idx],
            importance[sorted_idx],
            color="#e91010"
        )

        ax2.set_title("Feature Importance")

        st.pyplot(fig2)