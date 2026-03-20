import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests
from datetime import datetime
import os


st.set_page_config(page_title="Rainfall Dashboard", page_icon="🌧️", layout="wide")

st.write("🔥 NEW VERSION RUNNING")

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

# ================== API KEY ==================
api_key = os.getenv("API_KEY")

# ================== TITLE ==================
st.title("🌧️ Rainfall Prediction Dashboard")
st.divider()

# ================== CITY INPUT ==================
st.subheader("🌍 Live Weather Prediction")
city = st.text_input("Enter City Name", placeholder="e.g. Kolkata")
st.divider()

# ================== INPUT ==================
col1, col2 = st.columns(2)

with col1:
    pressure = st.number_input("Pressure", value=1015.9)
    dewpoint = st.number_input("Dew Point", value=19.9)
    humidity = st.slider("Humidity (%)", 0, 100, 95)
    cloud = st.slider("Cloud (%)", 0, 100, 81)

with col2:
    sunshine = st.number_input("Sunshine", value=0.0)
    winddirection = st.number_input("Wind Direction", value=40)
    windspeed = st.number_input("Wind Speed", value=13.7)

# ================== FETCH LIVE DATA ==================
if city:
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)

        # ✅ STATUS CHECK
        if response.status_code != 200:
            st.error(f"❌ API failed: {response.status_code}")
            st.write(response.text)
            st.stop()

        data = response.json()

        # ✅ ERROR HANDLING
        if str(data.get("cod")) != "200":
            st.error(f"❌ {data.get('message')}")
            st.stop()

        # ✅ SAFE EXTRACTION
        main = data.get("main", {})
        wind = data.get("wind", {})
        clouds_data = data.get("clouds", {})
        sys = data.get("sys", {})
        weather = data.get("weather", [{}])[0]

        # Update inputs
        pressure = main.get("pressure", pressure)
        humidity = main.get("humidity", humidity)
        windspeed = wind.get("speed", windspeed)
        cloud = clouds_data.get("all", cloud)
        winddirection = wind.get("deg", winddirection)

        dewpoint = 0
        sunshine = 0

        st.success(f"📍 Live data fetched for {city}")

        # ================== WEATHER UI ==================
        st.subheader(f"📍 Weather in {data.get('name')}, {sys.get('country')}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌡️ Temp (°C)", round(main.get("temp", 0),1))
        with col2:
            st.metric("🤒 Feels Like", round(main.get("feels_like", 0),1))
        with col3:
            st.metric("💧 Humidity", main.get("humidity", 0))

        st.markdown("---")

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("🌬️ Wind Speed", wind.get("speed", 0))
        with col5:
            st.metric("☁️ Clouds", clouds_data.get("all", 0))
        with col6:
            st.metric("🌡️ Pressure", main.get("pressure", 0))

        st.markdown("---")

        col7, col8 = st.columns(2)
        with col7:
            st.metric("🧭 Wind Dir", wind.get("deg", 0))
        with col8:
            st.metric("👁️ Visibility", data.get("visibility", 0))

        st.success(f"🌤️ {weather.get('description','').title()}")

        icon = weather.get("icon", "01d")
        icon_url = f"http://openweathermap.org/img/wn/{icon}@2x.png"
        st.image(icon_url, width=100)

        sunrise = datetime.fromtimestamp(sys.get("sunrise", 0))
        sunset = datetime.fromtimestamp(sys.get("sunset", 0))
        st.info(f"🌅 Sunrise: {sunrise.strftime('%H:%M')} | 🌇 Sunset: {sunset.strftime('%H:%M')}")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

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

    if hasattr(model,"predict_proba"):
        probs = model.predict_proba(input_df)[0]

        fig = go.Figure(
            data=[go.Bar(
                x=["No Rain","Rain"],
                y=probs
            )]
        )

        fig.update_layout(title="Prediction Confidence", yaxis_title="Probability")

        st.plotly_chart(fig, use_container_width=True)

        if probs[1] > 0.7:
            st.error("⚠️ Heavy Rain Alert!")

    if hasattr(model,"feature_importances_"):
        fig2, ax2 = plt.subplots()

        importance = model.feature_importances_
        sorted_idx = importance.argsort()

        ax2.barh([feature_names[i] for i in sorted_idx], importance[sorted_idx])
        ax2.set_title("Feature Importance")

        st.pyplot(fig2)