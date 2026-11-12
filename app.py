import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Outfit Recommender", page_icon="👕")
bundle = joblib.load("model.pkl")
model, features = bundle["model"], bundle["features"]

ADVICE = {
    "Rainwear": "🌧️ Raincoat or waterproof jacket, umbrella, water-resistant shoes.",
    "Light Clothing": "☀️ Breathable cotton/linen, shorts or light tops, sunglasses.",
    "Layered Clothing": "🧥 Jacket or sweater over a base layer; easy to add or remove.",
    "Regular Casual Wear": "👕 Everyday jeans/t-shirt or comfortable casual clothes.",
}

st.title("👕 Weather-Based Outfit Recommendation")
st.write("Enter the weather and your occasion to get an outfit suggestion.")

c1, c2 = st.columns(2)
temp = c1.slider("Temperature (°C)", 0.0, 50.0, 25.0, 0.1)
rain = c2.slider("Rainfall (mm)", 0.0, 60.0, 0.0, 0.1)
hum = c1.slider("Humidity (%)", 0.0, 100.0, 70.0, 0.1)
wind = c2.slider("Wind speed (km/h)", 0.0, 60.0, 10.0, 0.1)
occasion = st.selectbox("Occasion", ["Casual", "College", "Office", "Outdoor"])

if st.button("Recommend outfit", type="primary"):
    row = {"temperature_c": temp, "rainfall_mm": rain,
           "humidity_pct": hum, "wind_speed_kmh": wind}
    for o in ["Casual", "College", "Office", "Outdoor"]:
        row[f"occasion_{o}"] = occasion == o
    X = pd.DataFrame([row])[features]

    pred = model.predict(X)[0]
    st.success(f"Recommended: **{pred}**")
    st.info(ADVICE[pred])

    st.subheader("Prediction probabilities")
    probs = pd.Series(model.predict_proba(X)[0], index=model.classes_)
    st.bar_chart(probs)

with st.expander("Model evaluation"):
    st.metric("Test accuracy", f"{bundle['accuracy']:.1%}")
    rep = pd.DataFrame(bundle["report"]).T.round(3)
    st.dataframe(rep)
    for img in ["confusion_matrix", "feature_importance", "decision_tree"]:
        st.image(f"plots/{img}.png")
