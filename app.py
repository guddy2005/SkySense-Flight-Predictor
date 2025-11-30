import streamlit as st
import pickle
import numpy as np
import pandas as pd # Pandas import zaroori hai fix ke liye

# --- 1. Page Config & Setup ---
st.set_page_config(
    page_title="SkySense AI | Flight Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS ---
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }
        .main { background-color: #0E1117; }
        h1, h2, h3 { color: #FFFFFF !important; font-weight: 700 !important; }
        div.stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #FF4B2B 0%, #FF416C 100%);
            color: white; border: none; padding: 12px 24px;
            font-weight: 600; border-radius: 8px; transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            transform: scale(1.02); box-shadow: 0 4px 15px rgba(255, 75, 43, 0.4);
        }
        [data-testid="stMetricValue"] { font-size: 2rem !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Load Model ---
try:
    model = pickle.load(open('xgbmodel.pkl', 'rb'))
except FileNotFoundError:
    st.error("⚠️ Model not found! Please run 'python model_build.py' first.")
    st.stop()

# --- 3. Main Layout ---

# >>> HEADER IMAGE (FIXED WARNING) <<<
st.image("https://images.unsplash.com/photo-1436491865332-7a61a109cc05?q=80&w=1200&h=400&auto=format&fit=crop", 
         use_container_width=True) # New streamlit version supports this, agar warning aaye toh ignore karein

st.title("✈️ SkySense: AI-Powered Flight Delay Predictor")
st.markdown("Plan smarter travel. Use machine learning to estimate the probability of flight delays based on historical patterns.")
st.divider()

col_form, col_info = st.columns([2, 1], gap="large")

with col_form:
    st.subheader("🛠️ Enter Flight Details")
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            airline = st.selectbox("Choose Airline", ["Delta", "American", "United", "Southwest", "JetBlue"])
            airline_map = {"Delta": 0, "American": 1, "United": 2, "Southwest": 3, "JetBlue": 4}
            airline_val = airline_map[airline]
            month = st.slider("Month of Travel", 1, 12, 6)
        with c2:
            distance = st.number_input("Estimated Distance (miles)", min_value=50, max_value=5000, value=800, step=50)
            dep_time = st.slider("Departure Time (24h Format)", 0, 2359, 1430, step=30)
            st.caption("*Example: 1430 means 2:30 PM*")

        submit = st.form_submit_button("Analyze Risk 🚀")

    if submit:
        # >>> FIX: Using DataFrame to silence Scikit-Learn Warnings <<<
        features = pd.DataFrame([[airline_val, month, distance, dep_time]], 
                                columns=['Airline', 'Month', 'Distance', 'DepTime'])
        
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]

        st.write("")
        if prediction[0] == 1:
            st.error("⚠️ High Risk of Delay Detected")
            st.metric(label="Delay Probability", value=f"{probability*100:.1f}%", delta="- High Risk", delta_color="inverse")
        else:
            st.success("✅ Flight Likely On-Time")
            safe_prob = (1 - probability) * 100
            st.metric(label="On-Time Probability", value=f"{safe_prob:.1f}%", delta="+ Good to go")

with col_info:
    st.markdown("### ℹ️ About SkySense")
    st.info("SkySense uses a **Random Forest** model trained on historical aviation data.")
    st.markdown("### 🤔 Why these inputs?")
    with st.expander("See Explanation"):
        st.markdown("""
        **1. Airline:** Operational efficiency varies by carrier.
        **2. Departure Time:** Knock-on delays increase later in the day.
        **3. Month:** Accounts for seasonal weather.
        **4. Distance:** Longer routes have higher risk.
        """)
    st.divider()
    st.markdown("<center>SkySense AI Project © 2025</center>", unsafe_allow_html=True)