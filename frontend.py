import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# OpenAI.api_key = os.getenv("SECRET_KEY")






# Constants
MAX_CALLS = 10
if "call_count" not in st.session_state:
    st.session_state.call_count = 0

# --- Cached Loaders ---
@st.cache_data
def load_data(coin_id):
    return pd.read_csv(f"data/{coin_id}_features.csv", parse_dates=['timestamp'], index_col='timestamp')

@st.cache_resource
def load_model(coin_id):
    return joblib.load(f"models/{coin_id}_model.pkl")

# --- LLM Safe Call ---
def safe_llm_call(prompt):
    if st.session_state.get("call_count", 0) >= MAX_CALLS:
        return "⚠️ Free GPT usage limit reached (demo cap)."

    try:
        response = client.chat.completions.create(
             model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        st.session_state.call_count = st.session_state.get("call_count", 0) + 1
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error from LLM: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="Crypto Volatility Predictor", layout="centered")
st.title("🔮 Crypto Volatility Predictor")

# --- Coin Selection ---
coins = ["bitcoin", "ethereum", "solana", "binancecoin", "ripple"]
selected_coin = st.selectbox("Select Coin", coins)

# --- Load Data + Model ---
try:
    data = load_data(selected_coin)
    model = load_model(selected_coin)
except FileNotFoundError:
    st.error("❌ Data or model file missing. Please check your folders.")
    st.stop()

# --- Select Date ---
date_range = pd.to_datetime(data.index).date
default_date = date_range[-2] if len(date_range) >= 2 else date_range[-1]

selected_date = st.date_input(
    "Select Date for Prediction",
    default_date,
    min_value=date_range[0],
    max_value=default_date
)

# --- Predict ---
df_today = data[data.index.date == selected_date]

if not df_today.empty:
    today_features = df_today.iloc[0][['returns', 'ma7', 'ma14', 'volatility', 'rsi', 'sentiment']]
    prediction = model.predict([today_features])[0]
    confidence = model.predict_proba([today_features])[0][prediction]

    # --- Price + MA Chart ---
    st.subheader("📊 Price & Moving Averages")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['price'], name="Price", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data['ma7'], name="MA7", line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=data.index, y=data['ma14'], name="MA14", line=dict(color='green')))
    st.plotly_chart(fig, use_container_width=True)

    # --- Sentiment Chart ---
    st.subheader("📰 Sentiment Trend")
    st.line_chart(data['sentiment'])

    # --- Prediction Result ---
    st.subheader("📌 Prediction Result")
    if prediction == 1:
        st.success("✅ High Volatility Predicted Tomorrow")
    else:
        st.warning("⚪ Low/Normal Volatility Expected Tomorrow")

    st.write(f"🔒 Model Confidence: **{confidence:.2%}**")

    # --- Show Features (Expandable) ---
    with st.expander("📁 Show input features"):
        st.write(today_features.to_frame().T)

    # --- Explain with LLM ---
    # if st.button("🧠 Explain with GPT-4"):
    #     prompt = f"""
    #     I'm analyzing {selected_coin}'s volatility. Based on these features on {selected_date}:
    #     - Returns: {today_features['returns']:.4f}
    #     - RSI: {today_features['rsi']:.2f}
    #     - Volatility: {today_features['volatility']:.4f}
    #     - Sentiment: {today_features['sentiment']:.3f}

    #     Why would a model predict {'high' if prediction == 1 else 'low'} volatility for the next day?
    #     Explain this like a financial analyst in 2–3 sentences.
    #     """
    #     explanation = safe_llm_call(prompt)
    #     st.info(explanation)
else:
    st.warning("⚠️ No data available for the selected date. Try another day from the last few.")

