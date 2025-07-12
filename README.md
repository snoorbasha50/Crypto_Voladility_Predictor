# 🔮 Crypto Volatility Predictor

A full-stack machine learning project that predicts **next-day volatility** for top cryptocurrencies using historical price data and news sentiment.

### 🚀 Live Demo

**▶ Try it live:**  
🌐 [https://snoorbasha50-crypto-voladility-predictor-frontend-kr7aeu.streamlit.app/](https://snoorbasha50-crypto-voladility-predictor-frontend-kr7aeu.streamlit.app/)


---

## 📌 Key Features

✅ Predicts **next-day high or low volatility**  
✅ Combines **technical indicators + news sentiment**  
✅ Supports top 5 coins: Bitcoin, Ethereum, Solana, Ripple, Binance Coin  
✅ Interactive UI with **charts and model predictions**  
✅ Clean modular code and reusable pipeline

---

## 🧠 How It Works

### 1. 📈 Data Collection
- Historical OHLCV data from [CoinGecko API](https://www.coingecko.com/en/api)
- News headlines from [NewsAPI.org](https://newsapi.org)
- Top 5 crypto coins

### 2. 🏗 Feature Engineering
- Price-based: `returns`, `ma7`, `ma14`, `volatility`, `RSI`
- Sentiment-based: daily average VADER compound scores

### 3. 🧪 Modeling
- Binary classification (high/low volatility)
- Model: `RandomForestClassifier`
- Hyperparameter tuning with `GridSearchCV`
- Trained per-coin and saved using `joblib`

### 4. 💡 Frontend (Streamlit)
- Choose coin and date to get prediction
- Visualize price, moving averages, and sentiment
- See model confidence and input features

---

## 📂 Folder Structure
```
crypto-volatility-predictor/
│
├── data/ # Final CSVs for each coin (used by UI)
├── models/ # Saved model per coin (e.g. btc_model.pkl)
├── backend/ # Jupyter notebooks / pipeline code
├── frontend.py # Streamlit frontend
├── requirements.txt # Python dependencies
└── README.md # Project overview
```


## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/snoorbasha50/crypto-volatility-predictor.git
cd crypto-volatility-predictor

### 2.Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

###3. Install dependencies
pip install -r requirements.txt

4.Set up .env for NewsAPI(Optional)
   NEWSAPI_KEY=your_key_here

5. Run the app
streamlit run frontend.py




📌 Tech Stack

Python, Pandas, Scikit-learn
VADER for sentiment analysis
Streamlit for frontend
Plotly for charts
CoinGecko / NewsAPI for data


