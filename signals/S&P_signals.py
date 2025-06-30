import yfinance as yf
import json
import os
from datetime import datetime
from fredapi import Fred

# Load your FRED API key from environment variables (GitHub Actions secrets)
FRED_API_KEY = os.environ.get("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)

def fetch_fed_rate():
    """Fetch the latest Fed Funds Effective Rate (FEDFUNDS) from FRED."""
    fed_data = fred.get_series_latest_release('FEDFUNDS')
    latest_value = fed_data[-1]
    return latest_value

def fetch_jobs_data():
    """Fetch the latest unemployment rate (UNRATE) from FRED."""
    unrate_data = fred.get_series_latest_release('UNRATE')
    latest_value = unrate_data[-1]
    return latest_value

def analyze_technicals():
    """Download S&P500 data & calculate 50-day SMA and RSI."""
    spx = yf.Ticker("^GSPC").history(period="6mo")
    ma50 = spx['Close'].rolling(50).mean()
    latest_close = spx['Close'][-1]
    latest_ma50 = ma50[-1]

    # Simple RSI calculation (14-period)
    delta = spx['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    latest_rsi = rsi[-1]

    return latest_close, latest_ma50, latest_rsi

def explain_signal(signal, explanation):
    """Prints an explanation for a given signal."""
    print(f"🔹 {signal}: {explanation}")

def make_recommendation(fed_rate, unemployment, close, ma50, rsi):
    """Determine action recommendation based on if/else logic, with explanations. Also save to JSON."""
    timestamp = datetime.now().isoformat()
    print(f"\n📅 Date: {timestamp}")
    print(f"🏦 Fed Funds Rate: {fed_rate:.2f}%, 📈 Unemployment: {unemployment:.2f}%")
    print(f"📊 S&P500 Latest Close: {close:.2f}, 50-day SMA: {ma50:.2f}, RSI: {rsi:.1f}")

    # Macro signals
    easing_likely = unemployment > 4.5 or fed_rate > 5.5
    if easing_likely:
        macro_explanation = (
            f"Unemployment ({unemployment:.2f}%) > 4.5% or Fed rate ({fed_rate:.2f}%) > 5.5% → Fed may pivot to easing."
        )
        explain_signal("Macro", macro_explanation)
    else:
        macro_explanation = (
            f"Unemployment ({unemployment:.2f}%) ≤ 4.5% and Fed rate ({fed_rate:.2f}%) ≤ 5.5% → Fed likely to stay hawkish."
        )
        explain_signal("Macro", macro_explanation)

    # Technical signals
    bullish_technicals = close > ma50 and rsi < 70
    bearish_technicals = close < ma50 or rsi > 70

    if bullish_technicals:
        technical_explanation = (
            f"S&P500 price ({close:.2f}) is above 50-day SMA ({ma50:.2f}) and RSI ({rsi:.1f}) < 70 → bullish momentum."
        )
        explain_signal("Technicals", technical_explanation)
    elif bearish_technicals:
        technical_explanation = (
            f"S&P500 price ({close:.2f}) is below 50-day SMA ({ma50:.2f}) or RSI ({rsi:.1f}) > 70 → bearish or overbought."
        )
        explain_signal("Technicals", technical_explanation)
    else:
        technical_explanation = (
            f"S&P500 price and RSI provide mixed signals → market uncertain."
        )
        explain_signal("Technicals", technical_explanation)

    # Decision tree
    if easing_likely and bullish_technicals:
        action = (
            "🔵 **Increase Exposure**\n"
            "Fed is likely to ease (stimulus tailwind) *and* technicals confirm bullish trend.\n"
            "Consider allocating more to S&P500 or adding positions."
        )
    elif easing_likely and bearish_technicals:
        action = (
            "🟡 **Wait**\n"
            "Fed may ease, but technicals show weakness → better to wait for trend confirmation."
        )
    elif not easing_likely and bearish_technicals:
        action = (
            "🔴 **Defensive**\n"
            "Fed unlikely to ease (risk of continued high rates) *and* technicals are bearish.\n"
            "Consider reducing exposure or rotating to defensive assets."
        )
    else:
        action = (
            "🟢 **Maintain Core Exposure**\n"
            "No strong macro or technical signals → stay invested but avoid aggressive new positions."
        )

    print("\n🎯 Action Recommendation:")
    print(action)

    # Bundle everything into a dict
    result = {
        "timestamp": timestamp,
        "fed_rate": fed_rate,
        "unemployment": unemployment,
        "sp500_close": close,
        "sma_50": ma50,
        "rsi": rsi,
        "macro_explanation": macro_explanation,
        "technical_explanation": technical_explanation,
        "recommendation": action
    }

    # Save to JSON file
    with open("sp500_signal.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n✅ Results saved to sp500_signal.json")

if __name__ == "__main__":
    fed_rate = fetch_fed_rate()
    unemployment = fetch_jobs_data()
    close, ma50, rsi = analyze_technicals()
    make_recommendation(fed_rate, unemployment, close, ma50, rsi)
