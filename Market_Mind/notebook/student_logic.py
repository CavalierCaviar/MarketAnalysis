import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def run_full_model():
    outputs = {}

    # ====== DATA DOWNLOAD (from your notebook) ======
    df = yf.download(
        "AAPL",
        period="1y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    df.index = pd.to_datetime(df.index)

    # 🔴 THIS LINE IS THE REAL FIX
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)


    outputs["raw_data"] = df

    # ====== FEATURE ENGINEERING ======
    df["returns"] = df["Close"].pct_change()
    df["rolling_mean"] = df["Close"].rolling(20).mean()

    outputs["processed_data"] = df

    # ====== SUMMARY / METRICS ======
    outputs["mean_return"] = df["returns"].mean()
    outputs["volatility"] = df["returns"].std()

    # ====== PLOT 1: PRICE ======
    fig_price, ax_price = plt.subplots(figsize=(10, 5))
    ax_price.plot(df.index, df["Close"], label="Close")
    ax_price.plot(df.index, df["rolling_mean"], label="MA(20)")
    ax_price.set_title("Price & Moving Average")
    ax_price.legend()
    outputs["price_plot"] = fig_price

    # ====== PLOT 2: RETURNS ======
    fig_ret, ax_ret = plt.subplots(figsize=(10, 4))
    ax_ret.plot(df.index, df["returns"])
    ax_ret.set_title("Daily Returns")
    outputs["returns_plot"] = fig_ret

    return outputs

import plotly.graph_objects as go

def plot_candlestick(df):
    df = df.reset_index()

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["Date"],
                open=df["Open"].astype(float),
                high=df["High"].astype(float),
                low=df["Low"].astype(float),
                close=df["Close"].astype(float),
            )
        ]
    )

    fig.update_layout(
        title="Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
    )

    return fig



