"""
Stock Data Retrieval and Candlestick Pattern Visualization

This script retrieves stock data from Yahoo Finance using yfinance
and creates interactive candlestick charts using plotly.
"""

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta


def get_stock_data(ticker: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    """
    Retrieve stock data from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        period (str): Time period to fetch data for
                     Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval (str): Data interval
                       Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    
    Returns:
        pd.DataFrame: Stock data with OHLCV columns
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        
        return data
    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")
        return pd.DataFrame()


def create_candlestick_chart(data: pd.DataFrame, ticker: str, show_volume: bool = True) -> go.Figure:
    """
    Create an interactive candlestick chart using plotly.
    
    Args:
        data (pd.DataFrame): Stock data with OHLCV columns
        ticker (str): Stock ticker symbol for the chart title
        show_volume (bool): Whether to display volume subplot
    
    Returns:
        go.Figure: Plotly figure object
    """
    if data.empty:
        print("No data to plot")
        return None
    
    # Create subplots if volume is to be shown
    if show_volume:
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{ticker} Stock Price', 'Volume')
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Add volume bar chart
        colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] 
                  else 'green' for i in range(len(data))]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Stock Candlestick Chart',
            yaxis_title='Price (USD)',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=800,
            hovermode='x unified'
        )
        
    else:
        # Create simple candlestick chart without volume
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='OHLC',
                    increasing_line_color='green',
                    decreasing_line_color='red'
                )
            ]
        )
        
        fig.update_layout(
            title=f'{ticker} Stock Candlestick Chart',
            yaxis_title='Price (USD)',
            xaxis_title='Date',
            xaxis_rangeslider_visible=True,
            template='plotly_dark',
            height=600,
            hovermode='x unified'
        )
    
    return fig


def add_moving_averages(fig: go.Figure, data: pd.DataFrame, periods: list = [20, 50]) -> go.Figure:
    """
    Add moving averages to the candlestick chart.
    
    Args:
        fig (go.Figure): Existing plotly figure
        data (pd.DataFrame): Stock data
        periods (list): List of periods for moving averages
    
    Returns:
        go.Figure: Updated plotly figure
    """
    colors = ['orange', 'blue', 'purple', 'yellow']
    
    for i, period in enumerate(periods):
        ma = data['Close'].rolling(window=period).mean()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=ma,
                mode='lines',
                name=f'MA{period}',
                line=dict(color=colors[i % len(colors)], width=1.5)
            ),
            row=1, col=1
        )
    
    return fig


def print_stock_summary(data: pd.DataFrame, ticker: str):
    """
    Print a summary of the stock data.
    
    Args:
        data (pd.DataFrame): Stock data
        ticker (str): Stock ticker symbol
    """
    if data.empty:
        print("No data to summarize")
        return
    
    latest = data.iloc[-1]
    previous = data.iloc[-2] if len(data) > 1 else latest
    
    change = latest['Close'] - previous['Close']
    change_pct = (change / previous['Close']) * 100
    
    print(f"\n{'='*50}")
    print(f"Stock Summary for {ticker}")
    print(f"{'='*50}")
    print(f"Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"\nLatest Price: ${latest['Close']:.2f}")
    print(f"Change: ${change:.2f} ({change_pct:+.2f}%)")
    print(f"\nOpen: ${latest['Open']:.2f}")
    print(f"High: ${latest['High']:.2f}")
    print(f"Low: ${latest['Low']:.2f}")
    print(f"Close: ${latest['Close']:.2f}")
    print(f"Volume: {latest['Volume']:,.0f}")
    print(f"\nPeriod High: ${data['High'].max():.2f}")
    print(f"Period Low: ${data['Low'].min():.2f}")
    print(f"Average Volume: {data['Volume'].mean():,.0f}")
    print(f"{'='*50}\n")


def main():
    """
    Main function to demonstrate stock data retrieval and visualization.
    """
    # Example usage
    ticker = "AAPL"  # Apple Inc.
    period = "3mo"   # Last 3 months
    interval = "1d"  # Daily data
    
    print(f"Fetching stock data for {ticker}...")
    
    # Get stock data
    stock_data = get_stock_data(ticker, period=period, interval=interval)
    
    if not stock_data.empty:
        # Print summary
        print_stock_summary(stock_data, ticker)
        
        # Create candlestick chart with volume
        fig = create_candlestick_chart(stock_data, ticker, show_volume=True)
        
        # Add moving averages
        fig = add_moving_averages(fig, stock_data, periods=[20, 50])
        
        # Show the chart
        print("Displaying candlestick chart...")
        fig.show()
        
        # Optionally save to HTML
        output_file = f"{ticker}_candlestick_chart.html"
        fig.write_html(output_file)
        print(f"Chart saved to {output_file}")
        
    else:
        print("Failed to retrieve stock data. Please check the ticker symbol and try again.")


if __name__ == "__main__":
    main()
