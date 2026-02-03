"""
MACD (Moving Average Convergence Divergence) Indicator

MACD is a trend-following momentum indicator that shows the relationship
between two moving averages of a security's price. It consists of the MACD line,
signal line, and histogram.
"""

import pandas as pd
import numpy as np


def calculate_ema(data, period):
    """
    Calculate Exponential Moving Average.
    
    Parameters:
    -----------
    data : pd.Series
        Price data series
    period : int
        Number of periods for EMA
    
    Returns:
    --------
    pd.Series
        EMA values
    """
    return data.ewm(span=period, adjust=False).mean()


def calculate_macd(data, column='Close', fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD (Moving Average Convergence Divergence) indicator.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    column : str, default 'Close'
        Column name to use for MACD calculation
    fast_period : int, default 12
        Period for fast EMA
    slow_period : int, default 26
        Period for slow EMA
    signal_period : int, default 9
        Period for signal line EMA
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing MACD line, Signal line, and Histogram
    
    Example:
    --------
    >>> macd = calculate_macd(df, column='Close')
    >>> df['MACD'] = macd['MACD']
    >>> df['MACD_Signal'] = macd['Signal']
    >>> df['MACD_Histogram'] = macd['Histogram']
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    # Calculate fast and slow EMAs
    fast_ema = calculate_ema(data[column], fast_period)
    slow_ema = calculate_ema(data[column], slow_period)
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = calculate_ema(macd_line, signal_period)
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    # Create result DataFrame
    result = pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': histogram
    })
    
    return result


def calculate_macd_with_signals(data, column='Close', fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD with buy/sell signals.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    column : str, default 'Close'
        Column name to use for MACD calculation
    fast_period : int, default 12
        Period for fast EMA
    slow_period : int, default 26
        Period for slow EMA
    signal_period : int, default 9
        Period for signal line EMA
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with MACD values and crossover signals
    
    Example:
    --------
    >>> macd_signals = calculate_macd_with_signals(df)
    """
    macd = calculate_macd(data, column, fast_period, slow_period, signal_period)
    
    # Identify crossovers
    macd['Bullish_Crossover'] = (macd['MACD'] > macd['Signal']) & (macd['MACD'].shift(1) <= macd['Signal'].shift(1))
    macd['Bearish_Crossover'] = (macd['MACD'] < macd['Signal']) & (macd['MACD'].shift(1) >= macd['Signal'].shift(1))
    
    # Create signal column
    macd['Trading_Signal'] = np.where(macd['Bullish_Crossover'], 'Buy',
                                      np.where(macd['Bearish_Crossover'], 'Sell', 'Hold'))
    
    # Identify histogram trend
    macd['Histogram_Trend'] = np.where(macd['Histogram'] > macd['Histogram'].shift(1), 'Increasing',
                                       np.where(macd['Histogram'] < macd['Histogram'].shift(1), 'Decreasing', 'Flat'))
    
    return macd
