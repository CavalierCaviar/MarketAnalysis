"""
Relative Strength Index (RSI) Indicator

RSI is a momentum oscillator that measures the speed and magnitude of price changes.
It ranges from 0 to 100, with readings above 70 indicating overbought conditions
and readings below 30 indicating oversold conditions.
"""

import pandas as pd
import numpy as np


def calculate_rsi(data, column='Close', period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    column : str, default 'Close'
        Column name to use for RSI calculation
    period : int, default 14
        Number of periods to use for RSI calculation
    
    Returns:
    --------
    pd.Series
        Series containing RSI values
    
    Example:
    --------
    >>> df['RSI'] = calculate_rsi(df, column='Close', period=14)
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    # Calculate price changes
    delta = data[column].diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=period, min_periods=period).mean()
    avg_losses = losses.rolling(window=period, min_periods=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_rsi_with_levels(data, column='Close', period=14, overbought=70, oversold=30):
    """
    Calculate RSI with overbought and oversold signals.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    column : str, default 'Close'
        Column name to use for RSI calculation
    period : int, default 14
        Number of periods to use for RSI calculation
    overbought : int, default 70
        RSI level indicating overbought condition
    oversold : int, default 30
        RSI level indicating oversold condition
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with RSI, overbought signal, and oversold signal
    
    Example:
    --------
    >>> rsi_df = calculate_rsi_with_levels(df)
    """
    rsi = calculate_rsi(data, column, period)
    
    result = pd.DataFrame({
        'RSI': rsi,
        'Overbought': rsi >= overbought,
        'Oversold': rsi <= oversold,
        'Signal': np.where(rsi >= overbought, 'Overbought', 
                          np.where(rsi <= oversold, 'Oversold', 'Neutral'))
    })
    
    return result
