"""
ATR (Average True Range) Indicator

ATR measures market volatility by decomposing the entire range of an asset price
for that period. Higher ATR values indicate higher volatility, while lower values
indicate lower volatility.
"""

import pandas as pd
import numpy as np


def calculate_true_range(data):
    """
    Calculate True Range.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing High, Low, and Close columns
    
    Returns:
    --------
    pd.Series
        True Range values
    """
    required_columns = ['High', 'Low', 'Close']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift(1))
    low_close = abs(data['Low'] - data['Close'].shift(1))
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range


def calculate_atr(data, period=14):
    """
    Calculate Average True Range (ATR).
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing High, Low, and Close columns
    period : int, default 14
        Number of periods for ATR calculation
    
    Returns:
    --------
    pd.Series
        Series containing ATR values
    
    Example:
    --------
    >>> df['ATR'] = calculate_atr(df, period=14)
    """
    tr = calculate_true_range(data)
    
    # Use Wilder's smoothing (EMA with alpha = 1/period)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    return atr


def calculate_atr_bands(data, period=14, multiplier=2):
    """
    Calculate ATR-based volatility bands around price.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing High, Low, and Close columns
    period : int, default 14
        Number of periods for ATR calculation
    multiplier : float, default 2
        Multiplier for ATR bands
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with ATR, upper band, and lower band
    
    Example:
    --------
    >>> atr_bands = calculate_atr_bands(df, period=14, multiplier=2)
    """
    atr = calculate_atr(data, period)
    
    # Calculate bands around closing price
    result = pd.DataFrame({
        'ATR': atr,
        'Upper_Band': data['Close'] + (atr * multiplier),
        'Lower_Band': data['Close'] - (atr * multiplier),
        'Band_Width': atr * multiplier * 2
    })
    
    return result


def calculate_atr_percentage(data, period=14):
    """
    Calculate ATR as a percentage of the closing price.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing High, Low, and Close columns
    period : int, default 14
        Number of periods for ATR calculation
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with ATR and ATR percentage
    
    Example:
    --------
    >>> atr_pct = calculate_atr_percentage(df)
    """
    atr = calculate_atr(data, period)
    
    result = pd.DataFrame({
        'ATR': atr,
        'ATR_Percentage': (atr / data['Close']) * 100,
        'Volatility_Level': np.where((atr / data['Close']) * 100 > 3, 'High',
                                     np.where((atr / data['Close']) * 100 > 1.5, 'Medium', 'Low'))
    })
    
    return result


def calculate_atr_stop_loss(data, period=14, multiplier=2, position_type='long'):
    """
    Calculate stop-loss levels based on ATR.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing High, Low, and Close columns
    period : int, default 14
        Number of periods for ATR calculation
    multiplier : float, default 2
        ATR multiplier for stop-loss distance
    position_type : str, default 'long'
        'long' or 'short' position
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with ATR and stop-loss levels
    
    Example:
    --------
    >>> stop_loss = calculate_atr_stop_loss(df, multiplier=2, position_type='long')
    """
    atr = calculate_atr(data, period)
    
    if position_type.lower() == 'long':
        stop_loss = data['Close'] - (atr * multiplier)
        take_profit = data['Close'] + (atr * multiplier * 1.5)
    else:  # short position
        stop_loss = data['Close'] + (atr * multiplier)
        take_profit = data['Close'] - (atr * multiplier * 1.5)
    
    result = pd.DataFrame({
        'ATR': atr,
        'Entry_Price': data['Close'],
        'Stop_Loss': stop_loss,
        'Take_Profit': take_profit,
        'Risk_Amount': abs(data['Close'] - stop_loss),
        'Reward_Amount': abs(take_profit - data['Close']),
        'Risk_Reward_Ratio': abs(take_profit - data['Close']) / abs(data['Close'] - stop_loss)
    })
    
    return result
