"""
ADX (Average Directional Index) Indicator

ADX measures the strength of a trend, whether up or down. It's derived from the
Directional Movement Indicators (+DI and -DI). Values above 25 indicate a strong
trend, while values below 20 indicate a weak trend or ranging market.
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
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift(1))
    low_close = abs(data['Low'] - data['Close'].shift(1))
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range


def calculate_adx(data, period=14):
    """
    Calculate Average Directional Index (ADX) and directional indicators.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing High, Low, and Close columns
    period : int, default 14
        Number of periods for ADX calculation
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing +DI, -DI, ADX, and trend strength
    
    Example:
    --------
    >>> adx = calculate_adx(df, period=14)
    >>> df['ADX'] = adx['ADX']
    >>> df['+DI'] = adx['+DI']
    >>> df['-DI'] = adx['-DI']
    """
    required_columns = ['High', 'Low', 'Close']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Calculate directional movements
    up_move = data['High'] - data['High'].shift(1)
    down_move = data['Low'].shift(1) - data['Low']
    
    # Calculate +DM and -DM
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate True Range
    tr = calculate_true_range(data)
    
    # Smooth the values using Wilder's smoothing (EMA with alpha = 1/period)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    # Calculate DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Calculate ADX (smoothed DX)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    # Create result DataFrame
    result = pd.DataFrame({
        '+DI': plus_di,
        '-DI': minus_di,
        'ADX': adx,
        'Trend_Strength': np.where(adx > 25, 'Strong',
                                   np.where(adx > 20, 'Moderate', 'Weak')),
        'Trend_Direction': np.where(plus_di > minus_di, 'Bullish', 'Bearish')
    })
    
    return result


def calculate_adx_with_signals(data, period=14, adx_threshold=25):
    """
    Calculate ADX with trading signals based on DI crossovers.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing High, Low, and Close columns
    period : int, default 14
        Number of periods for ADX calculation
    adx_threshold : float, default 25
        ADX threshold for confirming trend strength
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with ADX values and trading signals
    
    Example:
    --------
    >>> adx_signals = calculate_adx_with_signals(df)
    """
    result = calculate_adx(data, period)
    
    # Identify DI crossovers
    result['Bullish_Crossover'] = (result['+DI'] > result['-DI']) & \
                                  (result['+DI'].shift(1) <= result['-DI'].shift(1))
    result['Bearish_Crossover'] = (result['+DI'] < result['-DI']) & \
                                  (result['+DI'].shift(1) >= result['-DI'].shift(1))
    
    # Generate signals only when ADX confirms trend strength
    result['Signal'] = np.where(
        result['Bullish_Crossover'] & (result['ADX'] > adx_threshold), 'Strong Buy',
        np.where(result['Bearish_Crossover'] & (result['ADX'] > adx_threshold), 'Strong Sell',
                np.where(result['Bullish_Crossover'], 'Weak Buy',
                        np.where(result['Bearish_Crossover'], 'Weak Sell', 'Hold')))
    )
    
    return result
