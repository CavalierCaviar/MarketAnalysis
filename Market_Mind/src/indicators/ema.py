"""
EMA (Exponential Moving Average) Indicator

The EMA is a type of moving average that places greater weight on recent data points,
making it more responsive to recent price changes compared to a simple moving average.
"""

import pandas as pd
import numpy as np


def calculate_ema(data, column='Close', period=20):
    """
    Calculate Exponential Moving Average (EMA).
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    column : str, default 'Close'
        Column name to use for EMA calculation
    period : int, default 20
        Number of periods for EMA
    
    Returns:
    --------
    pd.Series
        Series containing EMA values
    
    Example:
    --------
    >>> df['EMA_20'] = calculate_ema(df, column='Close', period=20)
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    return data[column].ewm(span=period, adjust=False).mean()


def calculate_multiple_emas(data, column='Close', periods=[9, 12, 20, 26, 50, 100, 200]):
    """
    Calculate multiple EMAs at once.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    column : str, default 'Close'
        Column name to use for EMA calculation
    periods : list, default [9, 12, 20, 26, 50, 100, 200]
        List of periods for which to calculate EMAs
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing all EMAs
    
    Example:
    --------
    >>> emas = calculate_multiple_emas(df, periods=[12, 26, 50])
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    result = pd.DataFrame()
    
    for period in periods:
        result[f'EMA_{period}'] = calculate_ema(data, column, period)
    
    return result


def calculate_ema_crossover(data, column='Close', fast_period=12, slow_period=26):
    """
    Calculate EMA crossover signals.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    column : str, default 'Close'
        Column name to use for calculation
    fast_period : int, default 12
        Period for fast EMA
    slow_period : int, default 26
        Period for slow EMA
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with fast EMA, slow EMA, and crossover signals
    
    Example:
    --------
    >>> ema_cross = calculate_ema_crossover(df, fast_period=12, slow_period=26)
    """
    fast_ema = calculate_ema(data, column, fast_period)
    slow_ema = calculate_ema(data, column, slow_period)
    
    # Identify crossovers
    bullish_crossover = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
    bearish_crossover = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
    
    result = pd.DataFrame({
        f'EMA_{fast_period}': fast_ema,
        f'EMA_{slow_period}': slow_ema,
        'Bullish_Crossover': bullish_crossover,
        'Bearish_Crossover': bearish_crossover,
        'Signal': np.where(bullish_crossover, 'Buy',
                          np.where(bearish_crossover, 'Sell', 'Hold')),
        'Trend': np.where(fast_ema > slow_ema, 'Bullish', 'Bearish')
    })
    
    return result


def calculate_ema_ribbon(data, column='Close', periods=[8, 13, 21, 34, 55]):
    """
    Calculate EMA ribbon (multiple EMAs for trend analysis).
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    column : str, default 'Close'
        Column name to use for calculation
    periods : list, default [8, 13, 21, 34, 55]
        List of Fibonacci-based periods for EMAs
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with multiple EMAs and trend strength
    
    Example:
    --------
    >>> ema_ribbon = calculate_ema_ribbon(df)
    """
    result = calculate_multiple_emas(data, column, periods)
    
    # Calculate ribbon alignment (all EMAs in order indicates strong trend)
    ema_cols = [f'EMA_{p}' for p in sorted(periods)]
    
    # Check if EMAs are in ascending order (bullish) or descending order (bearish)
    ascending = all(result[ema_cols[i]].iloc[-1] < result[ema_cols[i+1]].iloc[-1] 
                   for i in range(len(ema_cols)-1))
    descending = all(result[ema_cols[i]].iloc[-1] > result[ema_cols[i+1]].iloc[-1] 
                    for i in range(len(ema_cols)-1))
    
    result['Ribbon_Aligned'] = ascending or descending
    result['Ribbon_Trend'] = 'Bullish' if ascending else ('Bearish' if descending else 'Mixed')
    
    return result
