"""
Volume Analysis Indicators

Volume analysis helps identify the strength behind price movements. High volume
during price increases suggests strong buying pressure, while high volume during
price decreases suggests strong selling pressure.
"""

import pandas as pd
import numpy as np


def calculate_volume_sma(data, volume_column='Volume', period=20):
    """
    Calculate Simple Moving Average of volume.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing volume data
    volume_column : str, default 'Volume'
        Column name for volume data
    period : int, default 20
        Number of periods for moving average
    
    Returns:
    --------
    pd.Series
        Series containing volume SMA
    
    Example:
    --------
    >>> df['Volume_SMA'] = calculate_volume_sma(df, period=20)
    """
    if volume_column not in data.columns:
        raise ValueError(f"Column '{volume_column}' not found in DataFrame")
    
    return data[volume_column].rolling(window=period).mean()


def calculate_volume_analysis(data, close_column='Close', volume_column='Volume', period=20):
    """
    Comprehensive volume analysis with multiple indicators.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price and volume data
    close_column : str, default 'Close'
        Column name for closing price
    volume_column : str, default 'Volume'
        Column name for volume data
    period : int, default 20
        Number of periods for moving averages
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with volume analysis metrics
    
    Example:
    --------
    >>> vol_analysis = calculate_volume_analysis(df)
    """
    required_columns = [close_column, volume_column]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Calculate volume moving average
    volume_sma = calculate_volume_sma(data, volume_column, period)
    
    # Calculate relative volume
    relative_volume = data[volume_column] / volume_sma
    
    # Price change
    price_change = data[close_column].pct_change()
    
    # Volume trend
    volume_trend = data[volume_column].pct_change()
    
    result = pd.DataFrame({
        'Volume': data[volume_column],
        'Volume_SMA': volume_sma,
        'Relative_Volume': relative_volume,
        'Volume_Trend': volume_trend,
        'Price_Change': price_change,
        'High_Volume': relative_volume > 1.5,
        'Low_Volume': relative_volume < 0.5,
        'Volume_Signal': np.where(relative_volume > 1.5, 'High',
                                  np.where(relative_volume < 0.5, 'Low', 'Normal'))
    })
    
    return result


def calculate_obv(data, close_column='Close', volume_column='Volume'):
    """
    Calculate On-Balance Volume (OBV).
    
    OBV is a cumulative indicator that adds volume on up days and subtracts
    volume on down days. It helps confirm price trends.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price and volume data
    close_column : str, default 'Close'
        Column name for closing price
    volume_column : str, default 'Volume'
        Column name for volume data
    
    Returns:
    --------
    pd.Series
        Series containing OBV values
    
    Example:
    --------
    >>> df['OBV'] = calculate_obv(df)
    """
    required_columns = [close_column, volume_column]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Calculate price direction
    price_direction = np.sign(data[close_column].diff())
    
    # Calculate OBV
    obv = (price_direction * data[volume_column]).cumsum()
    
    return obv


def calculate_volume_price_trend(data, close_column='Close', volume_column='Volume'):
    """
    Calculate Volume Price Trend (VPT).
    
    VPT is similar to OBV but uses percentage price changes.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price and volume data
    close_column : str, default 'Close'
        Column name for closing price
    volume_column : str, default 'Volume'
        Column name for volume data
    
    Returns:
    --------
    pd.Series
        Series containing VPT values
    
    Example:
    --------
    >>> df['VPT'] = calculate_volume_price_trend(df)
    """
    required_columns = [close_column, volume_column]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Calculate percentage price change
    pct_change = data[close_column].pct_change()
    
    # Calculate VPT
    vpt = (pct_change * data[volume_column]).cumsum()
    
    return vpt


def calculate_volume_with_divergence(data, close_column='Close', volume_column='Volume', period=14):
    """
    Calculate volume indicators with price-volume divergence detection.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price and volume data
    close_column : str, default 'Close'
        Column name for closing price
    volume_column : str, default 'Volume'
        Column name for volume data
    period : int, default 14
        Number of periods for analysis
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with volume indicators and divergence signals
    
    Example:
    --------
    >>> vol_div = calculate_volume_with_divergence(df)
    """
    # Calculate OBV
    obv = calculate_obv(data, close_column, volume_column)
    obv_sma = obv.rolling(window=period).mean()
    
    # Price trend
    price_sma = data[close_column].rolling(window=period).mean()
    price_trend = data[close_column] > price_sma
    
    # OBV trend
    obv_trend = obv > obv_sma
    
    # Detect divergence
    bullish_divergence = (~price_trend) & obv_trend  # Price down, OBV up
    bearish_divergence = price_trend & (~obv_trend)  # Price up, OBV down
    
    result = pd.DataFrame({
        'OBV': obv,
        'OBV_SMA': obv_sma,
        'Price_Trend_Up': price_trend,
        'OBV_Trend_Up': obv_trend,
        'Bullish_Divergence': bullish_divergence,
        'Bearish_Divergence': bearish_divergence,
        'Divergence_Signal': np.where(bullish_divergence, 'Bullish Divergence',
                                      np.where(bearish_divergence, 'Bearish Divergence', 'No Divergence'))
    })
    
    return result
