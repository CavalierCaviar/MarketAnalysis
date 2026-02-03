"""
Bollinger Bands Indicator

Bollinger Bands consist of a middle band (SMA) with an upper and lower band
that are typically 2 standard deviations away from the middle band.
They help identify volatility and potential overbought/oversold conditions.
"""

import pandas as pd
import numpy as np


def calculate_bollinger_bands(data, column='Close', period=20, std_dev=2):
    """
    Calculate Bollinger Bands for a given dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    column : str, default 'Close'
        Column name to use for Bollinger Bands calculation
    period : int, default 20
        Number of periods for the moving average
    std_dev : float, default 2
        Number of standard deviations for the bands
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing Middle Band, Upper Band, and Lower Band
    
    Example:
    --------
    >>> bb = calculate_bollinger_bands(df, column='Close', period=20)
    >>> df['BB_Middle'] = bb['Middle_Band']
    >>> df['BB_Upper'] = bb['Upper_Band']
    >>> df['BB_Lower'] = bb['Lower_Band']
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    # Calculate middle band (SMA)
    middle_band = data[column].rolling(window=period).mean()
    
    # Calculate standard deviation
    rolling_std = data[column].rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    # Create result DataFrame
    result = pd.DataFrame({
        'Middle_Band': middle_band,
        'Upper_Band': upper_band,
        'Lower_Band': lower_band,
        'Bandwidth': upper_band - lower_band,
        '%B': (data[column] - lower_band) / (upper_band - lower_band)
    })
    
    return result


def calculate_bollinger_squeeze(data, column='Close', period=20, std_dev=2, squeeze_threshold=0.05):
    """
    Identify Bollinger Band squeeze conditions (low volatility).
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    column : str, default 'Close'
        Column name to use for calculation
    period : int, default 20
        Number of periods for the moving average
    std_dev : float, default 2
        Number of standard deviations for the bands
    squeeze_threshold : float, default 0.05
        Threshold for identifying squeeze (as percentage of middle band)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with Bollinger Bands and squeeze indicator
    
    Example:
    --------
    >>> bb_squeeze = calculate_bollinger_squeeze(df)
    """
    bb = calculate_bollinger_bands(data, column, period, std_dev)
    
    # Calculate bandwidth as percentage of middle band
    bandwidth_pct = bb['Bandwidth'] / bb['Middle_Band']
    
    # Identify squeeze
    bb['Squeeze'] = bandwidth_pct < squeeze_threshold
    bb['Bandwidth_Pct'] = bandwidth_pct
    
    # Identify position relative to bands
    bb['Position'] = np.where(data[column] > bb['Upper_Band'], 'Above Upper',
                              np.where(data[column] < bb['Lower_Band'], 'Below Lower', 'Within Bands'))
    
    return bb
