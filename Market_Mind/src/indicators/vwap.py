"""
VWAP (Volume Weighted Average Price) Indicator

VWAP is the average price a security has traded at throughout the day, based on
both volume and price. It's used as a benchmark to determine if a stock is trading
at a good price relative to its average price for the day.
"""

import pandas as pd
import numpy as np


def calculate_vwap(data, high_column='High', low_column='Low', close_column='Close', volume_column='Volume'):
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price and volume data
    high_column : str, default 'High'
        Column name for high price
    low_column : str, default 'Low'
        Column name for low price
    close_column : str, default 'Close'
        Column name for closing price
    volume_column : str, default 'Volume'
        Column name for volume data
    
    Returns:
    --------
    pd.Series
        Series containing VWAP values
    
    Example:
    --------
    >>> df['VWAP'] = calculate_vwap(df)
    """
    required_columns = [high_column, low_column, close_column, volume_column]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Calculate typical price
    typical_price = (data[high_column] + data[low_column] + data[close_column]) / 3
    
    # Calculate VWAP
    vwap = (typical_price * data[volume_column]).cumsum() / data[volume_column].cumsum()
    
    return vwap


def calculate_vwap_daily(data, high_column='High', low_column='Low', close_column='Close', 
                         volume_column='Volume', date_column=None):
    """
    Calculate VWAP reset daily (for intraday data).
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price and volume data with DatetimeIndex or date column
    high_column : str, default 'High'
        Column name for high price
    low_column : str, default 'Low'
        Column name for low price
    close_column : str, default 'Close'
        Column name for closing price
    volume_column : str, default 'Volume'
        Column name for volume data
    date_column : str, optional
        Column name for date (if not using DatetimeIndex)
    
    Returns:
    --------
    pd.Series
        Series containing daily VWAP values
    
    Example:
    --------
    >>> df['VWAP_Daily'] = calculate_vwap_daily(df)
    """
    required_columns = [high_column, low_column, close_column, volume_column]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Calculate typical price
    typical_price = (data[high_column] + data[low_column] + data[close_column]) / 3
    
    # Determine date grouping
    if date_column:
        date_group = pd.to_datetime(data[date_column]).dt.date
    elif isinstance(data.index, pd.DatetimeIndex):
        date_group = data.index.date
    else:
        # If no date information, calculate cumulative VWAP
        return calculate_vwap(data, high_column, low_column, close_column, volume_column)
    
    # Calculate daily VWAP
    data_copy = data.copy()
    data_copy['_date'] = date_group
    data_copy['_tp_volume'] = typical_price * data[volume_column]
    
    vwap = data_copy.groupby('_date').apply(
        lambda x: x['_tp_volume'].cumsum() / data[volume_column].loc[x.index].cumsum()
    )
    
    # Flatten the result
    if isinstance(vwap, pd.DataFrame):
        vwap = vwap.iloc[:, 0]
    
    return vwap.values


def calculate_vwap_bands(data, high_column='High', low_column='Low', close_column='Close', 
                        volume_column='Volume', std_dev=1):
    """
    Calculate VWAP with standard deviation bands.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price and volume data
    high_column : str, default 'High'
        Column name for high price
    low_column : str, default 'Low'
        Column name for low price
    close_column : str, default 'Close'
        Column name for closing price
    volume_column : str, default 'Volume'
        Column name for volume data
    std_dev : float, default 1
        Number of standard deviations for bands
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with VWAP and bands
    
    Example:
    --------
    >>> vwap_bands = calculate_vwap_bands(df, std_dev=2)
    """
    # Calculate VWAP
    vwap = calculate_vwap(data, high_column, low_column, close_column, volume_column)
    
    # Calculate typical price
    typical_price = (data[high_column] + data[low_column] + data[close_column]) / 3
    
    # Calculate variance
    cumulative_volume = data[volume_column].cumsum()
    variance = ((typical_price - vwap) ** 2 * data[volume_column]).cumsum() / cumulative_volume
    std = np.sqrt(variance)
    
    result = pd.DataFrame({
        'VWAP': vwap,
        'VWAP_Upper_1': vwap + std,
        'VWAP_Lower_1': vwap - std,
        'VWAP_Upper_2': vwap + (2 * std),
        'VWAP_Lower_2': vwap - (2 * std),
        'Distance_from_VWAP': typical_price - vwap,
        'Distance_Pct': ((typical_price - vwap) / vwap) * 100
    })
    
    return result


def calculate_vwap_with_signals(data, high_column='High', low_column='Low', 
                                close_column='Close', volume_column='Volume'):
    """
    Calculate VWAP with trading signals.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price and volume data
    high_column : str, default 'High'
        Column name for high price
    low_column : str, default 'Low'
        Column name for low price
    close_column : str, default 'Close'
        Column name for closing price
    volume_column : str, default 'Volume'
        Column name for volume data
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with VWAP and trading signals
    
    Example:
    --------
    >>> vwap_signals = calculate_vwap_with_signals(df)
    """
    # Calculate VWAP
    vwap = calculate_vwap(data, high_column, low_column, close_column, volume_column)
    
    # Price position relative to VWAP
    above_vwap = data[close_column] > vwap
    
    # VWAP crossovers
    bullish_cross = (data[close_column] > vwap) & (data[close_column].shift(1) <= vwap.shift(1))
    bearish_cross = (data[close_column] < vwap) & (data[close_column].shift(1) >= vwap.shift(1))
    
    result = pd.DataFrame({
        'VWAP': vwap,
        'Price': data[close_column],
        'Above_VWAP': above_vwap,
        'Bullish_Cross': bullish_cross,
        'Bearish_Cross': bearish_cross,
        'Signal': np.where(bullish_cross, 'Buy',
                          np.where(bearish_cross, 'Sell', 'Hold')),
        'Position': np.where(above_vwap, 'Above VWAP (Bullish)', 'Below VWAP (Bearish)')
    })
    
    return result
