"""
Fibonacci Retracement Indicator

Fibonacci retracement levels are horizontal lines that indicate potential support
and resistance levels based on Fibonacci ratios. These levels are derived from the
Fibonacci sequence and are commonly used to identify potential reversal points.
"""

import pandas as pd
import numpy as np


def calculate_fibonacci_retracement(data, high=None, low=None, price_column='Close', 
                                    trend='uptrend', custom_levels=None):
    """
    Calculate Fibonacci retracement levels.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    high : float, optional
        Highest price point (if None, uses max of data)
    low : float, optional
        Lowest price point (if None, uses min of data)
    price_column : str, default 'Close'
        Column name for price data
    trend : str, default 'uptrend'
        'uptrend' or 'downtrend' - determines level calculation direction
    custom_levels : list, optional
        Custom Fibonacci levels (default: [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1])
    
    Returns:
    --------
    dict
        Dictionary containing Fibonacci levels and their values
    
    Example:
    --------
    >>> fib_levels = calculate_fibonacci_retracement(df, trend='uptrend')
    >>> print(fib_levels['levels'])
    """
    if price_column not in data.columns:
        raise ValueError(f"Column '{price_column}' not found in DataFrame")
    
    # Default Fibonacci levels
    if custom_levels is None:
        levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    else:
        levels = custom_levels
    
    # Determine high and low points
    if high is None:
        high = data[price_column].max()
    if low is None:
        low = data[price_column].min()
    
    # Calculate difference
    diff = high - low
    
    # Calculate Fibonacci levels
    fib_levels = {}
    
    if trend.lower() == 'uptrend':
        # For uptrend, levels are calculated from high going down
        for level in levels:
            fib_levels[f'{level:.1%}'] = high - (diff * level)
    else:  # downtrend
        # For downtrend, levels are calculated from low going up
        for level in levels:
            fib_levels[f'{level:.1%}'] = low + (diff * level)
    
    return {
        'high': high,
        'low': low,
        'difference': diff,
        'trend': trend,
        'levels': fib_levels
    }


def calculate_fibonacci_extensions(data, high=None, low=None, price_column='Close', 
                                   trend='uptrend'):
    """
    Calculate Fibonacci extension levels (for profit targets).
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    high : float, optional
        Highest price point (if None, uses max of data)
    low : float, optional
        Lowest price point (if None, uses min of data)
    price_column : str, default 'Close'
        Column name for price data
    trend : str, default 'uptrend'
        'uptrend' or 'downtrend'
    
    Returns:
    --------
    dict
        Dictionary containing Fibonacci extension levels
    
    Example:
    --------
    >>> fib_ext = calculate_fibonacci_extensions(df, trend='uptrend')
    """
    if price_column not in data.columns:
        raise ValueError(f"Column '{price_column}' not found in DataFrame")
    
    # Extension levels
    extension_levels = [1.272, 1.414, 1.618, 2.0, 2.618]
    
    # Determine high and low points
    if high is None:
        high = data[price_column].max()
    if low is None:
        low = data[price_column].min()
    
    # Calculate difference
    diff = high - low
    
    # Calculate extension levels
    ext_levels = {}
    
    if trend.lower() == 'uptrend':
        # For uptrend, extensions are above the high
        for level in extension_levels:
            ext_levels[f'{level:.3f}'] = high + (diff * (level - 1))
    else:  # downtrend
        # For downtrend, extensions are below the low
        for level in extension_levels:
            ext_levels[f'{level:.3f}'] = low - (diff * (level - 1))
    
    return {
        'high': high,
        'low': low,
        'difference': diff,
        'trend': trend,
        'extension_levels': ext_levels
    }


def calculate_fibonacci_with_swing_points(data, price_column='Close', lookback=20):
    """
    Automatically identify swing high/low and calculate Fibonacci levels.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    price_column : str, default 'Close'
        Column name for price data
    lookback : int, default 20
        Number of periods to look back for swing points
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with Fibonacci levels and current price position
    
    Example:
    --------
    >>> fib_auto = calculate_fibonacci_with_swing_points(df, lookback=20)
    """
    if price_column not in data.columns:
        raise ValueError(f"Column '{price_column}' not found in DataFrame")
    
    # Find swing high and low in lookback period
    recent_data = data[price_column].tail(lookback)
    swing_high = recent_data.max()
    swing_low = recent_data.min()
    
    # Determine trend
    current_price = data[price_column].iloc[-1]
    trend = 'uptrend' if current_price > (swing_high + swing_low) / 2 else 'downtrend'
    
    # Calculate Fibonacci levels
    fib = calculate_fibonacci_retracement(data, swing_high, swing_low, price_column, trend)
    
    # Create result DataFrame with levels
    result_data = []
    for level_name, level_value in fib['levels'].items():
        result_data.append({
            'Level': level_name,
            'Price': level_value,
            'Distance_from_Current': level_value - current_price,
            'Distance_Pct': ((level_value - current_price) / current_price) * 100
        })
    
    result_df = pd.DataFrame(result_data)
    
    return {
        'swing_high': swing_high,
        'swing_low': swing_low,
        'current_price': current_price,
        'trend': trend,
        'levels_df': result_df,
        'levels': fib['levels']
    }


def identify_fibonacci_support_resistance(data, price_column='Close', high=None, low=None, 
                                          trend='uptrend', tolerance=0.01):
    """
    Identify when price is near Fibonacci levels (potential support/resistance).
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price data
    price_column : str, default 'Close'
        Column name for price data
    high : float, optional
        Highest price point
    low : float, optional
        Lowest price point
    trend : str, default 'uptrend'
        'uptrend' or 'downtrend'
    tolerance : float, default 0.01
        Tolerance for level detection (1% by default)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with price and nearby Fibonacci levels
    
    Example:
    --------
    >>> fib_sr = identify_fibonacci_support_resistance(df, tolerance=0.02)
    """
    # Calculate Fibonacci levels
    fib = calculate_fibonacci_retracement(data, high, low, price_column, trend)
    
    # Create result DataFrame
    result = pd.DataFrame({
        'Price': data[price_column]
    })
    
    # Check proximity to each Fibonacci level
    for level_name, level_value in fib['levels'].items():
        distance_pct = abs(data[price_column] - level_value) / level_value
        result[f'Near_{level_name}'] = distance_pct <= tolerance
    
    # Identify the nearest Fibonacci level
    def find_nearest_level(price):
        min_distance = float('inf')
        nearest_level = None
        for level_name, level_value in fib['levels'].items():
            distance = abs(price - level_value)
            if distance < min_distance:
                min_distance = distance
                nearest_level = level_name
        return nearest_level
    
    result['Nearest_Fib_Level'] = data[price_column].apply(find_nearest_level)
    result['At_Fib_Level'] = result[[col for col in result.columns if col.startswith('Near_')]].any(axis=1)
    
    return result
