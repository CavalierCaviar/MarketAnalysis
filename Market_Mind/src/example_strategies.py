"""
Example Trading Strategies

This module contains example strategies in the correct format for the backtester.

Strategy Function Requirements:
- Takes data (pd.DataFrame) as first argument
- Returns single string: 'BUY', 'SELL', 'HOLD', or None
- Returns None if not enough data for indicators
- Should NOT use future data (no look-ahead bias)
"""

import pandas as pd
import numpy as np


def simple_multi_indicator_strategy(data, rsi_oversold=30, rsi_overbought=70):
    """
    Example strategy combining multiple indicators.
    
    Returns single signal based on majority vote.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Must contain: RSI, MACD, MACD_Signal, EMA_20, EMA_50, ADX, +DI, -DI
    rsi_oversold : float, default 30
        RSI level for oversold condition
    rsi_overbought : float, default 70
        RSI level for overbought condition
    
    Returns:
    --------
    str
        'BUY', 'SELL', 'HOLD', or None if not ready
    """
    # Check if we have required columns
    required = ['RSI', 'MACD', 'MACD_Signal', 'EMA_20', 'EMA_50', 'Close',
                'BB_Upper', 'BB_Lower', 'ADX', '+DI', '-DI']
    
    for col in required:
        if col not in data.columns:
            return None
    
    # Need enough data for all indicators
    if len(data) < 50:
        return None
    
    # Check for NaN in indicators
    latest = data.iloc[-1]
    if latest[required].isnull().any():
        return None
    
    # Collect votes
    buy_votes = 0
    sell_votes = 0
    
    # 1. RSI signal
    if latest['RSI'] < rsi_oversold:
        buy_votes += 2  # Strong signal
    elif latest['RSI'] > rsi_overbought:
        sell_votes += 2
    
    # 2. Bollinger Bands
    if latest['Close'] < latest['BB_Lower']:
        buy_votes += 1
    elif latest['Close'] > latest['BB_Upper']:
        sell_votes += 1
    
    # 3. MACD
    if latest['MACD'] > latest['MACD_Signal']:
        buy_votes += 1
    else:
        sell_votes += 1
    
    # 4. EMA Trend
    if latest['EMA_20'] > latest['EMA_50']:
        buy_votes += 1
    else:
        sell_votes += 1
    
    # 5. ADX (only if strong trend)
    if latest['ADX'] > 25:
        if latest['+DI'] > latest['-DI']:
            buy_votes += 2
        else:
            sell_votes += 2
    
    # Decision
    if buy_votes >= sell_votes + 2:
        return "BUY"
    elif sell_votes >= buy_votes + 2:
        return "SELL"
    else:
        return "HOLD"


def rsi_mean_reversion_strategy(data, rsi_buy=30, rsi_sell=70):
    """
    Simple RSI mean reversion strategy.
    
    Buy when oversold, sell when overbought.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Must contain: RSI
    rsi_buy : float, default 30
        RSI level to trigger buy
    rsi_sell : float, default 70
        RSI level to trigger sell
    
    Returns:
    --------
    str
        'BUY', 'SELL', 'HOLD', or None
    """
    if 'RSI' not in data.columns or len(data) < 20:
        return None
    
    latest_rsi = data['RSI'].iloc[-1]
    
    if pd.isna(latest_rsi):
        return None
    
    if latest_rsi < rsi_buy:
        return "BUY"
    elif latest_rsi > rsi_sell:
        return "SELL"
    else:
        return "HOLD"


def macd_crossover_strategy(data):
    """
    Simple MACD crossover strategy.
    
    Buy when MACD crosses above signal, sell when crosses below.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Must contain: MACD, MACD_Signal
    
    Returns:
    --------
    str
        'BUY', 'SELL', 'HOLD', or None
    """
    if 'MACD' not in data.columns or 'MACD_Signal' not in data.columns:
        return None
    
    if len(data) < 30:
        return None
    
    current_macd = data['MACD'].iloc[-1]
    current_signal = data['MACD_Signal'].iloc[-1]
    prev_macd = data['MACD'].iloc[-2]
    prev_signal = data['MACD_Signal'].iloc[-2]
    
    if pd.isna(current_macd) or pd.isna(current_signal):
        return None
    
    # Bullish crossover
    if prev_macd <= prev_signal and current_macd > current_signal:
        return "BUY"
    # Bearish crossover
    elif prev_macd >= prev_signal and current_macd < current_signal:
        return "SELL"
    else:
        return "HOLD"


def trend_following_ema_strategy(data, fast_period=20, slow_period=50):
    """
    EMA trend following strategy.
    
    Buy when fast EMA crosses above slow EMA, sell when crosses below.
    Only trade with the trend.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Must contain: EMA_20, EMA_50 (or calculate them)
    fast_period : int, default 20
        Fast EMA period
    slow_period : int, default 50
        Slow EMA period
    
    Returns:
    --------
    str
        'BUY', 'SELL', 'HOLD', or None
    """
    ema_fast_col = f'EMA_{fast_period}'
    ema_slow_col = f'EMA_{slow_period}'
    
    if ema_fast_col not in data.columns or ema_slow_col not in data.columns:
        return None
    
    if len(data) < slow_period + 10:
        return None
    
    current_fast = data[ema_fast_col].iloc[-1]
    current_slow = data[ema_slow_col].iloc[-1]
    prev_fast = data[ema_fast_col].iloc[-2]
    prev_slow = data[ema_slow_col].iloc[-2]
    
    if pd.isna(current_fast) or pd.isna(current_slow):
        return None
    
    # Bullish crossover
    if prev_fast <= prev_slow and current_fast > current_slow:
        return "BUY"
    # Bearish crossover
    elif prev_fast >= prev_slow and current_fast < current_slow:
        return "SELL"
    else:
        return "HOLD"


def bollinger_breakout_strategy(data):
    """
    Bollinger Bands breakout strategy.
    
    Buy when price breaks below lower band (mean reversion).
    Sell when price breaks above upper band.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Must contain: Close, BB_Upper, BB_Lower
    
    Returns:
    --------
    str
        'BUY', 'SELL', 'HOLD', or None
    """
    required = ['Close', 'BB_Upper', 'BB_Lower']
    for col in required:
        if col not in data.columns:
            return None
    
    if len(data) < 25:
        return None
    
    latest = data.iloc[-1]
    
    if latest[required].isnull().any():
        return None
    
    # Mean reversion: buy oversold, sell overbought
    if latest['Close'] < latest['BB_Lower']:
        return "BUY"
    elif latest['Close'] > latest['BB_Upper']:
        return "SELL"
    else:
        return "HOLD"
