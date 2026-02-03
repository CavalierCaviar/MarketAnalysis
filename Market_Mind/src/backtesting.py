"""
Backtesting Engine for Trading Strategies

This module provides a complete backtesting framework to evaluate trading strategies
by simulating trades and calculating performance metrics.

Key Features:
- Realistic trade execution (no look-ahead bias)
- Risk-based position sizing
- Stop-loss and take-profit support
- Slippage modeling
- Comprehensive performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime


class Backtester:
    """
    A professional-grade backtesting engine that simulates trades based on strategy signals
    and calculates performance metrics.
    
    Fixes critical issues:
    - No look-ahead bias (executes on next candle)
    - Risk-based position sizing
    - Slippage and transaction costs
    - Stop-loss and take-profit management
    """
    
    def __init__(self, initial_capital=10000, commission=0.001, slippage=0.0005, 
                 risk_free_rate=0.02, risk_per_trade=0.02):
        """
        Initialize the backtester.
        
        Parameters:
        -----------
        initial_capital : float, default 10000
            Starting capital for backtesting
        commission : float, default 0.001
            Trading commission as a decimal (0.001 = 0.1%)
        slippage : float, default 0.0005
            Price slippage as a decimal (0.0005 = 0.05%)
        risk_free_rate : float, default 0.02
            Annual risk-free rate for Sharpe/Sortino calculations (0.02 = 2%)
        risk_per_trade : float, default 0.02
            Maximum capital to risk per trade (0.02 = 2%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.risk_per_trade = risk_per_trade
        self.results = None
        
    def run(self, data, strategy_function, strategy_params=None, 
            use_stop_loss=True, stop_loss_pct=0.05, take_profit_pct=0.10):
        """
        Run backtest on historical data using a strategy function.
        
        IMPORTANT EXECUTION MODEL:
        - Signals are generated using CLOSE prices (indicators computed on close)
        - Trades execute at NEXT SESSION OPEN (realistic, no look-ahead bias)
        - Equity is marked-to-market using CLOSE prices
        - Stop-loss/take-profit checked against intraday HIGH/LOW
        
        This prevents look-ahead bias while simulating realistic execution.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with OHLCV data and indicators
            Must have columns: Open, High, Low, Close, Volume
        strategy_function : callable
            Function that takes data slice and returns signal: 'BUY', 'SELL', or 'HOLD'
            Should return None or empty string if not enough data for indicators
        strategy_params : dict, optional
            Additional parameters to pass to strategy function
        use_stop_loss : bool, default True
            Whether to use stop-loss and take-profit
        stop_loss_pct : float, default 0.05
            Stop-loss percentage (0.05 = 5%)
        take_profit_pct : float, default 0.10
            Take-profit percentage (0.10 = 10%)
        
        Returns:
        --------
        dict
            Dictionary containing backtest results and performance metrics
        """
        if strategy_params is None:
            strategy_params = {}
        
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        
        # Initialize tracking variables
        equity = self.initial_capital
        position = 0  # Number of shares held
        cash = self.initial_capital
        entry_price = 0
        entry_cost = 0  # Total cost including commission
        entry_date = None  # Track when position was opened
        stop_loss_price = 0
        take_profit_price = 0
        trades = []
        equity_curve = []
        daily_positions = []
        
        # Track for each day
        for i in range(len(data)):
            current_price = data['Close'].iloc[i]
            current_high = data['High'].iloc[i]
            current_low = data['Low'].iloc[i]
            
            # Get data up to current point (no look-ahead)
            current_data = data.iloc[:i+1].copy()
            
            # Check stop-loss and take-profit FIRST (intraday)
            if position > 0 and use_stop_loss and entry_date is not None:
                # Check if stop-loss hit
                if current_low <= stop_loss_price:
                    # Exit at stop-loss
                    exit_price = stop_loss_price * (1 - self.slippage)  # Slippage on exit
                    proceeds = position * exit_price * (1 - self.commission)
                    cash += proceeds
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': data.index[i] if hasattr(data, 'index') else i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': position,
                        'pnl': proceeds - entry_cost,  # Correct: includes entry commission
                        'exit_reason': 'STOP_LOSS'
                    })
                    
                    position = 0
                    entry_price = 0
                    entry_cost = 0
                    entry_date = None
                    equity = cash
                    equity_curve.append(equity)
                    daily_positions.append(0)
                    continue
                
                # Check if take-profit hit
                elif current_high >= take_profit_price:
                    # Exit at take-profit
                    exit_price = take_profit_price * (1 - self.slippage)  # Slippage on exit
                    proceeds = position * exit_price * (1 - self.commission)
                    cash += proceeds
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': data.index[i] if hasattr(data, 'index') else i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': position,
                        'pnl': proceeds - entry_cost,  # Correct: includes entry commission
                        'exit_reason': 'TAKE_PROFIT'
                    })
                    
                    position = 0
                    entry_price = 0
                    entry_cost = 0
                    entry_date = None
                    equity = cash
                    equity_curve.append(equity)
                    daily_positions.append(0)
                    continue
            
            # Generate signal for TOMORROW (no look-ahead)
            try:
                signal = strategy_function(current_data, **strategy_params)
            except:
                # Strategy not ready (not enough data for indicators)
                signal = None
            
            # If not enough data or HOLD signal, skip
            if signal in (None, "", "HOLD"):
                equity_curve.append(equity)
                daily_positions.append(position)
                continue
            
            # Execute on NEXT candle (if available)
            if i + 1 >= len(data):
                # Last day - just update equity
                equity = cash + position * current_price
                equity_curve.append(equity)
                daily_positions.append(position)
                continue
            
            # Execution price is NEXT candle's open
            execution_price = data['Open'].iloc[i + 1]
            
            # Process signal
            if signal == "BUY" and position == 0:
                # Enter long position
                # Apply slippage (simulate adverse price movement)
                actual_price = execution_price * (1 + self.slippage)
                
                # Calculate position size based on risk
                if use_stop_loss:
                    # Risk-based sizing
                    stop_distance = actual_price * stop_loss_pct
                    risk_amount = cash * self.risk_per_trade
                    shares_to_buy = int(risk_amount / stop_distance)
                    
                    # Cap at available cash
                    max_shares = int(cash / (actual_price * (1 + self.commission)))
                    shares_to_buy = min(shares_to_buy, max_shares)
                else:
                    # Use 95% of cash (keep some buffer)
                    shares_to_buy = int((cash * 0.95) / (actual_price * (1 + self.commission)))
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * actual_price * (1 + self.commission)
                    
                    if cost <= cash:
                        cash -= cost
                        position = shares_to_buy
                        entry_price = actual_price
                        entry_cost = cost  # Store total cost including commission
                        entry_date = data.index[i+1] if hasattr(data, 'index') else i+1
                        
                        # Set stop-loss and take-profit
                        if use_stop_loss:
                            stop_loss_price = entry_price * (1 - stop_loss_pct)
                            take_profit_price = entry_price * (1 + take_profit_pct)
            
            elif signal == "SELL" and position > 0:
                # Exit long position
                # Apply slippage
                actual_price = execution_price * (1 - self.slippage)
                proceeds = position * actual_price * (1 - self.commission)
                cash += proceeds
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': data.index[i+1] if hasattr(data, 'index') else i+1,
                    'entry_price': entry_price,
                    'exit_price': actual_price,
                    'shares': position,
                    'pnl': proceeds - entry_cost,  # Correct: includes entry commission
                    'exit_reason': 'SIGNAL'
                })
                
                position = 0
                entry_price = 0
                entry_cost = 0
                entry_date = None
            
            # Calculate current equity
            equity = cash + position * current_price
            equity_curve.append(equity)
            daily_positions.append(position)
        
        # Close any open position at the end
        if position > 0 and entry_date is not None:
            final_price = data['Close'].iloc[-1] * (1 - self.slippage)
            proceeds = position * final_price * (1 - self.commission)
            cash += proceeds
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': data.index[-1] if hasattr(data, 'index') else len(data) - 1,
                'entry_price': entry_price,
                'exit_price': final_price,
                'shares': position,
                'pnl': proceeds - entry_cost,  # Correct: includes entry commission
                'exit_reason': 'END_OF_DATA'
            })
            
            position = 0
            equity = cash
        
        # Calculate metrics
        metrics = self._calculate_metrics(equity_curve, trades, data)
        
        # Store results
        self.results = {
            'metrics': metrics,
            'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
            'equity_curve': pd.Series(equity_curve, index=data.index if hasattr(data, 'index') else range(len(equity_curve))),
            'daily_positions': pd.Series(daily_positions, index=data.index if hasattr(data, 'index') else range(len(daily_positions))),
            'final_equity': equity,
            'initial_capital': self.initial_capital
        }
        
        return self.results
    
    def _calculate_metrics(self, equity_curve, trades, data):
        """Calculate performance metrics."""
        equity_series = pd.Series(equity_curve)
        
        # Total Return
        total_return = (equity_series.iloc[-1] / self.initial_capital) - 1
        
        # Annualized Return
        trading_days = len(equity_series)
        years = trading_days / 252  # Assuming 252 trading days per year
        annualized_return = ((1 + total_return) ** (1 / years)) - 1 if years > 0 else 0
        
        # Maximum Drawdown
        cumulative_max = equity_series.expanding().max()
        drawdown = (equity_series - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        
        # Daily returns
        daily_returns = equity_series.pct_change().dropna()
        
        # Sharpe Ratio
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1
            sharpe_ratio = (daily_returns.mean() - daily_rf) / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Sortino Ratio (only penalize downside volatility)
        if len(daily_returns) > 0:
            daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            sortino_ratio = (daily_returns.mean() - daily_rf) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        else:
            sortino_ratio = 0
        
        # Trade-based metrics
        if len(trades) > 0:
            trade_pnl = [t['pnl'] for t in trades]
            
            # Win Rate
            winning_trades = sum(1 for pnl in trade_pnl if pnl > 0)
            total_trades = len(trade_pnl)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Profit Factor
            gross_profit = sum(pnl for pnl in trade_pnl if pnl > 0)
            gross_loss = abs(sum(pnl for pnl in trade_pnl if pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)
            
            # Average trade metrics
            avg_trade = np.mean(trade_pnl)
            avg_win = np.mean([pnl for pnl in trade_pnl if pnl > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([pnl for pnl in trade_pnl if pnl < 0]) if (total_trades - winning_trades) > 0 else 0
            
            # Expectancy
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            
            # Exit reasons
            exit_reasons = {}
            for t in trades:
                reason = t.get('exit_reason', 'UNKNOWN')
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade = 0
            avg_win = 0
            avg_loss = 0
            total_trades = 0
            expectancy = 0
            exit_reasons = {}
        
        # Compile metrics
        metrics = {
            'Total Return': total_return,
            'Total Return %': total_return * 100,
            'Annualized Return': annualized_return,
            'Annualized Return %': annualized_return * 100,
            'Max Drawdown': max_drawdown,
            'Max Drawdown %': max_drawdown * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Win Rate': win_rate,
            'Win Rate %': win_rate * 100,
            'Profit Factor': profit_factor,
            'Total Trades': total_trades,
            'Average Trade': avg_trade,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Expectancy': expectancy,
            'Exit Reasons': exit_reasons,
            'Initial Capital': self.initial_capital,
            'Final Equity': equity_series.iloc[-1],
            'Total Profit/Loss': equity_series.iloc[-1] - self.initial_capital
        }
        
        return metrics
    
    def print_results(self):
        """Print formatted backtest results."""
        if self.results is None:
            print("No backtest results available. Run backtest first.")
            return
        
        metrics = self.results['metrics']
        
        print("\n" + "=" * 70)
        print("BACKTESTING RESULTS")
        print("=" * 70)
        
        print(f"\n{'CAPITAL METRICS':-^70}")
        print(f"Initial Capital:        ${metrics['Initial Capital']:,.2f}")
        print(f"Final Equity:           ${metrics['Final Equity']:,.2f}")
        print(f"Total Profit/Loss:      ${metrics['Total Profit/Loss']:,.2f}")
        
        print(f"\n{'RETURN METRICS':-^70}")
        print(f"Total Return:           {metrics['Total Return %']:>8.2f}%")
        print(f"Annualized Return:      {metrics['Annualized Return %']:>8.2f}%")
        
        print(f"\n{'RISK METRICS':-^70}")
        print(f"Max Drawdown:           {metrics['Max Drawdown %']:>8.2f}%")
        print(f"Sharpe Ratio:           {metrics['Sharpe Ratio']:>8.2f}")
        print(f"Sortino Ratio:          {metrics['Sortino Ratio']:>8.2f}")
        
        print(f"\n{'TRADE METRICS':-^70}")
        print(f"Total Trades:           {metrics['Total Trades']:>8.0f}")
        print(f"Win Rate:               {metrics['Win Rate %']:>8.2f}%")
        print(f"Profit Factor:          {metrics['Profit Factor']:>8.2f}")
        print(f"Average Trade:          ${metrics['Average Trade']:>8.2f}")
        print(f"Average Win:            ${metrics['Average Win']:>8.2f}")
        print(f"Average Loss:           ${metrics['Average Loss']:>8.2f}")
        print(f"Expectancy per Trade:   ${metrics['Expectancy']:>8.2f}")
        
        # Show exit reasons if any trades
        if metrics['Exit Reasons']:
            print(f"\n{'EXIT REASONS':-^70}")
            for reason, count in metrics['Exit Reasons'].items():
                pct = (count / metrics['Total Trades']) * 100
                print(f"{reason:20} {count:>5} trades ({pct:>5.1f}%)")
        
        print("\n" + "=" * 70)
        
        # Performance assessment
        print(f"\n{'PERFORMANCE ASSESSMENT':-^70}")
        self._assess_performance(metrics)
        print("=" * 70)
    
    def _assess_performance(self, metrics):
        """Provide qualitative assessment of strategy performance."""
        assessments = []
        
        # Return assessment
        if metrics['Annualized Return %'] > 15:
            assessments.append("✅ Excellent annualized returns")
        elif metrics['Annualized Return %'] > 8:
            assessments.append("✓ Good annualized returns")
        elif metrics['Annualized Return %'] > 0:
            assessments.append("⚠ Modest positive returns")
        else:
            assessments.append("❌ Negative returns")
        
        # Drawdown assessment
        if abs(metrics['Max Drawdown %']) < 10:
            assessments.append("✅ Low drawdown - well-controlled risk")
        elif abs(metrics['Max Drawdown %']) < 20:
            assessments.append("✓ Moderate drawdown")
        else:
            assessments.append("⚠ High drawdown - significant risk")
        
        # Sharpe ratio assessment
        if metrics['Sharpe Ratio'] > 2:
            assessments.append("✅ Excellent risk-adjusted returns")
        elif metrics['Sharpe Ratio'] > 1:
            assessments.append("✓ Good risk-adjusted returns")
        elif metrics['Sharpe Ratio'] > 0:
            assessments.append("⚠ Marginal risk-adjusted returns")
        else:
            assessments.append("❌ Poor risk-adjusted returns")
        
        # Win rate assessment
        if metrics['Win Rate %'] > 60:
            assessments.append("✅ High win rate")
        elif metrics['Win Rate %'] > 50:
            assessments.append("✓ Decent win rate")
        else:
            assessments.append("⚠ Low win rate")
        
        # Profit factor assessment
        if metrics['Profit Factor'] > 2:
            assessments.append("✅ Excellent profit factor")
        elif metrics['Profit Factor'] > 1.5:
            assessments.append("✓ Good profit factor")
        elif metrics['Profit Factor'] > 1:
            assessments.append("⚠ Marginal profit factor")
        else:
            assessments.append("❌ Unprofitable (profit factor < 1)")
        
        for assessment in assessments:
            print(f"  {assessment}")
    
    def get_trades_df(self):
        """Return trades as a DataFrame."""
        if self.results is None:
            return pd.DataFrame()
        return self.results['trades']
    
    def get_equity_curve(self):
        """Return equity curve as a Series."""
        if self.results is None:
            return pd.Series()
        return self.results['equity_curve']
    
    def plot_equity_curve(self):
        """Plot the equity curve over time."""
        try:
            import matplotlib.pyplot as plt
            
            if self.results is None:
                print("No backtest results available. Run backtest first.")
                return
            
            equity_curve = self.results['equity_curve']
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Plot equity curve
            ax.plot(range(len(equity_curve)), equity_curve.values, linewidth=2, label='Strategy Equity')
            ax.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
            
            # Mark trades on the chart (exits only)
            trades_df = self.results['trades']
            if not trades_df.empty:
                for _, trade in trades_df.iterrows():
                    # Get exit index from equity curve
                    if trade['exit_date'] in equity_curve.index:
                        exit_idx = equity_curve.index.get_loc(trade['exit_date'])
                        exit_equity = equity_curve.iloc[exit_idx]
                        color = 'green' if trade['pnl'] > 0 else 'red'
                        marker = '^' if trade['pnl'] > 0 else 'v'
                        ax.scatter(exit_idx, exit_equity, color=color, marker=marker, s=100, zorder=5, alpha=0.7)
            
            ax.set_title('Backtest Equity Curve', fontsize=16, fontweight='bold')
            ax.set_xlabel('Trading Days', fontsize=12)
            ax.set_ylabel('Equity ($)', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib is required for plotting. Install with: pip install matplotlib")


def convert_multi_signal_strategy(old_strategy_function):
    """
    Convert old-style strategy (returns list of signals) to new format (returns single signal).
    
    This is a compatibility wrapper for strategies that return:
    [(indicator, status, signal), ...]
    
    Parameters:
    -----------
    old_strategy_function : callable
        Old-style strategy that returns list of (indicator, status, signal) tuples
    
    Returns:
    --------
    callable
        New-style strategy that returns 'BUY', 'SELL', or 'HOLD'
    """
    def new_strategy(data, **kwargs):
        try:
            signals = old_strategy_function(data, **kwargs)
            
            if not signals:
                return None
            
            # Count signal types
            buy_signals = sum(1 for _, _, s in signals if s == "BUY")
            sell_signals = sum(1 for _, _, s in signals if s == "SELL")
            
            # Decision logic
            if buy_signals > sell_signals + 1:
                return "BUY"
            elif sell_signals > buy_signals + 1:
                return "SELL"
            else:
                return "HOLD"
        except:
            # Not enough data for indicators
            return None
    
    return new_strategy


def quick_backtest(data, strategy_function, initial_capital=10000, 
                   use_stop_loss=True, print_report=True):
    """
    Convenience function to quickly run a backtest.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Stock data with OHLCV and indicators
    strategy_function : callable
        Strategy function that returns 'BUY', 'SELL', or 'HOLD'
        (or old-style that returns list of signals - will be auto-converted)
    initial_capital : float, default 10000
        Starting capital
    use_stop_loss : bool, default True
        Whether to use stop-loss and take-profit
    print_report : bool, default True
        Whether to print the results report
    
    Returns:
    --------
    Backtester
        Backtester instance with results
    """
    bt = Backtester(initial_capital=initial_capital)
    
    # Auto-detect if old-style strategy (returns list) and convert
    try:
        test_result = strategy_function(data)
        if isinstance(test_result, list):
            print("⚠️  Detected old-style strategy - auto-converting to new format")
            strategy_function = convert_multi_signal_strategy(strategy_function)
    except:
        pass
    
    bt.run(data, strategy_function, use_stop_loss=use_stop_loss)
    
    if print_report:
        bt.print_results()
    
    return bt
