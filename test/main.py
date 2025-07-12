import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def getMyPosition(prcSoFar):
    """
    Main function for the trading competition.
    
    Args:
        prcSoFar: numpy array of shape (nInst, nt) containing price data
                 where nInst=50 instruments and nt=days so far
    
    Returns:
        numpy array of 50 integers representing desired positions
    """
    try:
        # Initialize positions
        positions = np.zeros(50, dtype=int)
        
        # Need at least 60 days of data for meaningful analysis
        if prcSoFar.shape[1] < 60:
            return positions
        
        # Get current prices and recent price data
        current_prices = prcSoFar[:, -1]
        recent_data = prcSoFar[:, -120:]  # Last 120 days for analysis
        
        # Find pairs and generate trading signals
        pairs_positions = pairs_trading_strategy(recent_data, current_prices)
        
        # Apply position limits ($10k per stock)
        for i in range(50):
            max_shares = int(10000 / current_prices[i])
            pairs_positions[i] = np.clip(pairs_positions[i], -max_shares, max_shares)
        
        return pairs_positions.astype(int)
    
    except Exception as e:
        print(f"Error in getMyPosition: {e}")
        return np.zeros(50, dtype=int)

def pairs_trading_strategy(price_data, current_prices, lookback_corr=60, lookback_zscore=30):
    """
    Implements a pairs trading strategy based on correlation and mean reversion.
    
    Args:
        price_data: numpy array of recent price data (50 x days)
        current_prices: current prices for position sizing
        lookback_corr: days to look back for correlation calculation
        lookback_zscore: days to look back for z-score calculation
    
    Returns:
        numpy array of positions for each instrument
    """
    n_instruments = price_data.shape[0]
    n_days = price_data.shape[1]
    positions = np.zeros(n_instruments)
    
    if n_days < lookback_corr:
        return positions
    
    # Calculate returns for correlation analysis
    returns = np.diff(np.log(price_data + 1e-8), axis=1)
    recent_returns = returns[:, -lookback_corr:]
    
    # Find highly correlated pairs
    correlation_matrix = np.corrcoef(recent_returns)
    pairs = find_trading_pairs(correlation_matrix, threshold=0.7)
    
    # For each pair, calculate trading signals
    for stock_a, stock_b in pairs:
        try:
            # Get recent price data for the pair
            prices_a = price_data[stock_a, -lookback_zscore:]
            prices_b = price_data[stock_b, -lookback_zscore:]
            
            # Calculate the spread and its z-score
            spread, z_score = calculate_spread_zscore(prices_a, prices_b)
            
            # Generate trading signals based on z-score
            position_a, position_b = generate_pair_signals(
                z_score, current_prices[stock_a], current_prices[stock_b]
            )
            
            # Update positions (average if multiple pairs involve same stock)
            if positions[stock_a] == 0:
                positions[stock_a] = position_a
            else:
                positions[stock_a] = (positions[stock_a] + position_a) / 2
                
            if positions[stock_b] == 0:
                positions[stock_b] = position_b
            else:
                positions[stock_b] = (positions[stock_b] + position_b) / 2
                
        except Exception as e:
            continue
    
    return positions

def find_trading_pairs(correlation_matrix, threshold=0.7, max_pairs=10):
    """
    Find pairs of stocks with high correlation for pairs trading.
    
    Args:
        correlation_matrix: correlation matrix of stock returns
        threshold: minimum correlation threshold
        max_pairs: maximum number of pairs to return
    
    Returns:
        list of tuples representing stock pairs
    """
    n_stocks = correlation_matrix.shape[0]
    pairs = []
    
    # Find pairs with correlation above threshold
    for i in range(n_stocks):
        for j in range(i + 1, n_stocks):
            if abs(correlation_matrix[i, j]) > threshold:
                pairs.append((i, j, abs(correlation_matrix[i, j])))
    
    # Sort by correlation strength and return top pairs
    pairs.sort(key=lambda x: x[2], reverse=True)
    return [(pair[0], pair[1]) for pair in pairs[:max_pairs]]

def calculate_spread_zscore(prices_a, prices_b, window=20):
    """
    Calculate the spread between two price series and its z-score.
    
    Args:
        prices_a, prices_b: price series for the two stocks
        window: rolling window for z-score calculation
    
    Returns:
        tuple of (current_spread, z_score)
    """
    # Use linear regression to find the hedge ratio
    X = prices_b.reshape(-1, 1)
    y = prices_a
    
    # Fit linear regression
    reg = LinearRegression().fit(X, y)
    hedge_ratio = reg.coef_[0]
    
    # Calculate spread
    spread = prices_a - hedge_ratio * prices_b
    
    # Calculate rolling z-score
    if len(spread) >= window:
        rolling_mean = np.mean(spread[-window:])
        rolling_std = np.std(spread[-window:])
        
        if rolling_std > 0:
            z_score = (spread[-1] - rolling_mean) / rolling_std
        else:
            z_score = 0
    else:
        z_score = 0
    
    return spread[-1], z_score

def generate_pair_signals(z_score, price_a, price_b, entry_threshold=2.0, exit_threshold=0.5):
    """
    Generate trading signals based on z-score of the spread.
    
    Args:
        z_score: current z-score of the spread
        price_a, price_b: current prices of the two stocks
        entry_threshold: z-score threshold for entering positions
        exit_threshold: z-score threshold for exiting positions
    
    Returns:
        tuple of (position_a, position_b)
    """
    position_a = 0
    position_b = 0
    
    # Position sizing based on volatility (simplified)
    base_position = min(1000, int(5000 / price_a), int(5000 / price_b))
    
    if z_score > entry_threshold:
        # Spread is too high: short A, long B
        position_a = -base_position
        position_b = base_position
    elif z_score < -entry_threshold:
        # Spread is too low: long A, short B
        position_a = base_position
        position_b = -base_position
    elif abs(z_score) < exit_threshold:
        # Close positions when spread reverts to mean
        position_a = 0
        position_b = 0
    
    return position_a, position_b

def calculate_portfolio_metrics(returns):
    """
    Calculate portfolio performance metrics.
    
    Args:
        returns: array of portfolio returns
    
    Returns:
        dict with performance metrics
    """
    if len(returns) == 0:
        return {'sharpe': 0, 'max_drawdown': 0, 'total_return': 0}
    
    # Calculate Sharpe ratio (assuming daily returns)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
    
    # Calculate maximum drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Total return
    total_return = cumulative[-1] - 1 if len(cumulative) > 0 else 0
    
    return {
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'total_return': total_return,
        'volatility': std_return * np.sqrt(252)
    }

# Additional utility functions for analysis
def backtest_strategy(price_data, start_day=120):
    """
    Backtest the pairs trading strategy.
    
    Args:
        price_data: full price dataset
        start_day: day to start backtesting from
    
    Returns:
        dict with backtest results
    """
    n_days = price_data.shape[1]
    portfolio_values = []
    positions_history = []
    
    current_positions = np.zeros(50)
    cash = 100000  # Starting cash
    
    for day in range(start_day, n_days):
        # Get price data up to current day
        data_so_far = price_data[:, :day+1]
        current_prices = data_so_far[:, -1]
        
        # Get new positions
        new_positions = getMyPosition(data_so_far)
        
        # Calculate trades and costs
        trades = new_positions - current_positions
        trade_costs = np.sum(np.abs(trades * current_prices)) * 0.0005  # 5 bps commission
        
        # Update portfolio
        portfolio_value = cash + np.sum(current_positions * current_prices)
        portfolio_values.append(portfolio_value)
        positions_history.append(current_positions.copy())
        
        # Update positions and cash
        cash -= np.sum(trades * current_prices) + trade_costs
        current_positions = new_positions.copy()
    
    # Calculate returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    metrics = calculate_portfolio_metrics(returns)
    
    return {
        'portfolio_values': portfolio_values,
        'returns': returns,
        'positions_history': positions_history,
        'metrics': metrics
    }