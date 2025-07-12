import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
from main import getMyPosition

def evaluate_algorithm(price_data, commission_rate=0.0005):
    """
    Evaluate the trading algorithm using the competition's scoring method.
    
    Args:
        price_data: numpy array of price data (nInst x nDays)
        commission_rate: commission rate (default 0.0005 = 5 bps)
    
    Returns:
        dict with evaluation results
    """
    nInst, nDays = price_data.shape
    
    # Initialize tracking variables
    positions = np.zeros(nInst)
    cash = 0
    portfolio_values = []
    daily_pnl = []
    trade_costs = []
    
    print("Starting evaluation...")
    
    # Start evaluation from day 60 to have enough data
    start_day = 60
    
    for day in range(start_day, nDays):
        if day % 50 == 0:
            print(f"Processing day {day}/{nDays}")
        
        # Get price data up to current day
        price_history = price_data[:, :day+1]
        current_prices = price_history[:, -1]
        
        # Get new positions from algorithm
        try:
            new_positions = getMyPosition(price_history)
        except Exception as e:
            print(f"Error getting positions on day {day}: {e}")
            new_positions = positions.copy()
        
        # Apply position limits ($10k per stock)
        for i in range(nInst):
            if current_prices[i] > 0:
                max_shares = int(10000 / current_prices[i])
                new_positions[i] = np.clip(new_positions[i], -max_shares, max_shares)
        
        # Calculate trades
        trades = new_positions - positions
        
        # Calculate trading costs
        trade_volume = np.sum(np.abs(trades * current_prices))
        trade_cost = trade_volume * commission_rate
        trade_costs.append(trade_cost)
        
        # Update cash (subtract cost of trades and commission)
        cash -= np.sum(trades * current_prices) + trade_cost
        
        # Calculate portfolio value
        position_value = np.sum(positions * current_prices)
        portfolio_value = cash + position_value
        portfolio_values.append(portfolio_value)
        
        # Calculate daily P&L
        if day > start_day:
            pnl = portfolio_value - portfolio_values[-2]
            daily_pnl.append(pnl)
        
        # Update positions
        positions = new_positions.copy()
    
    # Calculate final metrics
    daily_pnl = np.array(daily_pnl)
    
    # Competition scoring formula: mean(PL) - 0.1 * StdDev(PL)
    mean_pnl = np.mean(daily_pnl)
    std_pnl = np.std(daily_pnl)
    competition_score = mean_pnl - 0.1 * std_pnl
    
    # Additional metrics
    total_return = (portfolio_values[-1] - portfolio_values[0]) / abs(portfolio_values[0]) if portfolio_values[0] != 0 else 0
    sharpe_ratio = mean_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0
    max_drawdown = calculate_max_drawdown(portfolio_values)
    
    # Calculate win rate
    win_rate = np.sum(daily_pnl > 0) / len(daily_pnl) if len(daily_pnl) > 0 else 0
    
    return {
        'competition_score': competition_score,
        'mean_pnl': mean_pnl,
        'std_pnl': std_pnl,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'portfolio_values': portfolio_values,
        'daily_pnl': daily_pnl,
        'total_trade_costs': np.sum(trade_costs),
        'num_trades': len([x for x in trade_costs if x > 0])
    }

def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown"""
    portfolio_values = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    return np.min(drawdown)

def plot_evaluation_results(results):
    """Plot evaluation results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Portfolio value over time
    axes[0, 0].plot(results['portfolio_values'])
    axes[0, 0].set_title('Portfolio Value Over Time')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].grid(True)
    
    # Daily P&L
    axes[0, 1].plot(results['daily_pnl'])
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('Daily P&L')
    axes[0, 1].set_ylabel('Daily P&L ($)')
    axes[0, 1].grid(True)
    
    # P&L distribution
    axes[0, 2].hist(results['daily_pnl'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 2].set_title('P&L Distribution')
    axes[0, 2].set_xlabel('Daily P&L ($)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].grid(True)
    
    # Cumulative returns
    portfolio_values = np.array(results['portfolio_values'])
    cumulative_returns = (portfolio_values - portfolio_values[0]) / abs(portfolio_values[0])
    axes[1, 0].plot(cumulative_returns * 100)
    axes[1, 0].set_title('Cumulative Returns')
    axes[1, 0].set_ylabel('Cumulative Return (%)')
    axes[1, 0].grid(True)
    
    # Drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak * 100
    axes[1, 1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.7, color='red')
    axes[1, 1].set_title('Drawdown')
    axes[1, 1].set_ylabel('Drawdown (%)')
    axes[1, 1].grid(True)
    
    # Rolling Sharpe ratio
    window = 30
    rolling_sharpe = []
    for i in range(window, len(results['daily_pnl'])):
        window_pnl = results['daily_pnl'][i-window:i]
        if np.std(window_pnl) > 0:
            sharpe = np.mean(window_pnl) / np.std(window_pnl) * np.sqrt(252)
        else:
            sharpe = 0
        rolling_sharpe.append(sharpe)
    
    axes[1, 2].plot(rolling_sharpe)
    axes[1, 2].set_title('Rolling Sharpe Ratio (30-day)')
    axes[1, 2].set_ylabel('Sharpe Ratio')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()

def print_evaluation_summary(results):
    """Print comprehensive evaluation summary"""
    print("\n" + "="*60)
    print("ALGORITHM EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📊 COMPETITION METRICS:")
    print(f"   Competition Score: {results['competition_score']:.6f}")
    print(f"   Mean P&L: ${results['mean_pnl']:.2f}")
    print(f"   Std P&L: ${results['std_pnl']:.2f}")
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"   Total Return: {results['total_return']*100:.2f}%")
    print(f"   Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {results['max_drawdown']*100:.2f}%")
    print(f"   Win Rate: {results['win_rate']*100:.1f}%")
    
    print(f"\n💰 TRADING METRICS:")
    print(f"   Total Trade Costs: ${results['total_trade_costs']:.2f}")
    print(f"   Number of Trading Days: {results['num_trades']}")
    print(f"   Average Trade Cost: ${results['total_trade_costs']/max(results['num_trades'], 1):.2f}")
    
    print(f"\n📋 ADDITIONAL STATS:")
    print(f"   Best Day: ${np.max(results['daily_pnl']):.2f}")
    print(f"   Worst Day: ${np.min(results['daily_pnl']):.2f}")
    print(f"   Positive Days: {np.sum(results['daily_pnl'] > 0)}")
    print(f"   Negative Days: {np.sum(results['daily_pnl'] < 0)}")
    
    # Risk assessment
    print(f"\n⚠️  RISK ASSESSMENT:")
    daily_var_95 = np.percentile(results['daily_pnl'], 5)
    daily_var_99 = np.percentile(results['daily_pnl'], 1)
    print(f"   Daily VaR (95%): ${daily_var_95:.2f}")
    print(f"   Daily VaR (99%): ${daily_var_99:.2f}")
    
    print("="*60)

# Example usage and testing
if __name__ == "__main__":
    # Load your price data
    try:
        price_data = np.load('prices.npy')
        print(f"Loaded price data: {price_data.shape}")
    except FileNotFoundError:
        print("Price data not found. Generating synthetic data for testing...")
        # Generate synthetic data for testing
        np.random.seed(42)
        nInst, nDays = 50, 500
        price_data = np.zeros((nInst, nDays))
        price_data[:, 0] = 100 + np.random.normal(0, 10, nInst)
        
        for day in range(1, nDays):
            returns = np.random.normal(0, 0.02, nInst)
            price_data[:, day] = price_data[:, day-1] * (1 + returns)
    
    # Run evaluation
    print("Running algorithm evaluation...")
    results = evaluate_algorithm(price_data)
    
    # Print results
    print_evaluation_summary(results)
    
    # Plot results
    plot_evaluation_results(results)
    
    # Test individual function
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL FUNCTION CALL")
    print("="*60)
    
    # Test the getMyPosition function with current data
    test_positions = getMyPosition(price_data)
    current_prices = price_data[:, -1]
    
    print(f"Current positions: {len(test_positions)} stocks")
    print(f"Long positions: {np.sum(test_positions > 0)}")
    print(f"Short positions: {np.sum(test_positions < 0)}")
    print(f"Zero positions: {np.sum(test_positions == 0)}")
    
    # Calculate position values
    position_values = test_positions * current_prices
    total_long = np.sum(position_values[position_values > 0])
    total_short = np.abs(np.sum(position_values[position_values < 0]))
    
    print(f"Total long exposure: ${total_long:,.0f}")
    print(f"Total short exposure: ${total_short:,.0f}")
    print(f"Net exposure: ${total_long - total_short:,.0f}")
    
    # Check position limits
    over_limit = []
    for i in range(len(test_positions)):
        if current_prices[i] > 0:
            position_value = abs(test_positions[i] * current_prices[i])
            if position_value > 10000:
                over_limit.append((i, position_value))
    
    if over_limit:
        print(f"⚠️  Warning: {len(over_limit)} positions over $10k limit!")
        for stock, value in over_limit:
            print(f"   Stock {stock}: ${value:,.0f}")
    else:
        print("✅ All positions within $10k limit")
    
    print("\nAlgorithm evaluation complete!")