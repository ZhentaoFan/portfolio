"""
Regression Channel Strategy with Training and Testing Phases
"""
import os
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Ignore unnecessary warnings
warnings.filterwarnings('ignore')

# --------------------------- 1. Data Download and Processing ---------------------------

def download_stock_data(tickers, start_date, end_date):
    """
    Download historical adjusted close price data using yfinance.
    
    Parameters:
    - tickers: list of str, list of stock tickers
    - start_date: str, start date ('YYYY-MM-DD')
    - end_date: str, end date ('YYYY-MM-DD')
    
    Returns:
    - adj_close: pd.DataFrame, adjusted close prices of all tickers
    """
    try:
        print(f"Downloading data for: {', '.join(tickers)} from {start_date} to {end_date}")
        stock_data = yf.download(tickers, start=start_date, end=end_date)
        
        if stock_data.empty:
            print("Downloaded data is empty, please check tickers or date range.")
            return pd.DataFrame()
        
        # Extract 'Adj Close' data
        if 'Adj Close' in stock_data.columns:
            adj_close = stock_data['Adj Close']
        else:
            adj_close = stock_data.copy()
        
        # Remove columns with all NaN values
        adj_close = adj_close.dropna(how='all')
        
        return adj_close
    except Exception as e:
        print(f"Data download failed: {e}")
        return pd.DataFrame()

# --------------------------- 2. Strategy Functions ---------------------------

def apply_stop_loss_take_profit(strategy_returns, stop_loss=-0.05, take_profit=0.10):
    """
    Apply stop-loss and take-profit strategies.
    
    Parameters:
    - strategy_returns: pd.Series, daily returns of the strategy
    - stop_loss: float, stop-loss threshold (e.g., -0.05 for 5% loss)
    - take_profit: float, take-profit threshold (e.g., 0.10 for 10% gain)
    
    Returns:
    - adjusted_returns: pd.Series, returns after applying stop-loss/take-profit
    """
    cumulative_returns = (1 + strategy_returns).cumprod()
    drawdown = cumulative_returns / cumulative_returns.cummax() - 1
    exceed_stop_loss = drawdown < stop_loss
    exceed_take_profit = (cumulative_returns - 1) > take_profit
    
    # Exit position if stop-loss or take-profit is triggered
    positions = (~(exceed_stop_loss | exceed_take_profit)).astype(int)
    
    # Shift positions by 1 to avoid look-ahead bias
    adjusted_returns = strategy_returns * positions.shift(1).fillna(1)
    
    return adjusted_returns

def calculate_sharpe(strategy_returns, risk_free_rate=0.02):
    """
    Calculate the Sharpe ratio.
    
    Parameters:
    - strategy_returns: pd.Series, daily returns of the strategy
    - risk_free_rate: float, annual risk-free rate (default is 0.02)
    
    Returns:
    - sharpe_ratio: float, Sharpe ratio
    """
    excess_returns = strategy_returns - risk_free_rate / 252
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    return sharpe_ratio

def calculate_jensens_alpha(strategy_returns, benchmark_returns, risk_free_rate=0.02):
    """
    Calculate Jensen's Alpha.
    
    Parameters:
    - strategy_returns: pd.Series, daily returns of the strategy
    - benchmark_returns: pd.Series, daily returns of the benchmark
    - risk_free_rate: float, annual risk-free rate (default is 0.02)
    
    Returns:
    - annual_alpha: float, annualized Jensen's Alpha
    """
    data = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if data.empty:
        return np.nan
    
    excess_strategy = data['strategy'] - risk_free_rate / 252
    excess_benchmark = data['benchmark'] - risk_free_rate / 252

    covariance_matrix = np.cov(excess_strategy, excess_benchmark)
    if np.isnan(covariance_matrix).any():
        return np.nan
    
    covariance = covariance_matrix[0, 1]
    variance_benchmark = np.var(excess_benchmark)
    
    if variance_benchmark == 0:
        return np.nan
    
    beta = covariance / variance_benchmark
    alpha = excess_strategy.mean() - beta * excess_benchmark.mean()
    annual_alpha = alpha * 252
    return annual_alpha

# --------------------------- 3. Parameter Optimization (Training Phase) ---------------------------

def optimize_parameters(df_train, window_range, entryZscore_range):
    """
    Optimize parameters window and entryZscore using training data.
    
    Parameters:
    - df_train: pd.DataFrame, training data for stocks
    - window_range: list of int, candidate values for moving average window
    - entryZscore_range: list of float, candidate values for entryZscore
    
    Returns:
    - best_params: dict, contains optimal 'window' and 'entryZscore' values
    """
    best_sharpe = -np.inf
    best_params = {'window': None, 'entryZscore': None}
    
    print("\nStarting parameter optimization...")
    
    for window, entryZscore in product(window_range, entryZscore_range):
        df_temp = df_train.copy()
        df_temp['LOG_ADJ_CLOSE'] = np.log(df_temp['price'])
        df_temp['mean'] = df_temp['LOG_ADJ_CLOSE'].rolling(window=window).mean()
        df_temp['stdev'] = df_temp['LOG_ADJ_CLOSE'].rolling(window=window).std()
        df_temp['zScore'] = (df_temp['LOG_ADJ_CLOSE'] - df_temp['mean']) / df_temp['stdev']
        df_temp['long_entry'] = (df_temp['zScore'] < -entryZscore)
        df_temp['long_exit'] = (df_temp['zScore'] > 0)
        df_temp['num_units_long'] = np.nan
        df_temp.loc[df_temp['long_entry'], 'num_units_long'] = 1
        df_temp.loc[df_temp['long_exit'], 'num_units_long'] = 0
        df_temp['num_units_long'] = df_temp['num_units_long'].fillna(method='pad').fillna(0)
        df_temp['stance'] = df_temp['num_units_long']
        df_temp['market_rets'] = df_temp['price'].pct_change().fillna(0)
        df_temp['port_rets'] = df_temp['market_rets'] * df_temp['stance'].shift(1).fillna(0)
        df_temp['port_rets_adj'] = apply_stop_loss_take_profit(df_temp['port_rets'], stop_loss=-1, take_profit=1)
        sharpe = calculate_sharpe(df_temp['port_rets_adj'])
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params['window'] = window
            best_params['entryZscore'] = entryZscore
    
    print(f"Best parameters: window={best_params['window']}, entryZscore={best_params['entryZscore']}, Sharpe Ratio={best_sharpe:.2f}")
    return best_params

# --------------------------- 4. Main Program ---------------------------

def main():
    selected_tickers = ['WMT', 'NKE', 'AAPL']
    benchmark_ticker = 'SPY'
    start_date = '2015-01-01'
    end_date = '2020-12-31'
    window_range = [20, 25, 30, 35, 40, 60]
    entryZscore_range = [0.0, 0.5, 1.0, 1.5, 2.0]
    train_end_date = '2018-01-01'
    test_start_date = '2018-01-01'
    test_end_date = end_date
    all_tickers = selected_tickers + [benchmark_ticker]
    adj_close = download_stock_data(all_tickers, start_date, end_date)
    
    if adj_close.empty:
        print("Stock data download is empty, please check network or tickers.")
        return
    
    print("\nDownloaded data structure:")
    print(adj_close.head())
    print("\nTickers:", adj_close.columns.tolist())
    
    missing_tickers = [ticker for ticker in all_tickers if ticker not in adj_close.columns]
    if missing_tickers:
        print(f"\nError: Missing data for tickers: {missing_tickers}")
        return
    else:
        print("\nSuccessfully downloaded data for all tickers.")
    
    benchmark_data = adj_close[benchmark_ticker].loc[test_start_date:test_end_date].dropna()
    
    if not os.path.exists('Results'):
        os.makedirs('Results')

    portfolio_returns = pd.DataFrame()


    for stock in selected_tickers:
        print(f"\nProcessing stock: {stock}")
        df = pd.DataFrame()
        df['price'] = adj_close[stock].dropna()
        df = df.sort_index()
        df_train = df.loc[:train_end_date].copy()
        df_test = df.loc[test_start_date:test_end_date].copy()


        print(f"Training period: {df_train.index.min().date()} to {df_train.index.max().date()}")
        print(f"Testing period: {df_test.index.min().date()} to {df_test.index.max().date()}")
        
        best_params = optimize_parameters(df_train, window_range, entryZscore_range)
        best_window = best_params['window']
        best_entryZscore = best_params['entryZscore']
        
        df_test['LOG_ADJ_CLOSE'] = np.log(df_test['price'])
        df_test['mean'] = df_test['LOG_ADJ_CLOSE'].rolling(window=best_window).mean()
        df_test['stdev'] = df_test['LOG_ADJ_CLOSE'].rolling(window=best_window).std()
        df_test['zScore'] = (df_test['LOG_ADJ_CLOSE'] - df_test['mean']) / df_test['stdev']
        df_test['LB'] = df_test['mean'] - best_entryZscore * df_test['stdev']
        df_test['UB'] = df_test['mean'] + best_entryZscore * df_test['stdev']
        df_test['UpperEntryZscore'] = best_entryZscore
        df_test['LowerEntryZscore'] = -best_entryZscore
        df_test['long_entry'] = (df_test['zScore'] < -best_entryZscore)
        df_test['long_exit'] = (df_test['zScore'] > 0)
        df_test['num_units_long'] = np.nan
        df_test.loc[df_test['long_entry'], 'num_units_long'] = 1
        df_test.loc[df_test['long_exit'], 'num_units_long'] = 0
        df_test['num_units_long'] = df_test['num_units_long'].fillna(method='pad').fillna(0)
        df_test['stance'] = df_test['num_units_long']
        df_test['market_rets'] = df_test['price'].pct_change().fillna(0)
        df_test['port_rets'] = df_test['market_rets'] * df_test['stance'].shift(1).fillna(0)
        
        non_zero_rets = df_test['port_rets'].abs().sum()
        print(f"Total strategy returns (port_rets): {non_zero_rets}")
        
        df_test['port_rets_adj'] = apply_stop_loss_take_profit(df_test['port_rets'], stop_loss=-1, take_profit=1)
        df_test['I_adj'] = (1 + df_test['port_rets_adj']).cumprod()
        df_test['Market_Returns_Cummul'] = (1 + df_test['market_rets']).cumprod()
        df_test['I_adj'].iloc[0] = 1
        
        df_test.iloc[30:90].plot(y=["mean", "LB", "UB", "LOG_ADJ_CLOSE"], title=f"{stock} Regression Channel (Test Phase)")
        plt.show()
        
        df_test.iloc[30:90].plot(y=["zScore", "UpperEntryZscore", "LowerEntryZscore"], title=f"{stock} Z-Score (Test Phase)")
        plt.show()
        
        df_test[['Market_Returns_Cummul', 'I_adj']].plot(grid=True)
        plt.ylabel("Cumulative Returns")
        plt.title(f"{stock} Regression Channel - Cumulative Returns with Stop Loss/Take Profit")
        plt.show()

        start_val = df_test['I_adj'].iloc[0]
        end_val = df_test['I_adj'].iloc[-1]
        start_date_dt = df_test.index[0]
        end_date_dt = df_test.index[-1]
        days = (end_date_dt - start_date_dt).days
        periods = 360
        trading_periods = 252
        TotaAnnReturn = (end_val - start_val) / start_val / (days / periods)
        TotaAnnReturn_trading = (end_val - start_val) / start_val / (days / trading_periods)
        years = days / periods
        CAGR = ((end_val / start_val) ** (1 / years)) - 1
        
        try:
            sharpe = calculate_sharpe(df_test['port_rets_adj'])
        except ZeroDivisionError:
            sharpe = 0.0
        
        print(f"\n{stock} Strategy Performance (Test Phase):")
        print(f"Total Annual Return in percent = {TotaAnnReturn * 100:.2f}%")
        print(f"CAGR in percent = {CAGR * 100:.2f}%")
        print(f"Sharpe Ratio = {sharpe:.2f}")
        
        plt.plot(df_test.index, df_test['stance'], label='Stance')
        plt.title(f"{stock} Position Status (Test Phase)")
        plt.xlabel("Date")
        plt.ylabel("Position")
        plt.legend()
        plt.grid(True)
        plt.show()

        long_positions = (df_test['num_units_long'] == 1).sum()
        print(f"{stock} Number of Long Position Days: {long_positions}")
        
        result_filename = f'Results/dfP_Simple_{stock}.csv'
        df_test.to_csv(result_filename, header=True, index=True, encoding='utf-8')
        print(f"Results for {stock} saved to {result_filename}")
        portfolio_returns[stock] = df_test['port_rets_adj']

    
    portfolio_returns['Total_Portfolio'] = portfolio_returns.mean(axis=1)
    # Calculate cumulative returns of the total portfolio
    portfolio_returns['Cumulative_Portfolio'] = (1 + portfolio_returns['Total_Portfolio']).cumprod()

    # Calculate performance metrics for the portfolio
    start_val = 1
    end_val = portfolio_returns['Cumulative_Portfolio'].iloc[-1]
    days = (df_test.index[-1] - df_test.index[0]).days
    years = days / 360

    CAGR = ((end_val / start_val) ** (1 / years)) - 1
    sharpe_ratio = calculate_sharpe(portfolio_returns['Total_Portfolio'])

    print("\nPortfolio Performance:")
    print(f"CAGR: {CAGR * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    portfolio_returns['Cumulative_Portfolio'].plot(label='Portfolio Cumulative Returns')
    
    benchmark_returns = benchmark_data.pct_change().fillna(0)
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    plt.plot(benchmark_cumulative, label=f'{benchmark_ticker} Cumulative Returns (Benchmark)', color='y')
    plt.ylabel("Cumulative Returns")
    plt.title("Portfolio Regression Channel Strategy - Cumulative Returns")
    plt.legend()
    plt.grid(True)
    plt.show()

        
if __name__ == "__main__":
    main()
