"""
options_analyzer.py

A command-line tool for analyzing European stock options using the
Black-Scholes model. It fetches market data with yfinance, estimates
historical volatility, and computes option prices, Greeks, and P&L
visualizations for single options or option chains.

Created by Aidan Brinkley, 2025
"""

from scipy.stats import norm
import math
from typing import Literal
import yfinance as yf
import numpy as np
import pandas as pd
from numpy.typing import NDArray
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime

#CONSTANTS
DAYS_IN_YEAR = 365
INTEREST_RATE = .045 #hardcoded, common default rate
ANNUAL_PERIODS = 252 #trading days in a year
NUM_OPTIONS = 5      #different strikes in option chain
OUTPUT_DIR = "output"


def get_ticker() -> yf.Ticker:
    """Prompts user for a stock until input is valid. Returns the stock object."""
    while True:
        ticker_name = input("Enter stock symbol: ").strip().upper()
        try:
            ticker = yf.Ticker(ticker_name)
            if ticker.info.get("regularMarketPrice") is not None:
                return ticker
            print("\nNo market data found")
        except Exception:
            print("\nError getting yfinance data")


def get_time() -> float:
    """Prompts user for days to expiration until input is valid. 
    Returns the time in years as a decimal."""
    while True:
        try:
            days = float(input("Enter days to expiration: "))
            if days > 0:
                return  (days / DAYS_IN_YEAR)
            print("\nDays must be positive")
        except ValueError:
            print("\nInvalid expiration time")


def get_strike_price() -> float:
    """Prompts user for a strike price until input is valid and returns it."""
    while True:
        try:
            strike_price = float(input("Enter strike price: "))
            if strike_price > 0:
                return strike_price
            print("\nStrike price must be positive")
        except ValueError:
            print("\nInvalid strike price")


def get_option_type() -> str:
    """Prompts user to select call or put option type and returns it."""
    while True:
        option_type = input("Enter option type (call or put): ").strip().lower()
        if option_type in ("call", "put"):
            return option_type
        else:
            print("\nInvalid option type")


def get_rate() -> float:
    """Returns the annualized 3-month risk-free interest rate, currently hardcoded (decimal)."""
    return INTEREST_RATE


def get_current_price(ticker: yf.Ticker) -> float:
    """
    Extracts the current price from a yfinance API ticker object.

    Args:
        ticker (yf.Ticker): A yfinance stock object.
    
    Returns:
        float: The stock's current price.
    
    Raises:
        ValueError: If current price is invalid.
    """
    info = ticker.info
    try: 
        current_price = float(info.get("currentPrice"))
    except Exception:
        raise ValueError("No current price available")
    
    if current_price  <= 0:
        raise ValueError("current_price must be positive")
    return current_price


def get_closing_prices(ticker: yf.Ticker) -> NDArray[np.float64]:
    """
    Extracts a numpy array of closing prices from a yfinance API ticker object.

    Args:
        ticker (yf.Ticker): A yfinance stock object.
    
    Returns:
        NDArray[np.float64]: The stock's closing prices for the last 30 days.
    
    Raises:
        ValueError: If closing prices aren't available or are insufficient.
    """
    history = ticker.history(period="1mo")
    closings = history.get("Close")
    if closings is None:
        raise ValueError("No closing prices available")

    #convert to numpy and drop nans
    closing_prices = closings.to_numpy(dtype=float)
    closing_prices = closing_prices[~np.isnan(closing_prices)]

    if len(closing_prices) < 2: 
        raise ValueError("Not enough data")
    return closing_prices


def get_selection(prompt: str, options: tuple) -> int:
    """
    Prompts user to select from valid integer options.
    
    Args:
        prompt (str): The message to display.
        options (tuple): The valid integer choices.
    
    Returns:
        int: The selected option.
    """
    while True:
        print(prompt, end="")
        try:
            choice = int(input())
            if choice in options:
                return choice
            print(f"\nSelect from {options}")
        except ValueError:
            print("\nInvalid input")


def get_output_path(
    symbol: str, 
    time: float, 
    file_name: str
) -> str:
    """
    Ensures creation of output file path for saving results.

    Args:
        symbol (str): The name of the stock.
        time (float): The time to maturity in years (decimal).
        file_name (str): The file name
    
    Returns: 
        str: The output file path with file name

    Raises:
        ValueError: If input file_name is empty.
    """
    if not file_name or not file_name.strip():
        raise ValueError("Invalid file name")
    file_name = os.path.basename(file_name)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")
    days = f"{int(round(time * 365))}d"

    output_subdir = os.path.join(OUTPUT_DIR, f"{symbol}_{date}_{days}")
    os.makedirs(output_subdir, exist_ok=True)

    file_path = os.path.join(output_subdir, file_name)
    return file_path
    

def calculate_historical_volatility(closing_prices: NDArray[np.float64]) -> float:
    """
    Calculates the historical volatility of a stock using a 1 month lookback period, annualized.
    Formula: https://www.macroption.com/historical-volatility-calculation/
            
    Args:
        closing_prices (NDArray[np.float64]): The stock's closing prices for the last 30 days.
    
    Returns:
        float: The historical vol annualized (decimal).

    Raises: 
        ValueError: If historical volatility is 0.
    """
    returns = np.log(closing_prices[1:] / closing_prices[:-1])
    daily_vol = np.std(returns, ddof=1) #sample stddev
    annual_vol = daily_vol * np.sqrt(ANNUAL_PERIODS)

    if annual_vol == 0:
        raise ValueError("Volatility cannot be zero")

    return annual_vol


def _calculate_d1_d2(S: float, K: float, r: float, t: float, v: float) -> tuple[float, float]:
    '''Compute d1 & d2 for black-scholes and greeks.'''
    d1 = (math.log(S / K) + (r + (v**2 / 2)) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    return d1, d2


def black_scholes_pricer(S: float, K: float, r: float, t: float, v: float, option_type: Literal["call", "put"] = "call") -> float:
    """
    Calculates the price of an option using the Black-Scholes formula.
    Formula: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model

    Args:
        S (float): The underlying stock price.
        K (float): The strike price.
        r (float): The current annualized risk-free interest rate (decimal).
        t (float): The time to maturity in years (decimal).
        v (float): The volatility of the underlying stock (decimal).
        option_type (string literal): Specifies call or put, defaults to call.
    
    Returns:
        float: The price of the option.
    """
    d1, d2 = _calculate_d1_d2(S, K, r, t, v)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


def calculate_greeks(S: float, K: float, r: float, t: float, v: float, option_type: Literal["call", "put"] = "call") -> dict[str, float]:
    """
    Calculates the option greeks.
    Formula: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model

    Args:
        S (float): The underlying stock price.
        K (float): The strike price.
        r (float): The current annualized risk-free interest rate (decimal).
        t (float): The time to maturity in years (decimal).
        v (float): The volatility of the underlying stock (decimal).
        option_type (string literal): Specifies call or put, defaults to call.
    
    Returns:
        dict: The greeks(delta, theta, gamma, and vega) mapped to their values.
    """
    d1, d2 = _calculate_d1_d2(S, K, r, t, v)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = -((S * norm.pdf(d1) * v) / (2 * math.sqrt(t))) - (r * K * math.exp(-r * t) * norm.cdf(d2))
        theta = theta / 365 #for daily theta
    else:
        delta = norm.cdf(d1) - 1
        theta = -((S * norm.pdf(d1) * v) / (2 * math.sqrt(t))) + (r * K * math.exp(-r * t) * norm.cdf(-d2))
        theta = theta / 365 #for daily theta

    gamma = (norm.pdf(d1)) / (S * v * math.sqrt(t))
    vega = S * norm.pdf(d1) * math.sqrt(t)

    greeks = {
        "delta" : delta,
        "gamma" : gamma,
        "theta" : theta,
        "vega"  : vega 
    }
    return greeks


def analyze_option(
    symbol: str, 
    current_price: float, 
    rate: float, 
    time: float, 
    vol: float,
    option_type: Literal["call", "put"] = "call",
) -> None:
    """
    Analyzes an option at a specific strike price.
    Outputs option price, historical volatility, and greeks.

    Args:
        symbol (str): The name of the stock.
        current_price (float): The stock's current price.
        rate (float): The current annualized risk-free interest rate (decimal).
        time (float): The time to maturity in years (decimal).
        vol (float): The historical volatility.
        option_type (string literal): Specifies call or put, defaults to call.
    
    Returns:
        None
    
    Raises:
        ValueError: If closing prices array is invalid.
    """
    strike_price = get_strike_price()
    option_price = black_scholes_pricer(
        S=current_price, 
        K=strike_price, 
        r=rate, 
        t=time, 
        v=vol, 
        option_type=option_type)
    greeks = calculate_greeks(
        S=current_price, 
        K=strike_price, 
        r=rate, 
        t=time, 
        v=vol, 
        option_type=option_type)
    
    print(f"\nOption Price: ${option_price:.2f}") 
    print(f"Historical Volatility: {vol:.2%}")
    for greek, value in greeks.items():
        print(f"{greek.capitalize()}: {value:.2f}")

    plot_pnl_single(
        symbol=symbol, 
        current_price=current_price,
        option_price=option_price,
        strike=strike_price, 
        time=time,
        option_type=option_type)


def analyze_option_chain(
    symbol: str, 
    current_price: float, 
    rate: float, 
    time: float, 
    vol: float,
    option_type: Literal["call", "put"] = "call",
) -> None:
    """
    Constructs an option chain around the current price.
    Exports the chain as a csv.
    For each strike price, includes its option price and greeks.

    Args:
        symbol (str): The name of the stock.
        current_price (float): The stock's current price.
        rate (float): The current annualized risk-free interest rate (decimal).
        time (float): The time to maturity in years (decimal).
        vol (float): The historical volatility.
        option_type (string literal): Specifies call or put, defaults to call.
    
    Returns:
        None
    """
    strikes = np.linspace(current_price * .9, current_price * 1.1, NUM_OPTIONS)

    data = []
    for strike in strikes:
        price = black_scholes_pricer(S=current_price, K=strike, r=rate, t=time, v=vol, option_type=option_type)
        greeks = calculate_greeks(S=current_price, K=strike, r=rate, t=time, v=vol, option_type=option_type)
        data.append({"Strike": strike,
                     "Price" : price,
                     "Delta" : greeks["delta"],
                     "Gamma" : greeks["gamma"],
                     "Theta" : greeks["theta"],
                     "Vega"  : greeks["vega"]})
    df = pd.DataFrame(data)

    file_name = f"{option_type}_chain.csv"
    file_path = get_output_path(symbol=symbol, time=time, file_name=file_name)
    df.to_csv(file_path, index=False, float_format="%.4f")
    print(f"Option chain saved to {file_path}")

    plot_pnl_chain(
        symbol=symbol,
        current_price=current_price,
        time=time,
        data=file_path,
        option_type=option_type)
    plot_greeks_chain(
        symbol=symbol,
        time=time,
        data=file_path,
        option_type=option_type)


def plot_pnl_single(
    symbol: str, 
    current_price: float, 
    option_price: float, 
    strike: float, 
    time: float,
    option_type: Literal["call", "put"] = "call"
) -> None:
    """
    Plots a P&L curve for a single strike price. Saves to files.

    Args:
        symbol (str): The name of the stock.
        current_price (float): The stock's current price.
        option_price (float): The price of the option.
        strike (float): The strike price.
        time (float): The time to maturity in years (decimal).
        option_type (string literal): Specifies call or put, defaults to call.
    """
    price_range = np.linspace(current_price * .8, current_price * 1.2, 200)
    if option_type == "call":
        pnl = np.maximum(price_range - strike, 0) - option_price
        breakeven = strike + option_price
    else: 
        pnl = np.maximum(strike - price_range, 0) - option_price
        breakeven = strike - option_price

    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor="#1b1b1b")
    ax.set_title(f"{symbol} {option_type.capitalize()} P&L", fontsize=14, fontweight='bold', color="white")
    ax.set_xlabel("Stock Price($) at Expiration ", fontsize=12, color="white")
    ax.set_ylabel("Profit / Loss ($)", fontsize=12, color="white")

    ax.plot(price_range, pnl, linewidth=1.7, color="#0cd5d2ff", label=f"Strike ${strike:.2f}")
    ax.axvline(current_price, color="#6e5007", linewidth=1, linestyle='--', label='Current Price')
    ax.plot(breakeven, 0, marker='|', markersize=24, markeredgewidth=2, color="#b31926", zorder=5)
    ax.plot([], [], color="#b31926", linewidth=2, label='Breakeven')

    ax.grid(axis='y', color="#3b3b3b", linestyle=':')
    ax.axhline(0, color="#424242", linestyle=':')
    ax.tick_params(colors="white")
    ax.set_facecolor("#1b1b1b")

    ax.legend(facecolor="#2b2b2b", edgecolor="white", labelcolor="white")
    plt.tight_layout()

    file_name = f"{option_type}_{strike:.2f}_pnl.png"
    file_path = get_output_path(symbol=symbol, time=time, file_name=file_name)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"P&L plot saved to {file_path}")
    plt.close()


def plot_pnl_chain(
    symbol: str, 
    current_price: float, 
    time: float, 
    data: str,
    option_type: Literal["call", "put"] = "call",
) -> None:
    """
    Plots P&L curves for each strike price in option chain. Saves to files.

    Args:
        symbol (str): The name of the stock.
        current_price (float): The stock's current price.
        time (float): The time to maturity in years (decimal).
        data (str): The csv file path containing strikes and prices.
        option_type (string literal): Specifies call or put, defaults to call.
    """
    price_range = np.linspace(current_price * .8, current_price * 1.2, 200)

    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor="#1b1b1b")
    ax.set_title(f"{symbol} {option_type.capitalize()} Chain P&L", fontsize=14, fontweight='bold', color="white")
    ax.set_xlabel("Stock Price($) at Expiration ", fontsize=12, color="white")
    ax.set_ylabel("Profit / Loss ($)", fontsize=12, color="white")

    ax.axvline(current_price, color="#6e5007", linewidth=1, linestyle='--', label='Current Price')

    df = pd.read_csv(data)
    colors = cm.viridis(np.linspace(0, 1, len(df["Strike"])))
    for strike, price, color in zip(df["Strike"].values, df["Price"].values, colors):
        if option_type == "call":
            pnl = np.maximum(price_range - strike, 0) - price
        else: 
            pnl = np.maximum(strike - price_range, 0) - price
        ax.plot(price_range, pnl, linewidth=1.5, alpha=.8, color=color, label=f"Strike ${strike:.2f}")
        
    ax.grid(axis='y', color="#3b3b3b", linestyle=':')
    ax.axhline(0, color="#424242", linestyle=':')
    ax.tick_params(colors="white")
    ax.set_facecolor("#1b1b1b")

    ax.legend(facecolor="#2b2b2b", edgecolor="white", labelcolor="white")
    plt.tight_layout()

    file_name = f"{option_type}_chain_pnl.png"
    file_path = get_output_path(symbol=symbol, time=time, file_name=file_name)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"P&L chain plot saved to {file_path}")
    plt.close()


def plot_greeks_chain(
    symbol: str,
    time: float,
    data: str,
    option_type: Literal["call", "put"] = "call"
) -> None:
    """
    Has 4 subplots for each greek vs strikes in option chain. Saves to files.

    Args:
        symbol (str): The name of the stock.
        time (float): The time to maturity in years (decimal).
        data (str): The csv file path containing strikes and prices.
        option_type (string literal): Specifies call or put, defaults to call.
    """
    df = pd.read_csv(data)
    strikes = df["Strike"]
    delta = df["Delta"]
    gamma = df["Gamma"]
    theta = df["Theta"]
    vega = df["Vega"]

    fig, ax = plt.subplots(2, 2, figsize=(13,6.5), facecolor="#1b1b1b")
    fig.suptitle(f"{symbol} Greeks {option_type.capitalize()} Chain", fontsize=14, fontweight="bold", color="white")
    colors = ["#ffd700", "#1e90ff", "#ff4500", "#32cd32"]

    ax[0, 0].plot(strikes, delta, linewidth=1.5, color=colors[0], label="Delta")
    ax[0, 0].set_ylabel("Delta", fontsize=12, color="white")

    ax[0, 1].plot(strikes, gamma, linewidth=1.5, color=colors[1], label="Gamma")
    ax[0, 1].set_ylabel("Gamma", fontsize=12, color="white")

    ax[1, 0].plot(strikes, theta, linewidth=1.5, color=colors[2], label="Theta")
    ax[1, 0].set_ylabel("Theta", fontsize=12, color="white")

    ax[1, 1].plot(strikes, vega, linewidth=1.5, color=colors[3], label="Vega")
    ax[1, 1].set_ylabel("Vega", fontsize=12, color="white")

    for ax in ax.flat:
        ax.set_xlabel("Strike($)", fontsize=12, color="white")
        ax.grid(color="#3b3b3b", linestyle=':')
        ax.tick_params(colors="white")
        ax.set_facecolor("#1b1b1b")
        ax.legend(facecolor="#2b2b2b", edgecolor="white", labelcolor="white")
    plt.tight_layout()

    file_name = f"{option_type}_chain_greeks.png"
    file_path = get_output_path(symbol=symbol, time=time, file_name=file_name)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Greeks chain plot saved to {file_path}")
    plt.close()


def main():
    stock = get_ticker()
    symbol = stock.ticker

    try:
        closing_prices = get_closing_prices(stock)
    except ValueError as e:
        print(f"\nError: {e}")
        return
    hist_vol = calculate_historical_volatility(closing_prices=closing_prices)

    try:
        current_price = get_current_price(stock) 
    except ValueError as e:
        print(f"\nError: {e}")
        return

    option_type = get_option_type() 
    time = get_time()
    rate = get_rate()

    print(f"{symbol} current price: {current_price:.2f}.")
    prompt = "(1) single option or (2) option chain? "
    options = (1, 2)
    choice = get_selection(prompt=prompt, options=options)
    if choice == 1:
        analyze_option(
            symbol=symbol, 
            current_price=current_price, 
            rate=rate, 
            time=time, 
            vol=hist_vol,
            option_type=option_type)
    else:
        analyze_option_chain(
            symbol=symbol, 
            current_price=current_price, 
            rate=rate, 
            time=time, 
            vol=hist_vol,
            option_type=option_type)


if __name__ == "__main__":
    main()