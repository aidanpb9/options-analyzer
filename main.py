"""
A command-line tool for analyzing European stock options using the
Black-Scholes model. It fetches market data with yfinance, estimates
historical volatility, and computes option prices, Greeks, and P&L
visualizations for single options or option chains.

Created by Aidan Brinkley, 2025
"""
from typing import Literal
import numpy as np
import pandas as pd

from src.constants import NUM_STRIKES
from src.data import get_closing_prices, get_current_price
from src.plots import  plot_pnl_single, plot_pnl_chain, plot_greeks_chain
from src.pricer import calculate_price, calculate_greeks, calculate_historical_volatility
from src.utils import get_option_type, get_output_path, get_rate, get_selection, get_strike_price, get_ticker, get_time


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
    Outputs option price and greeks using historical volatility.

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
    option_price = calculate_price(
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
    strikes = np.linspace(current_price * .9, current_price * 1.1, NUM_STRIKES)

    data = []
    for strike in strikes:
        price = calculate_price(S=current_price, K=strike, r=rate, t=time, v=vol, option_type=option_type)
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
        csv_path=file_path,
        option_type=option_type)
    plot_greeks_chain(
        symbol=symbol,
        time=time,
        csv_path=file_path,
        option_type=option_type)


def main():
    """CLI entry point."""

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