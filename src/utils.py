"""User input and file functions for the CLI."""
import yfinance as yf
import os
from datetime import datetime
from src.constants import CALENDAR_DAYS_PER_YEAR, INTEREST_RATE, OUTPUT_DIR


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
                return  (days / CALENDAR_DAYS_PER_YEAR)
            print("\nDays must be positive")
        except ValueError:
            print("\nInvalid expiration days")


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


def get_selection(prompt: str, options: tuple[int, ...]) -> int:
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
    days = f"{int(round(time * CALENDAR_DAYS_PER_YEAR))}d"

    output_subdir = os.path.join(OUTPUT_DIR, f"{symbol}_{date}_{days}")
    os.makedirs(output_subdir, exist_ok=True)

    file_path = os.path.join(output_subdir, file_name)
    return file_path