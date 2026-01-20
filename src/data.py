"""Functions for fetching stock info with yfinance."""
import yfinance as yf
import numpy as np
from numpy.typing import NDArray


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