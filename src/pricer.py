"""Black-Scholes pricing, Greeks, and volatility calculations."""
from scipy.stats import norm
import math
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from src.constants import CALENDAR_DAYS_PER_YEAR, TRADING_DAYS_PER_YEAR


def _calculate_d1_d2(S: float, K: float, r: float, t: float, v: float) -> tuple[float, float]:
    """Compute d1 & d2 for black-scholes and greeks."""
    d1 = (math.log(S / K) + (r + (v**2 / 2)) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    return d1, d2


def calculate_price(S: float, K: float, r: float, t: float, v: float, option_type: Literal["call", "put"] = "call") -> float:
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
        theta = theta / CALENDAR_DAYS_PER_YEAR #for daily theta
    else:
        delta = norm.cdf(d1) - 1
        theta = -((S * norm.pdf(d1) * v) / (2 * math.sqrt(t))) + (r * K * math.exp(-r * t) * norm.cdf(-d2))
        theta = theta / CALENDAR_DAYS_PER_YEAR #for daily theta

    gamma = (norm.pdf(d1)) / (S * v * math.sqrt(t))
    vega = S * norm.pdf(d1) * math.sqrt(t) / 100

    greeks = {
        "delta" : delta,
        "gamma" : gamma,
        "theta" : theta,
        "vega"  : vega 
    }
    return greeks


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
    annual_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

    if annual_vol <= 0:
        raise ValueError("Volatility cannot be zero")

    return annual_vol