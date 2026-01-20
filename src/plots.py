"""Plotting functions for option P&L and Greeks."""
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.utils import get_output_path


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
    csv_path: str,
    option_type: Literal["call", "put"] = "call",
) -> None:
    """
    Plots P&L curves for each strike price in option chain. Saves to files.

    Args:
        symbol (str): The name of the stock.
        current_price (float): The stock's current price.
        time (float): The time to maturity in years (decimal).
        csv_path (str): The csv file path containing strikes and prices.
        option_type (string literal): Specifies call or put, defaults to call.
    """
    price_range = np.linspace(current_price * .8, current_price * 1.2, 200)

    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor="#1b1b1b")
    ax.set_title(f"{symbol} {option_type.capitalize()} Chain P&L", fontsize=14, fontweight='bold', color="white")
    ax.set_xlabel("Stock Price($) at Expiration ", fontsize=12, color="white")
    ax.set_ylabel("Profit / Loss ($)", fontsize=12, color="white")

    ax.axvline(current_price, color="#6e5007", linewidth=1, linestyle='--', label='Current Price')

    df = pd.read_csv(csv_path)
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
    csv_path: str,
    option_type: Literal["call", "put"] = "call"
) -> None:
    """
    Has 4 subplots for each greek vs strikes in option chain. Saves to files.

    Args:
        symbol (str): The name of the stock.
        time (float): The time to maturity in years (decimal).
        csv_path (str): The csv file path containing strikes and prices.
        option_type (string literal): Specifies call or put, defaults to call.
    """
    df = pd.read_csv(csv_path)
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