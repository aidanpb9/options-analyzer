'''
Test suite for data functions:
get_current_price(), get_closing_prices()
'''

import pytest
from unittest.mock import Mock
import pandas as pd
import numpy as np
from options_analyzer import get_current_price, get_closing_prices

'''get_current_price() tests'''
'''--------------------------------------------------------------------------'''
#Test: default case
def test_get_current_price():
    mock_ticker = Mock()
    mock_ticker.info = {"currentPrice": 86.85}
    current_price = get_current_price(mock_ticker)
    assert current_price == 86.85


#Test: missing info
def test_get_current_price_missing():
    mock_ticker = Mock()
    mock_ticker.info = {}
    with pytest.raises(ValueError, match="No current price available"):
        get_current_price(mock_ticker)
    

#Test: info is None
def test_get_current_price_none():
    mock_ticker = Mock()
    mock_ticker.info = {"currentPrice": None}
    with pytest.raises(ValueError, match="No current price available"):
        get_current_price(mock_ticker)


#Test: info is string (can't convert to float)
def test_get_current_price_string():
    mock_ticker = Mock()
    mock_ticker.info = {"currentPrice": "abc"}
    with pytest.raises(ValueError, match="No current price available"):
        get_current_price(mock_ticker)


#Test: positive price
def test_get_current_price_negative():
    mock_ticker = Mock()
    mock_ticker.info = {"currentPrice": -2.33}
    with pytest.raises(ValueError, match="current_price must be positive"):
        get_current_price(mock_ticker)



'''get_closing_prices() tests'''
'''--------------------------------------------------------------------------'''
#Test: defualt case
def test_get_closing_prices():
    mock_ticker = Mock()
    mock_history = pd.DataFrame({"Close": [1, 2, 3, 4, 5]})
    mock_ticker.history.return_value = mock_history
    closing_prices = get_closing_prices(mock_ticker)
    assert isinstance(closing_prices, np.ndarray)
    assert len(closing_prices) == 5


#Test: missing array
def test_get_closing_prices_missing():
    mock_ticker = Mock()
    mock_ticker.history.return_value = {}
    with pytest.raises(ValueError, match="No closing prices available"):
        get_closing_prices(mock_ticker)


#Test: array has np.nan
def test_get_closing_prices_nans():
    mock_ticker = Mock()
    mock_history = pd.DataFrame({"Close": [1, np.nan, 2]})
    mock_ticker.history.return_value = mock_history
    closing_prices = get_closing_prices(mock_ticker)
    assert len(closing_prices) == 2


#Test: array length < 2
def test_get_closing_prices_short():
    mock_ticker = Mock()
    mock_history = pd.DataFrame({"Close": [1]})
    mock_ticker.history.return_value = mock_history
    with pytest.raises(ValueError, match="Not enough data"):
        get_closing_prices(mock_ticker)