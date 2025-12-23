'''
Test suite for data functions.
Historical vol tests calculated by this tool, verified by real vol 
at time of test creation, shown at tests/images/historical_vol_tests
'''

import pytest
import numpy as np
from options_analyzer import calculate_historical_volatility as calc_hist_vol


#Test: default case 1 (AAPL 12/20/2025)
def test_calculate_historical_volatility1():
    closing_prices = np.array([266.25, 271.489990234375, 275.9200134277344, 276.9700012207031, 277.54998779296875, 
                       278.8500061035156, 283.1000061035156, 286.19000244140625, 284.1499938964844, 
                       280.70001220703125, 278.7799987792969, 277.8900146484375, 277.17999267578125, 
                       278.7799987792969, 278.0299987792969, 278.2799987792969, 274.1099853515625, 
                       274.6099853515625, 271.8399963378906, 272.19000244140625, 273.6700134277344])
    hist_vol = calc_hist_vol(closing_prices=closing_prices)
    verified_vol = 0.14846690666599718
    assert pytest.approx(hist_vol) == verified_vol
    
    
#Test: default case 2 (TSLA 12/20/2025)
def test_calculate_historical_volatility2():
    closing_prices = np.array([395.2300109863281, 391.0899963378906, 417.7799987792969, 419.3999938964844, 
                               426.5799865722656, 430.1700134277344, 430.1400146484375, 429.239990234375, 
                               446.739990234375, 454.5299987792969, 455.0, 439.5799865722656, 445.1700134277344, 
                               451.45001220703125, 446.8900146484375, 458.9599914550781, 475.30999755859375, 
                               489.8800048828125, 467.260009765625, 483.3699951171875, 481.20001220703125])
    hist_vol = calc_hist_vol(closing_prices=closing_prices)
    verified_vol = 0.41051143338756674
    assert pytest.approx(hist_vol) == verified_vol


#Test: hist_vol calculation gives 0.
def test_calculate_historical_volatility_zero():
    closing_prices = np.array([100, 100, 100, 100, 100, 100, 100, 100])
    with pytest.raises(ValueError, match="Volatility cannot be zero"):
        calc_hist_vol(closing_prices=closing_prices)