'''
Test suite for black_scholes_pricer function.
Verified correct values by hand, shown at tests/images/black_scholes_tests
Verified cdf values using scipy.stats.norm.cdf.
'''

import pytest
from options_analyzer import black_scholes_pricer


#Test: default call price1
def test_bspricer_call1():
    verified_price = 2.334017338
    S = 20.00
    K = 18.00
    r = 0.05
    t = 0.25
    v = 0.20
    option_type = "call"
    price = black_scholes_pricer(S, K, r, t, v, option_type=option_type)
    assert pytest.approx(price) == verified_price


#Test: default call price2
def test_bspricer_call2():
    verified_price= 13.59351845
    S = 139.22
    K = 125.88
    r = .0028
    t = .72
    v = .003
    price= black_scholes_pricer(S, K, r, t, v)
    assert pytest.approx(price) == verified_price


#Test: default put price1
def test_bspricer_put1():
    verified_price = .1104177473
    S = 20.00
    K = 18.00
    r = 0.05
    t = 0.25
    v = 0.20
    option_type = "put"
    price = black_scholes_pricer(S, K, r, t, v, option_type=option_type)
    assert pytest.approx(price) == verified_price


#Test: default put price2
def test_bspricer_put2():
    verified_price = 0
    S = 139.22
    K = 125.88
    r = .0028
    t = .72
    v = .003
    option_type = "put"
    price = black_scholes_pricer(S, K, r, t, v, option_type=option_type)
    assert pytest.approx(price) == verified_price


#Test: interest rate = 0
def test_bspricer_rate_zero():
    verified_price = 4.083117826
    S = 46.938
    K = 47
    r = 0
    t = .16
    v = .55
    price = black_scholes_pricer(S, K, r, t, v)
    assert pytest.approx(price) == verified_price


#Test: call is deep ITM
def test_bspricer_itm():
    verified_price = 80.78421144
    S = 100.
    K = 20
    r = .1
    t = .4
    v = .5
    price = black_scholes_pricer(S, K, r, t, v)
    assert pytest.approx(price) == verified_price


#Test: call is deep OTM
def test_bspricer_otm():
    verified_price = 8.92637807e-7
    S = 20
    K = 100
    r = .1
    t = .4
    v = .5
    price = black_scholes_pricer(S, K, r, t, v)
    assert pytest.approx(price) == verified_price


#Test: call is ATM
def test_bspricer_atm():
    verified_price = 3.025434796
    S = 20
    K = 20
    r = .05
    t = .5
    v = .5
    price = black_scholes_pricer(S, K, r, t, v)
    assert pytest.approx(price) == verified_price


#Test: call has low time value
def test_bspricer_low_time():
    verified_price = 6.72769913e-4
    S = 49
    K = 50
    r = .05
    t = .001
    v = .25
    price = black_scholes_pricer(S, K, r, t, v)
    assert pytest.approx(price) == verified_price


#Test: call has high volatility
def test_bspricer_high_vol():
    verified_price = 55.64251569
    S = 80
    K = 77.5
    r = .05
    t = 1
    v = 2
    price = black_scholes_pricer(S, K, r, t, v)
    assert pytest.approx(price) == verified_price