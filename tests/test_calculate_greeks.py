'''
Test suite for calculate_greeks function.
Verified correct values by hand, shown at tests/images/greeks_tests
Verified cdf and pdf values using scipy.stats.norm.cdf and scipy.stats.norm.pdf.
'''

import pytest
from options_analyzer import calculate_greeks


#Test: default call greeks
def test_greeks_call():
    S = 25
    K = 24
    r = .06
    t = .22
    v = .38
    greeks = calculate_greeks(S, K, r, t, v)

    verified_delta = .65254875
    verified_gamma = .08290324
    verified_theta = -.01252591
    verified_vega = 4.33169445
    assert pytest.approx(greeks["delta"]) == verified_delta
    assert pytest.approx(greeks["gamma"]) == verified_gamma
    assert pytest.approx(greeks["theta"]) == verified_theta
    assert pytest.approx(greeks["vega"]) == verified_vega


#Test: default put greeks
def test_greeks_put():
    S = 25
    K = 24
    r = .06
    t = .22
    v = .38
    option_type = "put"
    greeks = calculate_greeks(S, K, r, t, v, option_type=option_type)

    verified_delta = -.34745125
    verified_gamma = .08290324
    verified_theta = -.00863245
    verified_vega = 4.33169445
    assert pytest.approx(greeks["delta"]) == verified_delta
    assert pytest.approx(greeks["gamma"]) == verified_gamma
    assert pytest.approx(greeks["theta"]) == verified_theta
    assert pytest.approx(greeks["vega"]) == verified_vega


#Test: call is ATM
def test_greeks_call_atm():
    S = 15
    K = 15
    r = .02
    t = .5
    v = .35
    greeks = calculate_greeks(S, K, r, t, v)

    verified_delta = .56519339
    verified_gamma = .106026566
    verified_theta = -.004383076
    verified_vega = 4.174796026
    assert pytest.approx(greeks["delta"]) == verified_delta
    assert pytest.approx(greeks["gamma"]) == verified_gamma
    assert pytest.approx(greeks["theta"]) == verified_theta
    assert pytest.approx(greeks["vega"]) == verified_vega

#Test: put is way ITM
def test_greeks_put_itm():
    S = 10
    K = 20
    r = .05
    t = .3
    v = .1
    option_type = "put"
    greeks = calculate_greeks(S, K, r, t, v, option_type=option_type)

    verified_delta = -1
    verified_gamma = 0
    verified_theta = .0026989368
    verified_vega = 0
    assert pytest.approx(greeks["delta"]) == verified_delta
    assert pytest.approx(greeks["gamma"]) == verified_gamma
    assert pytest.approx(greeks["theta"]) == verified_theta
    assert pytest.approx(greeks["vega"]) == verified_vega

#Test: interest rate = 0
def test_greeks_rate_zero():
    S = 30
    K = 32
    r = 0
    t = .2
    v = .25
    option_type = "call"
    greeks = calculate_greeks(S, K, r, t, v, option_type=option_type)

    verified_delta = .301062051
    verified_gamma = .103827474
    verified_theta = -0.0080004046
    verified_vega = 4.672236311
    assert pytest.approx(greeks["delta"]) == verified_delta
    assert pytest.approx(greeks["gamma"]) == verified_gamma
    assert pytest.approx(greeks["theta"]) == verified_theta
    assert pytest.approx(greeks["vega"]) == verified_vega

#Test: call has low time value
def test_greeks_low_time():
    S = 30
    K = 32
    r = .05
    t = .001
    v = .25
    option_type = "call"
    greeks = calculate_greeks(S, K, r, t, v, option_type=option_type)

    verified_delta = 1.77e-16
    verified_gamma = 6.177e-15
    verified_theta = -4.767e-16
    verified_vega = 1.3898e-15
    assert pytest.approx(greeks["delta"]) == verified_delta
    assert pytest.approx(greeks["gamma"]) == verified_gamma
    assert pytest.approx(greeks["theta"]) == verified_theta
    assert pytest.approx(greeks["vega"]) == verified_vega

#Test: call has high volatility
def test_greeks_high_vol():
    S = 30
    K = 32
    r = .05
    t = .1
    v = 3
    option_type = "call"
    greeks = calculate_greeks(S, K, r, t, v, option_type=option_type)

    verified_delta = .659677283
    verified_gamma = .012879019
    verified_theta = -0.14419349
    verified_vega = 3.477335048
    assert pytest.approx(greeks["delta"]) == verified_delta
    assert pytest.approx(greeks["gamma"]) == verified_gamma
    assert pytest.approx(greeks["theta"]) == verified_theta
    assert pytest.approx(greeks["vega"]) == verified_vega