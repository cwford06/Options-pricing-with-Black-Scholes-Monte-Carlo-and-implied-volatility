import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2 / 2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call
import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2 / 2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call
def monte_carlo_call(S, K, T, r, sigma, simulations=10000):
    Z = np.random.standard_normal(simulations)
    ST = S * np.exp((r - 0.5 * sigma**2)*T + sigma * np.sqrt(T)*Z)
    
    payoff = np.maximum(ST - K, 0)
    price = np.exp(-r*T) * np.mean(payoff)
    
    return price
# =========================
# OPTION PRICING PROJECT
# =========================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime

# =========================
# BLACK-SCHOLES FUNCTION
# =========================
def black_scholes_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0
    
    d1 = (np.log(S/K) + (r + sigma**2 / 2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

# =========================
# MONTE CARLO FUNCTION
# =========================
def monte_carlo_call(S, K, T, r, sigma, simulations=10000):
    if T <= 0 or sigma <= 0:
        return 0
    
    Z = np.random.standard_normal(simulations)
    ST = S * np.exp((r - 0.5 * sigma**2)*T + sigma * np.sqrt(T)*Z)
    
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r*T) * np.mean(payoff)

# =========================
# IMPLIED VOLATILITY
# =========================
def implied_volatility_call(market_price, S, K, T, r):
    try:
        func = lambda sigma: black_scholes_call(S, K, T, r, sigma) - market_price
        return brentq(func, 1e-6, 5)
    except:
        return np.nan

# =========================
# GET DATA
# =========================
ticker = yf.Ticker("SPY")

# Current stock price
stock_price = ticker.history(period="1d")['Close'][0]
print("Stock Price:", stock_price)

# Expiration dates
expirations = ticker.options
print("Expirations:", expirations)

# Pick first expiration
expiration = expirations[0]
print("Using expiration:", expiration)

# Option chain
options = ticker.option_chain(expiration)
calls = options.calls

# Clean data
calls = calls[['strike', 'lastPrice', 'impliedVolatility']].dropna()

# Time to expiration
today = datetime.today()
expiry_date = datetime.strptime(expiration, "%Y-%m-%d")
T = (expiry_date - today).days / 365

# Risk-free rate
r = 0.04

# =========================
# CALCULATIONS
# =========================
results = []

for _, row in calls.iterrows():
    K = row['strike']
    market_price = row['lastPrice']
    sigma = row['impliedVolatility']
    
    bs_price = black_scholes_call(stock_price, K, T, r, sigma)
    mc_price = monte_carlo_call(stock_price, K, T, r, sigma)
    iv = implied_volatility_call(market_price, stock_price, K, T, r)
    
    results.append([K, market_price, bs_price, mc_price, iv])

df = pd.DataFrame(results, columns=[
    "Strike", "Market Price", "BS Price", "MC Price", "Implied Vol"
])

# =========================
# ERROR ANALYSIS
# =========================
df["BS Error"] = df["BS Price"] - df["Market Price"]
df["MC Error"] = df["MC Price"] - df["Market Price"]

print(df.head())

# =========================
# PLOTS
# =========================

# Volatility Smile
plt.figure()
plt.scatter(df["Strike"], df["Implied Vol"])
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.title("Volatility Smile")
plt.show()

# Model Comparison
plt.figure()
plt.plot(df["Strike"], df["Market Price"], label="Market")
plt.plot(df["Strike"], df["BS Price"], label="Black-Scholes")
plt.plot(df["Strike"], df["MC Price"], label="Monte Carlo")
plt.legend()
plt.title("Model Price Comparison")
plt.show()

# Error Plot
plt.figure()
plt.plot(df["Strike"], df["BS Error"], label="BS Error")
plt.plot(df["Strike"], df["MC Error"], label="MC Error")
plt.legend()
plt.title("Pricing Errors")
plt.show()