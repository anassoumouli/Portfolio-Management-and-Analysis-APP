import streamlit as st 

import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

#Grab Data
import yfinance as yf

#Usual Suspects
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import my_functions as mf 

# Use PyPortfolioOpt for Calculations
from pypfopt import EfficientFrontier, objective_functions
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel #, plotting
from pypfopt import DiscreteAllocation




st.title('🎩 Black-Litterman Asset Allocation Model ')
assets = st.sidebar.text_input('Assets','AAPL') # AAPL,CTAS, URI, BLDR,PCAR
start = st.sidebar.date_input('Starting date :',pd.to_datetime('2020-01-01'))
end = st.sidebar.date_input('Ending date :')
benchmark_ticker = st.sidebar.text_input('Benchmark','SPY')

df = yf.download(assets,start,end)['Adj Close']
#df = pd.DataFrame()
#for i in assets:
#    df[i]=yf.download(i,start,end)['Adj Close']
returns = df.pct_change().dropna()

#Constructing Benchmark
df_benchmark = yf.download(benchmark_ticker,start,end)['Adj Close']
benchmark_rets = df_benchmark.pct_change().dropna()

#Grap Market Capitalization for each stock in portfolio
mcaps = {}
for t in df.columns:
    stock = yf.Ticker(t)
    mcaps[t] = stock.info["marketCap"]
    
#Getting Priors
if df.shape[1] > 1 :
    S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    delta = black_litterman.market_implied_risk_aversion(df_benchmark)
    market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)

    


Data,Getting_Priors, Integrating_Views,Posterior_Returns,Portfolio_Allocation = st.tabs(['Market Data','Getting Priors','Integrating Views','Posterior Estimate Returns','Portfolio Allocation'])

with Data :
    st.header('Historical Data')
    st.write(df)
    st.header('Market Capitalizations')
    st.write(pd.Series(mcaps))
    st.header('Prices mouvements')
    st.line_chart(df)
    st.header('Performance metrics')
    st.write(mf.summary_stats(returns))
    st.header('Correlation Matrix')
    #if st.button('Calculer'):
    st.write(mf.correlation_matrix(returns))
        
        
        
with Getting_Priors :
    st.write('Delta',delta)
    st.write('Sigma',S)
    st.header('Market implied prior returns')
    st.write(market_prior)
    st.bar_chart(market_prior)
   

with Integrating_Views :
    
    
    st.title("Integrating Views")

    # Step 1: Get the number of key-value pairs
    num_pairs = st.number_input("Enter the number of Views:", min_value=1, step=1)

    # Step 2: Generate input fields for each key-value pair
    viewdict = {}
    if num_pairs:
        keys = []
        values = []
        
        st.subheader("Enter Asset and View")
        for i in range(num_pairs):
            key = st.text_input(f"Asset {i+1}")
            value = st.text_input(f"View {i+1}")
            keys.append(key)
            values.append(value)
        
        # Step 3: Construct the dictionary
        if all(keys) and all(values):
            viewdict = {keys[i]: values[i] for i in range(num_pairs)}
        else:
            st.warning("Please fill in all key-value pairs.")  
            
            
        # Step 2: Generate input fields for each interval
    intervals = []

    if num_pairs:
        st.subheader("Enter the interval Confidence:")
        for i in range(num_pairs):
            col1, col2 = st.columns(2)
            with col1:
                start = st.number_input(f"Start of interval {i+1}", key=f"start_{i}")
            with col2:
                end = st.number_input(f"End of interval {i+1}", key=f"end_{i}")
            intervals.append((start, end))
         
    

    
    
with Posterior_Returns :
    #intervals = [(0, 0.75),(0, 0.75)]
    
    variances = []
    for lb, ub in intervals:
        sigma = (ub - lb)/2
        variances.append(sigma ** 2)

    omega = np.diag(variances)
    
    bl = BlackLittermanModel(S, pi="market", market_caps=mcaps, risk_aversion=delta,absolute_views=viewdict, omega=omega)
    ret_bl = bl.bl_returns()
    st.header('Posterior estimate of returns')
    st.write(ret_bl)
    st.bar_chart(ret_bl)
    
    
with    Portfolio_Allocation :
    S_bl = bl.bl_cov()
    ef = EfficientFrontier(ret_bl, S_bl)
    ef.add_objective(objective_functions.L2_reg)
    ef.max_sharpe()
    weights = ef.clean_weights()
    weights=pd.Series(weights)
    
    st.header('Portfolio Allocation')
    st.write(weights.T) 
    fig, ax = plt.subplots ()
    ax.pie(weights, labels=returns.columns, autopct='%1.1f%%', textprops={'color': 'black'})
    st.pyplot(fig)
   