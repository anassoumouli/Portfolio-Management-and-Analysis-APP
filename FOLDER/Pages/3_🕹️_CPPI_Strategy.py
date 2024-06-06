import streamlit as st 





import numpy as np 
import pandas as pd 
import yfinance as yf 
import matplotlib.pyplot as plt 
import plotly.express as px
from riskfolio import PlotFunctions
import my_functions as mf 
import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)






st.title(' 🕹️  CPPI Strategy Backtest')
risky_assets = st.sidebar.text_input('Risky Assets','AAPL') # AAPL,CTAS, URI, BLDR,PCAR
start = st.sidebar.date_input('Starting date :',pd.to_datetime('2020-01-01'))
end = st.sidebar.date_input('Ending date :')
safe_assets = st.sidebar.text_input('Safe Assets','SCHO')


df_risky = yf.download(risky_assets,start,end)['Adj Close']
risky_r = df_risky.pct_change().dropna()

df_safe = yf.download(safe_assets,start,end)['Adj Close']
safe_r = df_safe.pct_change().dropna()


# Sidebar inputs for CPPI parameters
st.sidebar.header('CPPI Parameters')
m = st.sidebar.slider('Multiplier (m)', min_value=1.0, max_value=10.0, value=3.0, step=0.1)
start_value = st.sidebar.number_input('Starting Value', value=1000)
floor = st.sidebar.slider('Floor Value (%)', min_value=0.0, max_value=1.0, value=0.8, step=0.05)
riskfree_rate = st.sidebar.number_input('Risk-free Rate (%)', value=0.04, step=0.01)
drawdown = st.sidebar.slider('Max Drawdown (%)', min_value=0.0, max_value=1.0, value=0.2, step=0.05)







if st.button('Run CPPI'):
    #portfolio optimization 
    weights,data = mf.portfolio_max_sharpe(risky_r) 
    risky_r = mf.portfolio_return(weights,risky_r)
    
    btr2 = mf.run_cppi(risky_r, safe_r, m=m, start=start_value, floor=floor, riskfree_rate=riskfree_rate, drawdown=drawdown)

    plt.figure(figsize=(10, 4))
    plt.plot(btr2["Wealth"], label='CPPI Drawdown Constraints')
    plt.plot(btr2["Risky Wealth"], label='Risky Portfolio')
    plt.axhline(y=floor*start_value, color='r', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Valeur')
    plt.legend()
    #st.write(plt.show())
    st.pyplot()
    
    
    a=mf.summary_stats(btr2["Risky Wealth"].pct_change().dropna())
    b=mf.summary_stats(btr2["Wealth"].pct_change().dropna())
    s=pd.concat([a,b])
    s.index=['Risky Portfolio','CPPI Drawdown Constraints']
    st.header('Performance Metrics')
    st.write(s)
    
    #streamlit run 1_🏛️_Home_Page.py 