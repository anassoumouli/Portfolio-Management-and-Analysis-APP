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













#st.title('Portfolio optimisation web App ')
assets = st.sidebar.text_input('Assets','AAPL') # AAPL,CTAS, URI, BLDR,PCAR
start = st.sidebar.date_input('Starting date :',pd.to_datetime('2020-01-01'))
end = st.sidebar.date_input('Ending date :')


df = yf.download(assets,start,end)['Adj Close']
returns = df.pct_change().dropna()


choix = st.radio('Optimization Methode',['Portfolio max Sharpe','Portfolio min variance','portfolio max Sortino'])
if choix == 'Portfolio min variance':
    weights,data = mf.portfolio_nim_vol(returns) 
if choix == 'Portfolio max Sharpe':
    weights,data = mf.portfolio_max_sharpe(returns) 
if choix == 'portfolio max Sortino':
    weights,data = mf.portfolio_max_sharpe(returns) 


weights_ = pd.DataFrame({'weight':weights},index=returns.columns)






Data,Allocations, frontier,stats,Backtest = st.tabs(['Market Data','Allocations','Portfolio Frontier','Optimal Portfolio Statistics','Backtest'])
with Data :
    st.header('Historical Data')
    st.write(df)
    st.header('Prices mouvements')
    st.line_chart(df)
    st.header('Performance metrics')
    st.write(mf.summary_stats(returns))
    st.header('Correlation Matrix')
    #if st.button('Calculer'):
    st.write(mf.correlation_matrix(returns))
        
    
    
    
with Allocations :
    st.header('Optimal Portfolio')
    st.write(weights_.T)
    fig, ax = plt.subplots ()
    ax.pie(weights, labels=returns.columns, autopct='%1.1f%%', textprops={'color': 'black'})
    st.pyplot(fig)
    
with frontier:
    st.header('Portfolio Frontier')
    st.pyplot(mf.plot_efficient_frontier(data))
    
with stats:
    st.header('Optimal Portfolio Statistics')
    st.write(mf.summary_stats(pd.DataFrame({'                                                                                            Statistics':mf.portfolio_return(weights,returns)})).T)
    
with Backtest:
    benchmark = st.text_input('Benchmark','AAPL') 
    start_ = st.date_input('Starting date:',pd.to_datetime('2020-01-01'))
    end_ = st.date_input('Ending date:')
    amount_invested = st.number_input('Amount Invested',1)
    df_ = yf.download(assets,start_,end_)['Adj Close']
    df_benchmark = yf.download(benchmark,start_,end_)['Adj Close']
    
    st.pyplot(mf.backtest(df_,weights,df_benchmark,amount_invested,))
    
    initial_prices = df_.iloc[0]
    backtest = np.sum(df_*(amount_invested/initial_prices) * weights,axis=1) 
    s1=mf.summary_stats(pd.DataFrame({'Benchmark Portfolio':df_benchmark}).pct_change().dropna())
    s2=mf.summary_stats(pd.DataFrame({'Optimal Portfolio':backtest}).pct_change().dropna())
    st.header('Performance metrics')
    st.write(pd.concat([s1,s2]).T)