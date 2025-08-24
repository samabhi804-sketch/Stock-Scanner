import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

class StockDataManager:
    """Handles stock data retrieval and caching for both Indian and US markets"""
    
    def __init__(self):
        self.cache_duration = 300  # 5 minutes cache
    
    @st.cache_data(ttl=300)
    def get_stock_data(_self, symbol: str, period: str = '1y') -> pd.DataFrame | None:
        """
        Fetch stock data using yfinance
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'RELIANCE.NS')
            period: Time period ('1mo', '3mo', '6mo', '1y', '2y', '5y')
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                st.error(f"No data found for symbol: {symbol}")
                return None
            
            # Reset index to make Date a column
            data = data.reset_index()
            data.set_index('Date', inplace=True)
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    @st.cache_data(ttl=300)
    def get_stock_info(_self, symbol: str) -> dict:
        """
        Get detailed stock information
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with stock info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
            
        except Exception as e:
            st.error(f"Error fetching info for {symbol}: {str(e)}")
            return {}
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_multiple_stocks_data(_self, symbols: list, period: str = '1y') -> dict:
        """
        Fetch data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            period: Time period
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        stocks_data = {}
        
        for symbol in symbols:
            data = _self.get_stock_data(symbol, period)
            if data is not None and not data.empty:
                stocks_data[symbol] = data
        
        return stocks_data
    
    def validate_symbol(self, symbol: str, market: str) -> str:
        """
        Validate and format stock symbol based on market
        
        Args:
            symbol: Raw stock symbol
            market: 'Indian' or 'US'
            
        Returns:
            Formatted symbol
        """
        symbol = symbol.upper().strip()
        
        if market == 'Indian':
            if not (symbol.endswith('.NS') or symbol.endswith('.BO')):
                # Default to NSE
                symbol = f"{symbol}.NS"
        
        return symbol
    
    def get_market_hours(self, market: str) -> dict:
        """
        Get market trading hours
        
        Args:
            market: 'Indian' or 'US'
            
        Returns:
            Dictionary with market hours info
        """
        if market == 'Indian':
            return {
                'open_time': '09:15',
                'close_time': '15:30',
                'timezone': 'Asia/Kolkata',
                'currency': 'INR'
            }
        else:  # US
            return {
                'open_time': '09:30',
                'close_time': '16:00',
                'timezone': 'America/New_York',
                'currency': 'USD'
            }
