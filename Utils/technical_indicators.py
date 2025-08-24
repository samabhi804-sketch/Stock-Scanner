import pandas as pd
import numpy as np
import streamlit as st

class TechnicalIndicators:
    """Calculate various technical indicators for stock analysis"""
    
    def __init__(self):
        pass
    
    def calculate_sma(self, data: pd.DataFrame, window: int):
        """Calculate Simple Moving Average"""
        return data['Close'].rolling(window=window).mean()
    
    def calculate_ema(self, data: pd.DataFrame, window: int):
        """Calculate Exponential Moving Average"""
        return data['Close'].ewm(span=window).mean()
    
    def calculate_rsi(self, data: pd.DataFrame, window: int = 14):
        """Calculate Relative Strength Index"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, window: int = 20, num_std: float = 2) -> tuple:
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(data, window)
        std = data['Close'].rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    def calculate_stochastic(self, data: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> tuple:
        """Calculate Stochastic Oscillator"""
        low_min = data['Low'].rolling(window=k_window).min()
        high_max = data['High'].rolling(window=k_window).max()
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    def calculate_williams_r(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_max = data['High'].rolling(window=window).max()
        low_min = data['Low'].rolling(window=window).min()
        williams_r = -100 * ((high_max - data['Close']) / (high_max - low_min))
        return williams_r
    
    def calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close_prev = np.abs(data['High'] - data['Close'].shift())
        low_close_prev = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=window).mean()
        return atr
    
    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = []
        obv.append(0)
        
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        return pd.Series(obv, index=data.index)
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators and return as DataFrame"""
        try:
            indicators = pd.DataFrame(index=data.index)
            
            # Moving Averages
            indicators['SMA_10'] = self.calculate_sma(data, 10)
            indicators['SMA_20'] = self.calculate_sma(data, 20)
            indicators['SMA_50'] = self.calculate_sma(data, 50)
            indicators['SMA_200'] = self.calculate_sma(data, 200)
            
            indicators['EMA_12'] = self.calculate_ema(data, 12)
            indicators['EMA_26'] = self.calculate_ema(data, 26)
            
            # RSI
            indicators['RSI'] = self.calculate_rsi(data, 14)
            
            # MACD
            macd, signal, histogram = self.calculate_macd(data)
            indicators['MACD'] = macd
            indicators['MACD_Signal'] = signal
            indicators['MACD_Histogram'] = histogram
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data)
            indicators['BB_Upper'] = bb_upper
            indicators['BB_Middle'] = bb_middle
            indicators['BB_Lower'] = bb_lower
            
            # Stochastic
            stoch_k, stoch_d = self.calculate_stochastic(data)
            indicators['Stoch_K'] = stoch_k
            indicators['Stoch_D'] = stoch_d
            
            # Williams %R
            indicators['Williams_R'] = self.calculate_williams_r(data)
            
            # ATR
            indicators['ATR'] = self.calculate_atr(data)
            
            # OBV
            indicators['OBV'] = self.calculate_obv(data)
            
            return indicators
            
        except Exception as e:
            st.error(f"Error calculating technical indicators: {str(e)}")
            return pd.DataFrame()
    
    def get_signal_summary(self, indicators: pd.DataFrame, current_price: float) -> dict:
        """Generate trading signals based on technical indicators"""
        signals = {}
        
        try:
            latest = indicators.iloc[-1]
            
            # RSI Signal
            if latest['RSI'] > 70:
                signals['RSI'] = 'SELL'
            elif latest['RSI'] < 30:
                signals['RSI'] = 'BUY'
            else:
                signals['RSI'] = 'HOLD'
            
            # MACD Signal
            if latest['MACD'] > latest['MACD_Signal']:
                signals['MACD'] = 'BUY'
            else:
                signals['MACD'] = 'SELL'
            
            # Moving Average Signal
            if current_price > latest['SMA_20'] > latest['SMA_50']:
                signals['MA'] = 'BUY'
            elif current_price < latest['SMA_20'] < latest['SMA_50']:
                signals['MA'] = 'SELL'
            else:
                signals['MA'] = 'HOLD'
            
            # Bollinger Bands Signal
            bb_position = (current_price - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
            if bb_position > 0.8:
                signals['BB'] = 'SELL'
            elif bb_position < 0.2:
                signals['BB'] = 'BUY'
            else:
                signals['BB'] = 'HOLD'
            
            return signals
            
        except Exception as e:
            st.error(f"Error generating signal summary: {str(e)}")
            return {}
