"""
Comprehensive Test Suite for Multi-Market Investment Platform

This test suite covers:
1. Stock Data Management
2. Technical Indicators Calculation
3. AI Prediction System
4. Mutual Funds Management
5. SIP Calculator
6. Market Data Functions
7. Integration Tests
"""

# import pytest  # Commented out - not needed for basic testing
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add utils to path for testing
sys.path.append('Utils')

from stock_data import StockDataManager
from technical_indicators import TechnicalIndicators
from ai_predictor import AIPredictor
from mutual_funds_data import MutualFundsManager, SIPCalculator
from market_data import *


class TestStockDataManager:
    """Test cases for Stock Data Management"""
    
    def setup_method(self):
        self.stock_manager = StockDataManager()
    
    def test_stock_data_retrieval(self):
        """Test fetching stock data"""
        # Test with valid Indian stock
        data = self.stock_manager.get_stock_data('RELIANCE.NS', '1mo')
        assert data is not None
        assert not data.empty
        assert 'Close' in data.columns
        assert 'Volume' in data.columns
        
        # Test with valid US stock  
        data_us = self.stock_manager.get_stock_data('AAPL', '1mo')
        assert data_us is not None
        assert not data_us.empty
    
    def test_stock_info_retrieval(self):
        """Test fetching stock information"""
        info = self.stock_manager.get_stock_info('AAPL')
        assert isinstance(info, dict)
        # Note: Some keys might not be available depending on yfinance response
    
    def test_symbol_validation(self):
        """Test symbol validation for different markets"""
        # Test Indian market symbol formatting
        formatted = self.stock_manager.validate_symbol('RELIANCE', 'Indian')
        assert formatted == 'RELIANCE.NS'
        
        # Test US market symbol (should remain unchanged)
        formatted_us = self.stock_manager.validate_symbol('AAPL', 'US')
        assert formatted_us == 'AAPL'
    
    def test_market_hours(self):
        """Test market hours information"""
        indian_hours = self.stock_manager.get_market_hours('Indian')
        assert 'open_time' in indian_hours
        assert 'close_time' in indian_hours
        assert indian_hours['timezone'] == 'Asia/Kolkata'
        
        us_hours = self.stock_manager.get_market_hours('US')
        assert us_hours['timezone'] == 'America/New_York'


class TestTechnicalIndicators:
    """Test cases for Technical Indicators"""
    
    def setup_method(self):
        self.tech_indicators = TechnicalIndicators()
        # Create sample data for testing
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic stock price data
        initial_price = 100
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.sample_data = pd.DataFrame({
            'Open': [p * 0.99 for p in prices],
            'High': [p * 1.02 for p in prices], 
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    
    def test_sma_calculation(self):
        """Test Simple Moving Average calculation"""
        sma_20 = self.tech_indicators.calculate_sma(self.sample_data, 20)
        assert len(sma_20) == len(self.sample_data)
        assert not np.isnan(sma_20.iloc[-1])  # Last value should not be NaN
        
        # Test that SMA is actually the mean of last 20 values
        manual_sma = self.sample_data['Close'].iloc[-20:].mean()
        assert abs(sma_20.iloc[-1] - manual_sma) < 0.01
    
    def test_ema_calculation(self):
        """Test Exponential Moving Average calculation"""
        ema_12 = self.tech_indicators.calculate_ema(self.sample_data, 12)
        assert len(ema_12) == len(self.sample_data)
        assert not np.isnan(ema_12.iloc[-1])
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        rsi = self.tech_indicators.calculate_rsi(self.sample_data, 14)
        assert len(rsi) == len(self.sample_data)
        
        # RSI should be between 0 and 100
        last_rsi = rsi.iloc[-1]
        assert 0 <= last_rsi <= 100
    
    def test_macd_calculation(self):
        """Test MACD calculation"""
        macd, signal, histogram = self.tech_indicators.calculate_macd(self.sample_data)
        assert len(macd) == len(self.sample_data)
        assert len(signal) == len(self.sample_data)
        assert len(histogram) == len(self.sample_data)
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        upper, middle, lower = self.tech_indicators.calculate_bollinger_bands(self.sample_data)
        
        # Upper band should be greater than middle, middle greater than lower
        assert upper.iloc[-1] > middle.iloc[-1] > lower.iloc[-1]
    
    def test_all_indicators(self):
        """Test comprehensive indicator calculation"""
        indicators = self.tech_indicators.calculate_all_indicators(self.sample_data)
        
        expected_columns = [
            'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Middle', 'BB_Lower',
            'Stoch_K', 'Stoch_D', 'Williams_R', 'ATR', 'OBV'
        ]
        
        for col in expected_columns:
            assert col in indicators.columns
    
    def test_signal_summary(self):
        """Test trading signal generation"""
        indicators = self.tech_indicators.calculate_all_indicators(self.sample_data)
        current_price = self.sample_data['Close'].iloc[-1]
        
        signals = self.tech_indicators.get_signal_summary(indicators, current_price)
        
        assert isinstance(signals, dict)
        assert 'RSI' in signals
        assert 'MACD' in signals
        assert 'MA' in signals
        assert 'BB' in signals
        
        # Check that signals are valid
        for signal in signals.values():
            assert signal in ['BUY', 'SELL', 'HOLD']


class TestAIPredictor:
    """Test cases for AI Prediction System"""
    
    def setup_method(self):
        self.ai_predictor = AIPredictor()
        self.tech_indicators = TechnicalIndicators()
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        initial_price = 100
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.sample_data = pd.DataFrame({
            'Open': [p * 0.99 for p in prices],
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        self.indicators = self.tech_indicators.calculate_all_indicators(self.sample_data)
    
    def test_technical_data_preparation(self):
        """Test preparation of technical data for AI"""
        tech_data = self.ai_predictor.prepare_technical_data(self.sample_data, self.indicators)
        
        expected_keys = [
            'current_price', 'price_change_1d', 'price_change_5d',
            'volume_ratio', 'rsi', 'macd', 'macd_signal',
            'bb_position', 'sma_20_position', 'sma_50_position',
            'williams_r', 'stoch_k'
        ]
        
        for key in expected_keys:
            assert key in tech_data
        
        assert isinstance(tech_data['current_price'], (int, float))
        assert isinstance(tech_data['rsi'], (int, float))
    
    def test_bb_position_calculation(self):
        """Test Bollinger Band position calculation"""
        current_price = self.sample_data['Close'].iloc[-1]
        indicators_series = self.indicators.iloc[-1]
        
        bb_position = self.ai_predictor.calculate_bb_position(current_price, indicators_series)
        
        assert isinstance(bb_position, float)
        assert 0 <= bb_position <= 100 or bb_position == 50.0  # 50.0 is default fallback
    
    def test_rule_based_prediction(self):
        """Test rule-based prediction system"""
        tech_data = self.ai_predictor.prepare_technical_data(self.sample_data, self.indicators)
        
        signal, confidence, explanation = self.ai_predictor.generate_rule_based_prediction(tech_data)
        
        assert signal in ['BUY', 'SELL', 'HOLD']
        assert 0 <= confidence <= 100
        assert isinstance(explanation, str)
        assert len(explanation) > 0
    
    def test_signal_prediction(self):
        """Test complete signal prediction"""
        signal, confidence, explanation = self.ai_predictor.predict_signal(self.sample_data, self.indicators)
        
        assert signal in ['BUY', 'SELL', 'HOLD']
        assert 0 <= confidence <= 100
        assert isinstance(explanation, str)


class TestMutualFundsManager:
    """Test cases for Mutual Funds Management"""
    
    def setup_method(self):
        self.mf_manager = MutualFundsManager()
    
    def test_mutual_funds_data_structure(self):
        """Test mutual funds data structures"""
        assert isinstance(self.mf_manager.INDIAN_MUTUAL_FUNDS, dict)
        assert isinstance(self.mf_manager.MF_CATEGORIES, dict)
        assert len(self.mf_manager.INDIAN_MUTUAL_FUNDS) > 0
    
    def test_fund_data_retrieval(self):
        """Test mutual fund data fetching"""
        sample_symbol = list(self.mf_manager.INDIAN_MUTUAL_FUNDS.values())[0]
        data = self.mf_manager.get_mutual_fund_data(sample_symbol, '1y')
        
        assert data is not None
        assert not data.empty
        assert 'Close' in data.columns
    
    def test_fund_info_retrieval(self):
        """Test mutual fund information retrieval"""
        sample_fund = list(self.mf_manager.INDIAN_MUTUAL_FUNDS.keys())[0]
        info = self.mf_manager.get_mutual_fund_info(sample_fund)
        
        assert isinstance(info, dict)
        assert 'nav' in info
        assert 'category' in info
        assert 'expense_ratio' in info
    
    def test_returns_calculation(self):
        """Test returns calculation for mutual funds"""
        # Create sample NAV data
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        nav_values = [50 + i * 0.01 for i in range(len(dates))]  # Simple upward trend
        
        sample_data = pd.DataFrame({
            'Close': nav_values,
            'Volume': [100000] * len(dates)
        }, index=dates)
        
        returns = self.mf_manager.calculate_returns(sample_data)
        
        assert isinstance(returns, dict)
        expected_periods = ['1M', '3M', '6M', '1Y', '2Y', '5Y']
        for period in expected_periods:
            if period in returns:
                assert isinstance(returns[period], (int, float))
    
    def test_category_funds(self):
        """Test getting funds by category"""
        category = 'Large Cap'
        funds = self.mf_manager.get_top_funds_by_category(category)
        
        assert isinstance(funds, list)
        assert len(funds) > 0


class TestSIPCalculator:
    """Test cases for SIP Calculator"""
    
    def setup_method(self):
        self.sip_calc = SIPCalculator()
    
    def test_sip_calculation_basic(self):
        """Test basic SIP calculation"""
        monthly_amount = 5000
        years = 10
        annual_return = 12.0
        
        result = self.sip_calc.calculate_sip_returns(monthly_amount, years, annual_return)
        
        assert isinstance(result, dict)
        assert 'future_value' in result
        assert 'total_investment' in result
        assert 'wealth_gain' in result
        
        # Basic validations
        assert result['total_investment'] == monthly_amount * years * 12
        assert result['future_value'] > result['total_investment']  # Should grow
        assert result['wealth_gain'] == result['future_value'] - result['total_investment']
    
    def test_sip_calculation_edge_cases(self):
        """Test SIP calculation edge cases"""
        # Test with zero return
        result_zero = self.sip_calc.calculate_sip_returns(1000, 5, 0)
        assert result_zero['future_value'] == result_zero['total_investment']
        
        # Test with minimum values
        result_min = self.sip_calc.calculate_sip_returns(500, 1, 1.0)
        assert result_min['total_investment'] == 500 * 12
        assert result_min['future_value'] > result_min['total_investment']
    
    def test_sip_projection_generation(self):
        """Test year-wise SIP projection"""
        monthly_amount = 3000
        years = 15
        annual_return = 10.0
        
        projection = self.sip_calc.generate_sip_projection(monthly_amount, years, annual_return)
        
        assert isinstance(projection, pd.DataFrame)
        assert len(projection) == years
        assert 'Year' in projection.columns
        assert 'Total Investment' in projection.columns
        assert 'Future Value' in projection.columns
        assert 'Wealth Gain' in projection.columns
        
        # Test that values increase year over year
        assert projection['Future Value'].iloc[-1] > projection['Future Value'].iloc[0]
        assert projection['Total Investment'].iloc[-1] == monthly_amount * years * 12
    
    def test_sip_compound_growth(self):
        """Test that SIP calculations reflect compound growth"""
        monthly_amount = 10000
        years = 20
        annual_return = 15.0
        
        result = self.sip_calc.calculate_sip_returns(monthly_amount, years, annual_return)
        
        # With 15% return over 20 years, wealth gain should be substantial
        wealth_gain_percentage = (result['wealth_gain'] / result['total_investment']) * 100
        assert wealth_gain_percentage > 100  # Should more than double the investment


class TestMarketDataFunctions:
    """Test cases for Market Data Functions"""
    
    def test_market_timezone(self):
        """Test market timezone functions"""
        indian_tz = get_market_timezone('Indian')
        us_tz = get_market_timezone('US')
        
        assert indian_tz is not None
        assert us_tz is not None
    
    def test_trading_hours(self):
        """Test trading hours information"""
        indian_hours = get_trading_hours('Indian')
        us_hours = get_trading_hours('US')
        
        assert 'regular' in indian_hours
        assert 'currency_symbol' in indian_hours
        assert indian_hours['currency_symbol'] == '₹'
        assert us_hours['currency_symbol'] == '$'
    
    def test_market_data_structures(self):
        """Test market data structures"""
        assert isinstance(INDIAN_STOCKS, dict)
        assert isinstance(US_STOCKS, dict)
        assert len(INDIAN_STOCKS) > 0
        assert len(US_STOCKS) > 0
        
        # Test that Indian stocks have proper suffixes
        for symbol in INDIAN_STOCKS.keys():
            assert symbol.endswith('.NS') or symbol.endswith('.BO')
    
    def test_popular_stocks_filtering(self):
        """Test popular stocks filtering by sector"""
        indian_tech_stocks = get_popular_stocks('Indian', 'IT')
        assert isinstance(indian_tech_stocks, dict)
        
        us_tech_stocks = get_popular_stocks('US', 'Technology')
        assert isinstance(us_tech_stocks, dict)
    
    def test_currency_formatting(self):
        """Test currency formatting for different markets"""
        indian_formatted = format_currency(10000.50, 'Indian')
        us_formatted = format_currency(10000.50, 'US')
        
        assert '₹' in indian_formatted
        assert '$' in us_formatted
        assert '10,000.50' in indian_formatted
        assert '10,000.50' in us_formatted


class TestIntegration:
    """Integration tests for the complete system"""
    
    def setup_method(self):
        self.stock_manager = StockDataManager()
        self.tech_indicators = TechnicalIndicators()
        self.ai_predictor = AIPredictor()
        self.mf_manager = MutualFundsManager()
        self.sip_calc = SIPCalculator()
    
    def test_complete_stock_analysis_workflow(self):
        """Test complete workflow from data fetch to prediction"""
        # This test uses sample data to avoid API dependencies
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        initial_price = 100
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        sample_data = pd.DataFrame({
            'Open': [p * 0.99 for p in prices],
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Test complete workflow
        indicators = self.tech_indicators.calculate_all_indicators(sample_data)
        assert not indicators.empty
        
        signal, confidence, explanation = self.ai_predictor.predict_signal(sample_data, indicators)
        assert signal in ['BUY', 'SELL', 'HOLD']
        assert 0 <= confidence <= 100
        assert len(explanation) > 0
    
    def test_mutual_fund_analysis_workflow(self):
        """Test complete mutual fund analysis workflow"""
        sample_fund = list(self.mf_manager.INDIAN_MUTUAL_FUNDS.keys())[0]
        
        # Get fund data (will use sample data if real data unavailable)
        fund_data = self.mf_manager.get_mutual_fund_data('sample_symbol', '1y')
        fund_info = self.mf_manager.get_mutual_fund_info(sample_fund)
        returns = self.mf_manager.calculate_returns(fund_data)
        
        assert fund_data is not None
        assert isinstance(fund_info, dict)
        assert isinstance(returns, dict)
    
    def test_sip_calculator_workflow(self):
        """Test complete SIP calculation workflow"""
        monthly_amount = 5000
        years = 10
        annual_return = 12.0
        
        # Calculate returns
        sip_results = self.sip_calc.calculate_sip_returns(monthly_amount, years, annual_return)
        
        # Generate projections
        projections = self.sip_calc.generate_sip_projection(monthly_amount, years, annual_return)
        
        assert isinstance(sip_results, dict)
        assert isinstance(projections, pd.DataFrame)
        assert len(projections) == years


def run_test_suite():
    """Run the complete test suite and generate a report"""
    
    print("🧪 Starting Multi-Market Investment Platform Test Suite")
    print("=" * 60)
    
    test_classes = [
        TestStockDataManager,
        TestTechnicalIndicators, 
        TestAIPredictor,
        TestMutualFundsManager,
        TestSIPCalculator,
        TestMarketDataFunctions,
        TestIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}")
        print("-" * 40)
        
        test_instance = test_class()
        if hasattr(test_instance, 'setup_method'):
            test_instance.setup_method()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                getattr(test_instance, test_method)()
                print(f"✅ {test_method}")
                passed_tests += 1
            except Exception as e:
                print(f"❌ {test_method}: {str(e)}")
                failed_tests.append(f"{test_class.__name__}.{test_method}: {str(e)}")
    
    # Generate test report
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY REPORT")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ✅")
    print(f"Failed: {len(failed_tests)} ❌")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests:
        print("\n❌ FAILED TESTS:")
        for failure in failed_tests:
            print(f"  - {failure}")
    
    print("\n🎯 FEATURE COVERAGE:")
    print("✅ Stock Data Management")
    print("✅ Technical Indicators (RSI, MACD, SMA, EMA, etc.)")
    print("✅ AI-Powered Predictions")
    print("✅ Mutual Funds Analysis")
    print("✅ SIP Calculator")
    print("✅ Multi-Market Support (Indian & US)")
    print("✅ Integration Testing")
    
    return passed_tests, len(failed_tests), total_tests


if __name__ == "__main__":
    run_test_suite()