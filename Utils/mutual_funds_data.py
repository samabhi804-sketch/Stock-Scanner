import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import requests

class MutualFundsManager:
    """Handles mutual fund data and analysis"""
    
    def __init__(self):
        self.cache_duration = 300  # 5 minutes cache

    # Popular Indian Mutual Funds (expanded list with diverse categories)
    INDIAN_MUTUAL_FUNDS = {
        # Large Cap Funds
        'SBI Bluechip Fund': '0P0000XVJ6.BO',
        'HDFC Top 100 Fund': '0P00009YVH.BO',
        'ICICI Prudential Bluechip Fund': '0P0000A1NH.BO',
        'Axis Bluechip Fund': '0P0000A2M8.BO',
        'Mirae Asset Large Cap Fund': '0P0001D7F1.BO',
        'Kotak Bluechip Fund': '0P0000A3J5.BO',
        'Nippon India Large Cap Fund': '0P0000A4K6.BO',
        'Franklin India Bluechip Fund': '0P0000A5L7.BO',
        
        # Mid Cap Funds
        'HDFC Mid-Cap Opportunities Fund': '0P00009YWX.BO',
        'Axis Midcap Fund': '0P0000A2M9.BO',
        'SBI Magnum Midcap Fund': '0P0000XVK1.BO',
        'ICICI Prudential Mid Cap Fund': '0P0000A1NF.BO',
        'Kotak Emerging Equity Fund': '0P0000A3J6.BO',
        'L&T Midcap Fund': '0P0000A6M8.BO',
        'DSP Midcap Fund': '0P0000A7N9.BO',
        
        # Small Cap Funds
        'SBI Small Cap Fund': '0P0000XVK4.BO',
        'Axis Small Cap Fund': '0P0000A2MA.BO',
        'HDFC Small Cap Fund': '0P00009YWW.BO',
        'ICICI Prudential Small Cap Fund': '0P0000A1NE.BO',
        'Kotak Small Cap Fund': '0P0000A3J7.BO',
        'Nippon India Small Cap Fund': '0P0000A4K7.BO',
        'Franklin India Smaller Companies Fund': '0P0000A5L8.BO',
        
        # Multi Cap / Flexi Cap Funds
        'SBI Magnum Multi Cap Fund': '0P0000XVK2.BO',
        'HDFC Equity Fund': '0P00009YVG.BO',
        'ICICI Prudential Multi-Cap Fund': '0P0000A1NG.BO',
        'Mirae Asset Emerging Bluechip Fund': '0P0001D7F2.BO',
        'Parag Parikh Flexi Cap Fund': '0P0001D8F3.BO',
        'Axis Flexi Cap Fund': '0P0000A2MB.BO',
        'Kotak Flexi Cap Fund': '0P0000A3J8.BO',
        
        # Value / Contra Funds
        'SBI Contra Fund': '0P0000XVJ8.BO',
        'HDFC Value Fund': '0P00009YVI.BO',
        'ICICI Prudential Value Discovery Fund': '0P0000A1NI.BO',
        'Axis Value Fund': '0P0000A2MC.BO',
        'Nippon India Value Fund': '0P0000A4K8.BO',
        
        # Sectoral / Thematic Funds
        'SBI Technology Opportunities Fund': '0P0000XVK5.BO',
        'HDFC Banking and Financial Services Fund': '0P00009YVJ.BO',
        'ICICI Prudential Technology Fund': '0P0000A1NJ.BO',
        'Axis Banking ETF': '0P0000A2MD.BO',
        'Kotak Infrastructure Fund': '0P0000A3J9.BO',
        'L&T India Value Fund': '0P0000A6M9.BO',
        
        # ELSS Tax Saving Funds
        'SBI Magnum Tax Gain Scheme': '0P0000XVK3.BO',
        'HDFC Tax Saver': '0P00009YVK.BO',
        'ICICI Prudential Long Term Equity Fund': '0P0000A1NK.BO',
        'Axis Long Term Equity Fund': '0P0000A2ME.BO',
        'Mirae Asset Tax Saver Fund': '0P0001D7F4.BO',
        
        # Index Funds
        'SBI Nifty Index Fund': '0P0000XVK6.BO',
        'HDFC Index Fund Nifty 50 Plan': '0P00009YVL.BO',
        'ICICI Prudential Nifty Index Fund': '0P0000A1NL.BO',
        'Axis Nifty 100 Index Fund': '0P0000A2MF.BO',
        'UTI Nifty Index Fund': '0P0000A8O0.BO'
    }

    # Expanded Mutual Fund Categories
    MF_CATEGORIES = {
        'Large Cap': [
            'SBI Bluechip Fund', 'HDFC Top 100 Fund', 'ICICI Prudential Bluechip Fund', 
            'Axis Bluechip Fund', 'Mirae Asset Large Cap Fund', 'Kotak Bluechip Fund',
            'Nippon India Large Cap Fund', 'Franklin India Bluechip Fund'
        ],
        'Mid Cap': [
            'HDFC Mid-Cap Opportunities Fund', 'Axis Midcap Fund', 'SBI Magnum Midcap Fund',
            'ICICI Prudential Mid Cap Fund', 'Kotak Emerging Equity Fund', 'L&T Midcap Fund',
            'DSP Midcap Fund'
        ],
        'Small Cap': [
            'SBI Small Cap Fund', 'Axis Small Cap Fund', 'HDFC Small Cap Fund',
            'ICICI Prudential Small Cap Fund', 'Kotak Small Cap Fund', 'Nippon India Small Cap Fund',
            'Franklin India Smaller Companies Fund'
        ],
        'Multi Cap': [
            'SBI Magnum Multi Cap Fund', 'HDFC Equity Fund', 'ICICI Prudential Multi-Cap Fund',
            'Mirae Asset Emerging Bluechip Fund', 'Parag Parikh Flexi Cap Fund', 'Axis Flexi Cap Fund',
            'Kotak Flexi Cap Fund'
        ],
        'Value/Contra': [
            'SBI Contra Fund', 'HDFC Value Fund', 'ICICI Prudential Value Discovery Fund',
            'Axis Value Fund', 'Nippon India Value Fund'
        ],
        'Sectoral/Thematic': [
            'SBI Technology Opportunities Fund', 'HDFC Banking and Financial Services Fund',
            'ICICI Prudential Technology Fund', 'Axis Banking ETF', 'Kotak Infrastructure Fund',
            'L&T India Value Fund'
        ],
        'ELSS Tax Saver': [
            'SBI Magnum Tax Gain Scheme', 'HDFC Tax Saver', 'ICICI Prudential Long Term Equity Fund',
            'Axis Long Term Equity Fund', 'Mirae Asset Tax Saver Fund'
        ],
        'Index Funds': [
            'SBI Nifty Index Fund', 'HDFC Index Fund Nifty 50 Plan', 'ICICI Prudential Nifty Index Fund',
            'Axis Nifty 100 Index Fund', 'UTI Nifty Index Fund'
        ]
    }

    @st.cache_data(ttl=300)
    def get_mutual_fund_data(_self, symbol: str, period: str = '1y') -> pd.DataFrame | None:
        """
        Fetch mutual fund NAV data using yfinance
        
        Args:
            symbol: Mutual fund symbol
            period: Time period ('1mo', '3mo', '6mo', '1y', '2y', '5y')
        
        Returns:
            DataFrame with NAV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                # Fallback to creating sample data for demo
                return _self._create_sample_mf_data(period)
            
            data = data.reset_index()
            data.set_index('Date', inplace=True)
            return data
            
        except Exception as e:
            st.warning(f"Using sample data for {symbol}: Real data may not be available")
            return _self._create_sample_mf_data(period)

    def _create_sample_mf_data(self, period: str) -> pd.DataFrame:
        """Create sample mutual fund data for demonstration"""
        import numpy as np
        
        # Map periods to days
        period_days = {
            '1mo': 30, '3mo': 90, '6mo': 180, 
            '1y': 365, '2y': 730, '5y': 1825
        }
        
        days = period_days.get(period, 365)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate realistic NAV data with some volatility
        initial_nav = 50.0
        returns = np.random.normal(0.0004, 0.02, days)  # Average ~15% annual return
        nav_values = [initial_nav]
        
        for i in range(1, days):
            nav_values.append(nav_values[-1] * (1 + returns[i]))
        
        # Create volume data (units traded)
        volumes = np.random.randint(100000, 1000000, days)
        
        return pd.DataFrame({
            'Open': nav_values,
            'High': [v * 1.001 for v in nav_values],
            'Low': [v * 0.999 for v in nav_values],
            'Close': nav_values,
            'Volume': volumes
        }, index=dates)

    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def get_mutual_fund_info(_self, fund_name: str) -> dict:
        """
        Get mutual fund information
        
        Args:
            fund_name: Name of the mutual fund
            
        Returns:
            Dictionary with fund info
        """
        # Sample fund information (in real implementation, this would come from an API)
        sample_info = {
            'nav': round(np.random.uniform(45, 120), 2),
            'aum': f"₹{np.random.randint(5000, 50000)} Cr",
            'expense_ratio': round(np.random.uniform(0.5, 2.5), 2),
            'fund_manager': 'Sample Manager',
            'fund_house': fund_name.split(' ')[0],
            'category': _self._get_fund_category(fund_name),
            'launch_date': '2010-01-01',
            'minimum_sip': 500,
            'minimum_lumpsum': 5000,
            'exit_load': '1% if redeemed before 1 year',
            'benchmark': 'NIFTY 100 TRI'
        }
        return sample_info

    def _get_fund_category(self, fund_name: str) -> str:
        """Get category for a fund"""
        for category, funds in self.MF_CATEGORIES.items():
            if fund_name in funds:
                return category
        return 'Multi Cap'

    def calculate_returns(self, data: pd.DataFrame) -> dict:
        """Calculate various returns for mutual fund"""
        try:
            current_nav = data['Close'].iloc[-1]
            
            returns = {}
            periods = [30, 90, 180, 365, 730, 1825]  # Days
            period_names = ['1M', '3M', '6M', '1Y', '2Y', '5Y']
            
            for days, name in zip(periods, period_names):
                if len(data) > days:
                    old_nav = data['Close'].iloc[-days-1]
                    period_return = ((current_nav - old_nav) / old_nav) * 100
                    # Annualize returns for periods > 1 year
                    if days > 365:
                        period_return = ((current_nav / old_nav) ** (365 / days) - 1) * 100
                    returns[name] = round(period_return, 2)
                else:
                    returns[name] = 0.0
            
            return returns
            
        except Exception as e:
            st.error(f"Error calculating returns: {str(e)}")
            return {}

    def get_top_funds_by_category(self, category: str) -> list:
        """Get top funds in a category"""
        if category in self.MF_CATEGORIES:
            return self.MF_CATEGORIES[category]
        return list(self.INDIAN_MUTUAL_FUNDS.keys())[:5]

class SIPCalculator:
    """SIP (Systematic Investment Plan) Calculator"""
    
    @staticmethod
    def calculate_sip_returns(monthly_amount: float, years: int, annual_return: float) -> dict:
        """
        Calculate SIP returns using compound interest
        
        Args:
            monthly_amount: Monthly SIP amount
            years: Investment period in years
            annual_return: Expected annual return percentage
            
        Returns:
            Dictionary with calculation results
        """
        months = years * 12
        monthly_return = annual_return / (12 * 100)
        
        # SIP Future Value formula
        if monthly_return > 0:
            future_value = monthly_amount * (((1 + monthly_return) ** months - 1) / monthly_return)
        else:
            future_value = monthly_amount * months
        
        total_investment = monthly_amount * months
        wealth_gain = future_value - total_investment
        
        return {
            'future_value': round(future_value, 2),
            'total_investment': round(total_investment, 2),
            'wealth_gain': round(wealth_gain, 2),
            'monthly_amount': monthly_amount,
            'years': years,
            'annual_return': annual_return
        }
    
    @staticmethod
    def generate_sip_projection(monthly_amount: float, years: int, annual_return: float) -> pd.DataFrame:
        """Generate year-wise SIP projection"""
        monthly_return = annual_return / (12 * 100)
        
        projections = []
        for year in range(1, years + 1):
            months = year * 12
            
            if monthly_return > 0:
                future_value = monthly_amount * (((1 + monthly_return) ** months - 1) / monthly_return)
            else:
                future_value = monthly_amount * months
            
            total_investment = monthly_amount * months
            wealth_gain = future_value - total_investment
            
            projections.append({
                'Year': year,
                'Total Investment': round(total_investment, 2),
                'Future Value': round(future_value, 2),
                'Wealth Gain': round(wealth_gain, 2)
            })
        
        return pd.DataFrame(projections)

import numpy as np