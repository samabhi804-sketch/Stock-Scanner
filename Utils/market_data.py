import pytz
from datetime import datetime

# Indian Stock Market - Popular NSE stocks
INDIAN_STOCKS = {
    'RELIANCE.NS': 'Reliance Industries Ltd',
    'TCS.NS': 'Tata Consultancy Services',
    'HDFCBANK.NS': 'HDFC Bank Ltd',
    'INFY.NS': 'Infosys Ltd',
    'ICICIBANK.NS': 'ICICI Bank Ltd',
    'HINDUNILVR.NS': 'Hindustan Unilever Ltd',
    'SBIN.NS': 'State Bank of India',
    'BHARTIARTL.NS': 'Bharti Airtel Ltd',
    'ITC.NS': 'ITC Ltd',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank',
    'LT.NS': 'Larsen & Toubro Ltd',
    'AXISBANK.NS': 'Axis Bank Ltd',
    'MARUTI.NS': 'Maruti Suzuki India Ltd',
    'ASIANPAINT.NS': 'Asian Paints Ltd',
    'HCLTECH.NS': 'HCL Technologies Ltd',
    'WIPRO.NS': 'Wipro Ltd',
    'ULTRACEMCO.NS': 'UltraTech Cement Ltd',
    'TITAN.NS': 'Titan Company Ltd',
    'NESTLEIND.NS': 'Nestle India Ltd',
    'POWERGRID.NS': 'Power Grid Corporation',
    'NTPC.NS': 'NTPC Ltd',
    'TECHM.NS': 'Tech Mahindra Ltd',
    'BAJFINANCE.NS': 'Bajaj Finance Ltd',
    'ONGC.NS': 'Oil & Natural Gas Corporation',
    'M&M.NS': 'Mahindra & Mahindra Ltd',
    'SUNPHARMA.NS': 'Sun Pharmaceutical Industries',
    'TATASTEEL.NS': 'Tata Steel Ltd',
    'BAJAJFINSV.NS': 'Bajaj Finserv Ltd',
    'DIVISLAB.NS': 'Divi\'s Laboratories Ltd',
    'DRREDDY.NS': 'Dr. Reddy\'s Laboratories'
}

# US Stock Market - Popular NASDAQ/NYSE stocks
US_STOCKS = {
    'AAPL': 'Apple Inc',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc Class A',
    'GOOG': 'Alphabet Inc Class C',
    'AMZN': 'Amazon.com Inc',
    'TSLA': 'Tesla Inc',
    'META': 'Meta Platforms Inc',
    'NVDA': 'NVIDIA Corporation',
    'NFLX': 'Netflix Inc',
    'AMD': 'Advanced Micro Devices',
    'PYPL': 'PayPal Holdings Inc',
    'ADBE': 'Adobe Inc',
    'CRM': 'Salesforce Inc',
    'INTC': 'Intel Corporation',
    'CSCO': 'Cisco Systems Inc',
    'PEP': 'PepsiCo Inc',
    'CMCSA': 'Comcast Corporation',
    'COST': 'Costco Wholesale Corporation',
    'TMUS': 'T-Mobile US Inc',
    'QCOM': 'QUALCOMM Incorporated',
    'TXN': 'Texas Instruments Incorporated',
    'AVGO': 'Broadcom Inc',
    'HON': 'Honeywell International Inc',
    'SBUX': 'Starbucks Corporation',
    'GILD': 'Gilead Sciences Inc',
    'MDLZ': 'Mondelez International Inc',
    'ADP': 'Automatic Data Processing',
    'BKNG': 'Booking Holdings Inc',
    'ISRG': 'Intuitive Surgical Inc',
    'VRTX': 'Vertex Pharmaceuticals Incorporated',
    'JD': 'JD.com Inc',
    'BIDU': 'Baidu Inc',
    'ZM': 'Zoom Video Communications',
    'MRNA': 'Moderna Inc',
    'SPY': 'SPDR S&P 500 ETF Trust',
    'QQQ': 'Invesco QQQ Trust',
    'DIS': 'The Walt Disney Company',
    'V': 'Visa Inc',
    'MA': 'Mastercard Incorporated',
    'JPM': 'JPMorgan Chase & Co',
    'JNJ': 'Johnson & Johnson',
    'WMT': 'Walmart Inc',
    'PG': 'Procter & Gamble Company',
    'UNH': 'UnitedHealth Group Incorporated',
    'HD': 'The Home Depot Inc',
    'BAC': 'Bank of America Corporation',
    'XOM': 'Exxon Mobil Corporation',
    'CVX': 'Chevron Corporation',
    'PFE': 'Pfizer Inc',
    'KO': 'The Coca-Cola Company'
}

# Market sectors for Indian stocks
INDIAN_SECTORS = {
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS'],
    'IT': ['TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS'],
    'Oil & Gas': ['RELIANCE.NS', 'ONGC.NS'],
    'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS'],
    'Automobile': ['MARUTI.NS', 'M&M.NS'],
    'Pharmaceuticals': ['SUNPHARMA.NS', 'DIVISLAB.NS', 'DRREDDY.NS'],
    'Infrastructure': ['LT.NS', 'ULTRACEMCO.NS', 'POWERGRID.NS', 'NTPC.NS']
}

# Market sectors for US stocks
US_SECTORS = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'AMD', 'ADBE', 'CRM', 'INTC', 'CSCO'],
    'Communication': ['NFLX', 'TMUS', 'CMCSA', 'DIS', 'ZM'],
    'Consumer': ['TSLA', 'PYPL', 'PEP', 'COST', 'SBUX', 'MDLZ', 'WMT', 'HD', 'KO'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'GILD', 'VRTX', 'ISRG', 'MRNA'],
    'Financial': ['V', 'MA', 'JPM', 'BAC', 'ADP', 'BKNG'],
    'Industrial': ['HON', 'TXN', 'AVGO', 'QCOM'],
    'Energy': ['XOM', 'CVX'],
    'ETF': ['SPY', 'QQQ'],
    'Chinese ADR': ['JD', 'BIDU']
}

def get_market_timezone(market: str):
    """Get timezone for the specified market"""
    if market == 'Indian':
        return pytz.timezone('Asia/Kolkata')
    else:  # US market
        return pytz.timezone('America/New_York')

def get_trading_hours(market: str) -> dict:
    """Get trading hours for the specified market"""
    if market == 'Indian':
        return {
            'pre_market': '09:00 - 09:15',
            'regular': '09:15 - 15:30',
            'post_market': '15:40 - 16:00',
            'timezone': 'Asia/Kolkata',
            'currency': 'INR',
            'currency_symbol': '₹'
        }
    else:  # US market
        return {
            'pre_market': '04:00 - 09:30',
            'regular': '09:30 - 16:00',
            'post_market': '16:00 - 20:00',
            'timezone': 'America/New_York',
            'currency': 'USD',
            'currency_symbol': '$'
        }

def is_market_open(market: str) -> bool:
    """Check if the specified market is currently open"""
    tz = get_market_timezone(market)
    current_time = datetime.now(tz)
    current_hour = current_time.hour
    current_minute = current_time.minute
    weekday = current_time.weekday()
    
    # Skip weekends
    if weekday >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    if market == 'Indian':
        # Indian market: 9:15 AM to 3:30 PM
        market_open = (current_hour == 9 and current_minute >= 15) or (9 < current_hour < 15) or (current_hour == 15 and current_minute <= 30)
    else:  # US market
        # US market: 9:30 AM to 4:00 PM
        market_open = (current_hour == 9 and current_minute >= 30) or (9 < current_hour < 16)
    
    return market_open

def get_popular_stocks(market: str, sector: str | None = None) -> dict:
    """Get popular stocks for a market, optionally filtered by sector"""
    if market == 'Indian':
        if sector and sector in INDIAN_SECTORS:
            sector_stocks = INDIAN_SECTORS[sector]
            return {symbol: INDIAN_STOCKS[symbol] for symbol in sector_stocks if symbol in INDIAN_STOCKS}
        return INDIAN_STOCKS
    else:  # US market
        if sector and sector in US_SECTORS:
            sector_stocks = US_SECTORS[sector]
            return {symbol: US_STOCKS[symbol] for symbol in sector_stocks if symbol in US_STOCKS}
        return US_STOCKS

def get_market_sectors(market: str) -> dict:
    """Get available sectors for a market"""
    if market == 'Indian':
        return INDIAN_SECTORS
    else:
        return US_SECTORS

def format_currency(amount: float, market: str) -> str:
    """Format currency based on market"""
    trading_hours = get_trading_hours(market)
    symbol = trading_hours['currency_symbol']
    
    if market == 'Indian':
        # Indian number format with commas
        return f"{symbol}{amount:,.2f}"
    else:
        # US number format
        return f"{symbol}{amount:,.2f}"

def get_market_holidays(market: str, year: int | None = None) -> list:
    """Get market holidays for the specified market and year"""
    if year is None:
        year = datetime.now().year
    
    if market == 'Indian':
        # Major Indian market holidays (simplified list)
        return [
            f"{year}-01-26",  # Republic Day
            f"{year}-03-08",  # Holi (approximate)
            f"{year}-04-14",  # Ram Navami (approximate)
            f"{year}-08-15",  # Independence Day
            f"{year}-10-02",  # Gandhi Jayanti
            f"{year}-11-12",  # Diwali (approximate)
        ]
    else:  # US market
        # Major US market holidays
        return [
            f"{year}-01-01",  # New Year's Day
            f"{year}-01-15",  # Martin Luther King Jr. Day (approximate)
            f"{year}-02-19",  # Presidents' Day (approximate)
            f"{year}-05-27",  # Memorial Day (approximate)
            f"{year}-07-04",  # Independence Day
            f"{year}-09-02",  # Labor Day (approximate)
            f"{year}-11-28",  # Thanksgiving (approximate)
            f"{year}-12-25",  # Christmas Day
        ]
