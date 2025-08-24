import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import pytz
from utils.stock_data import StockDataManager
from utils.technical_indicators import TechnicalIndicators
from utils.ai_predictor import AIPredictor
from utils.market_data import INDIAN_STOCKS, US_STOCKS, get_market_timezone
from utils.mutual_funds_data import MutualFundsManager, SIPCalculator

# Page configuration
st.set_page_config(
    page_title="Multi-Market Investment Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'selected_market' not in st.session_state:
    st.session_state.selected_market = 'Indian'
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = 'RELIANCE.NS'

# Initialize managers
@st.cache_resource
def get_managers():
    stock_manager = StockDataManager()
    tech_indicators = TechnicalIndicators()
    ai_predictor = AIPredictor()
    mf_manager = MutualFundsManager()
    sip_calc = SIPCalculator()
    return stock_manager, tech_indicators, ai_predictor, mf_manager, sip_calc

stock_manager, tech_indicators, ai_predictor, mf_manager, sip_calc = get_managers()

# Sidebar
st.sidebar.title("🌐 Market Selection")

# Market selector
market_options = ['Indian', 'US']
selected_market = st.sidebar.selectbox(
    "Select Market",
    market_options,
    index=market_options.index(st.session_state.selected_market)
)

if selected_market != st.session_state.selected_market:
    st.session_state.selected_market = selected_market
    st.rerun()

# Stock selection based on market
if selected_market == 'Indian':
    stock_list = INDIAN_STOCKS
    currency_symbol = '₹'
    market_suffix = '.NS'
else:
    stock_list = US_STOCKS
    currency_symbol = '$'
    market_suffix = ''

st.sidebar.subheader(f"{selected_market} Stocks")

# Popular stocks
selected_stock = st.sidebar.selectbox(
    "Popular Stocks",
    options=list(stock_list.keys()),
    format_func=lambda x: f"{x} - {stock_list[x]}"
)

# Custom stock input
custom_stock = st.sidebar.text_input(
    "Or Enter Custom Symbol",
    placeholder="e.g., AAPL or RELIANCE.NS"
)

# Use custom stock if provided
if custom_stock:
    symbol = custom_stock.upper()
    if selected_market == 'Indian' and not (symbol.endswith('.NS') or symbol.endswith('.BO')):
        symbol = f"{symbol}.NS"
    selected_stock_symbol = symbol
    stock_name = custom_stock.upper()
else:
    selected_stock_symbol = selected_stock
    if selected_market == 'Indian' and not selected_stock_symbol.endswith(('.NS', '.BO')):
        selected_stock_symbol = f"{selected_stock}{market_suffix}"
    stock_name = stock_list.get(selected_stock, selected_stock)

# Time period selection
time_periods = {
    '1M': '1mo',
    '3M': '3mo',
    '6M': '6mo',
    '1Y': '1y',
    '2Y': '2y',
    '5Y': '5y'
}

selected_period = st.sidebar.selectbox(
    "Time Period",
    options=list(time_periods.keys()),
    index=3  # Default to 1Y
)

# Main content
st.title("📈 Multi-Market Investment Platform")

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["📊 Stock Screener", "🏛️ Mutual Funds", "💰 SIP Calculator"])

with tab1:
    st.markdown(f"**Market:** {selected_market} | **Stock:** {stock_name}")

    # Fetch stock data
    try:
        with st.spinner(f"Fetching data for {selected_stock_symbol}..."):
            stock_data = stock_manager.get_stock_data(selected_stock_symbol, time_periods[selected_period])
            
            if stock_data is None or stock_data.empty:
                st.error(f"No data found for {selected_stock_symbol}. Please check the symbol and try again.")
                st.stop()

            # Get current stock info
            stock_info = stock_manager.get_stock_info(selected_stock_symbol)
            
            # Calculate technical indicators
            indicators = tech_indicators.calculate_all_indicators(stock_data)
            
            # Get AI prediction
            ai_signal, confidence, explanation = ai_predictor.predict_signal(stock_data, indicators)

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.stop()

    # Display current stock information
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_price = stock_data['Close'].iloc[-1]
        st.metric(
            "Current Price",
            f"{currency_symbol}{current_price:.2f}",
            delta=f"{stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]:.2f}"
        )

    with col2:
        volume = stock_data['Volume'].iloc[-1]
        st.metric("Volume", f"{volume:,}")

    with col3:
        market_cap = stock_info.get('marketCap', 'N/A')
        if market_cap != 'N/A':
            if market_cap > 1e12:
                market_cap_display = f"{currency_symbol}{market_cap/1e12:.2f}T"
            elif market_cap > 1e9:
                market_cap_display = f"{currency_symbol}{market_cap/1e9:.2f}B"
            elif market_cap > 1e6:
                market_cap_display = f"{currency_symbol}{market_cap/1e6:.2f}M"
            else:
                market_cap_display = f"{currency_symbol}{market_cap:,.0f}"
        else:
            market_cap_display = "N/A"
        st.metric("Market Cap", market_cap_display)

    with col4:
        pe_ratio = stock_info.get('trailingPE', 'N/A')
        pe_display = f"{pe_ratio:.2f}" if pe_ratio != 'N/A' and pe_ratio is not None else "N/A"
        st.metric("P/E Ratio", pe_display)

    # AI Prediction Section
    st.subheader("🤖 AI-Powered Signal Prediction")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Signal display with color coding
        signal_colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'orange'}
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {signal_colors[ai_signal]}20; border: 2px solid {signal_colors[ai_signal]};">
                <h2 style="color: {signal_colors[ai_signal]}; margin: 0;">{ai_signal}</h2>
                <p style="margin: 5px 0;">Confidence: {confidence:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.subheader("Analysis Explanation")
        st.markdown(explanation)

    # Technical Indicators Summary
    st.subheader("📊 Technical Indicators Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        rsi_value = indicators['RSI'].iloc[-1]
        rsi_signal = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
        st.metric("RSI (14)", f"{rsi_value:.2f}", rsi_signal)

    with col2:
        macd_value = indicators['MACD'].iloc[-1]
        macd_signal_value = indicators['MACD_Signal'].iloc[-1]
        macd_trend = "Bullish" if macd_value > macd_signal_value else "Bearish"
        st.metric("MACD", f"{macd_value:.4f}", macd_trend)

    with col3:
        bb_position = ((stock_data['Close'].iloc[-1] - indicators['BB_Lower'].iloc[-1]) / 
                       (indicators['BB_Upper'].iloc[-1] - indicators['BB_Lower'].iloc[-1])) * 100
        bb_signal = "Upper Band" if bb_position > 80 else "Lower Band" if bb_position < 20 else "Middle"
        st.metric("Bollinger Bands", f"{bb_position:.1f}%", bb_signal)

    with col4:
        sma_20 = indicators['SMA_20'].iloc[-1]
        price_vs_sma = "Above" if current_price > sma_20 else "Below"
        st.metric("Price vs SMA(20)", f"{((current_price/sma_20 - 1) * 100):+.2f}%", price_vs_sma)

    # Interactive Charts
    st.subheader("📈 Interactive Price Chart with Technical Indicators")

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Moving Averages', 'Volume', 'RSI & MACD'),
        row_width=[0.6, 0.2, 0.2]
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Moving averages
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=indicators['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=2)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=indicators['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    # Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=indicators['BB_Upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dash'),
            fill=None
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=indicators['BB_Lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)'
        ),
        row=1, col=1
    )

    # Volume
    fig.add_trace(
        go.Bar(
            x=stock_data.index,
            y=stock_data['Volume'],
            name='Volume',
            marker_color='lightblue'
        ),
        row=2, col=1
    )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=indicators['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ),
        row=3, col=1
    )

    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # MACD
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=indicators['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=2)
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=indicators['MACD_Signal'],
            mode='lines',
            name='MACD Signal',
            line=dict(color='red', width=2)
        ),
        row=3, col=1
    )

    # Update layout
    fig.update_layout(
        title=f"{stock_name} - Technical Analysis",
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )

    fig.update_yaxes(title_text=f"Price ({currency_symbol})", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI / MACD", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Recent Performance Table
    st.subheader("📅 Recent Performance")

    # Calculate returns
    returns_data = []
    periods = [1, 5, 10, 30]

    for period in periods:
        if len(stock_data) > period:
            old_price = stock_data['Close'].iloc[-(period+1)]
            new_price = stock_data['Close'].iloc[-1]
            return_pct = ((new_price - old_price) / old_price) * 100
            returns_data.append({
                'Period': f'{period}D',
                'Return (%)': f'{return_pct:+.2f}%',
                'Price Change': f'{currency_symbol}{new_price - old_price:+.2f}'
            })

    if returns_data:
        returns_df = pd.DataFrame(returns_data)
        st.dataframe(returns_df, hide_index=True)

    # Market Status
    st.subheader("🌍 Market Status")
    market_tz = get_market_timezone(selected_market)
    current_time = datetime.now(market_tz)
    st.info(f"Current {selected_market} Market Time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

with tab2:
    st.header("🏛️ Mutual Funds Screener")
    
    # Mutual Fund Category Selection
    st.sidebar.subheader("Mutual Fund Filters")
    
    # Category filter
    categories = ['All'] + list(mf_manager.MF_CATEGORIES.keys())
    selected_category = st.sidebar.selectbox("Fund Category", categories)
    
    # Get funds based on category
    if selected_category == 'All':
        available_funds = list(mf_manager.INDIAN_MUTUAL_FUNDS.keys())
    else:
        available_funds = mf_manager.get_top_funds_by_category(selected_category)
    
    # Fund selection
    selected_fund = st.selectbox("Select Mutual Fund", available_funds)
    
    # Time period for MF
    mf_period = st.selectbox("Time Period", list(time_periods.keys()), index=3, key="mf_period")
    
    # Fetch mutual fund data
    try:
        with st.spinner(f"Fetching data for {selected_fund}..."):
            mf_symbol = mf_manager.INDIAN_MUTUAL_FUNDS.get(selected_fund, '')
            mf_data = mf_manager.get_mutual_fund_data(mf_symbol, time_periods[mf_period])
            mf_info = mf_manager.get_mutual_fund_info(selected_fund)
            mf_returns = mf_manager.calculate_returns(mf_data)
        
        # Display fund information
        st.subheader(f"📊 {selected_fund}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current NAV", f"₹{mf_info['nav']}")
        
        with col2:
            st.metric("AUM", mf_info['aum'])
        
        with col3:
            st.metric("Expense Ratio", f"{mf_info['expense_ratio']}%")
        
        with col4:
            st.metric("Category", mf_info['category'])
        
        # Returns table
        st.subheader("📈 Fund Performance")
        
        returns_col1, returns_col2 = st.columns(2)
        
        with returns_col1:
            returns_df = pd.DataFrame([
                {"Period": "1 Month", "Return": f"{mf_returns.get('1M', 0)}%"},
                {"Period": "3 Months", "Return": f"{mf_returns.get('3M', 0)}%"},
                {"Period": "6 Months", "Return": f"{mf_returns.get('6M', 0)}%"}
            ])
            st.dataframe(returns_df, hide_index=True)
        
        with returns_col2:
            returns_df2 = pd.DataFrame([
                {"Period": "1 Year", "Return": f"{mf_returns.get('1Y', 0)}%"},
                {"Period": "2 Years", "Return": f"{mf_returns.get('2Y', 0)}%"},
                {"Period": "5 Years", "Return": f"{mf_returns.get('5Y', 0)}%"}
            ])
            st.dataframe(returns_df2, hide_index=True)
        
        # NAV Chart
        st.subheader("📊 NAV Performance Chart")
        
        fig_mf = go.Figure()
        fig_mf.add_trace(
            go.Scatter(
                x=mf_data.index,
                y=mf_data['Close'],
                mode='lines',
                name='NAV',
                line=dict(color='blue', width=2)
            )
        )
        
        fig_mf.update_layout(
            title=f"{selected_fund} - NAV Performance",
            xaxis_title="Date",
            yaxis_title="NAV (₹)",
            height=400
        )
        
        st.plotly_chart(fig_mf, use_container_width=True)
        
        # Fund Details
        st.subheader("🏛️ Fund Details")
        
        details_col1, details_col2 = st.columns(2)
        
        with details_col1:
            st.info(f"""
            **Fund House:** {mf_info['fund_house']}  
            **Fund Manager:** {mf_info['fund_manager']}  
            **Benchmark:** {mf_info['benchmark']}  
            **Launch Date:** {mf_info['launch_date']}
            """)
        
        with details_col2:
            st.info(f"""
            **Minimum SIP:** ₹{mf_info['minimum_sip']}  
            **Minimum Lumpsum:** ₹{mf_info['minimum_lumpsum']}  
            **Exit Load:** {mf_info['exit_load']}
            """)
    
    except Exception as e:
        st.error(f"Error fetching mutual fund data: {str(e)}")

with tab3:
    st.header("💰 SIP Calculator")
    
    st.markdown("Plan your Systematic Investment Plan (SIP) and see how your money can grow over time!")
    
    # SIP Calculator inputs
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_sip = st.number_input(
            "Monthly SIP Amount (₹)",
            min_value=500,
            max_value=1000000,
            value=5000,
            step=500
        )
        
        investment_years = st.slider(
            "Investment Period (Years)",
            min_value=1,
            max_value=30,
            value=10
        )
    
    with col2:
        expected_return = st.slider(
            "Expected Annual Return (%)",
            min_value=1.0,
            max_value=30.0,
            value=12.0,
            step=0.5
        )
        
        st.info(f"""
        **Investment Details:**  
        Monthly Amount: ₹{monthly_sip:,}  
        Investment Period: {investment_years} years  
        Expected Return: {expected_return}% per annum
        """)
    
    # Calculate SIP returns
    sip_results = sip_calc.calculate_sip_returns(monthly_sip, investment_years, expected_return)
    
    # Display results
    st.subheader("📊 SIP Calculation Results")
    
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.metric(
            "Total Investment",
            f"₹{sip_results['total_investment']:,.0f}",
            help="Total amount you will invest over the period"
        )
    
    with result_col2:
        st.metric(
            "Future Value",
            f"₹{sip_results['future_value']:,.0f}",
            help="Total value of your investment at maturity"
        )
    
    with result_col3:
        st.metric(
            "Wealth Gain",
            f"₹{sip_results['wealth_gain']:,.0f}",
            delta=f"{((sip_results['wealth_gain']/sip_results['total_investment'])*100):.1f}%",
            help="Profit earned on your investment"
        )
    
    # Year-wise projection
    st.subheader("📈 Year-wise Investment Growth")
    
    projection_df = sip_calc.generate_sip_projection(monthly_sip, investment_years, expected_return)
    
    # Create projection chart
    fig_sip = go.Figure()
    
    fig_sip.add_trace(
        go.Scatter(
            x=projection_df['Year'],
            y=projection_df['Total Investment'],
            mode='lines+markers',
            name='Total Investment',
            line=dict(color='orange', width=3),
            marker=dict(size=6)
        )
    )
    
    fig_sip.add_trace(
        go.Scatter(
            x=projection_df['Year'],
            y=projection_df['Future Value'],
            mode='lines+markers',
            name='Future Value',
            line=dict(color='green', width=3),
            marker=dict(size=6)
        )
    )
    
    fig_sip.update_layout(
        title="SIP Growth Projection",
        xaxis_title="Years",
        yaxis_title="Amount (₹)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_sip, use_container_width=True)
    
    # Show projection table
    st.subheader("📅 Detailed Year-wise Breakdown")
    
    # Format the projection dataframe for better display
    display_df = projection_df.copy()
    display_df['Total Investment'] = display_df['Total Investment'].apply(lambda x: f"₹{x:,.0f}")
    display_df['Future Value'] = display_df['Future Value'].apply(lambda x: f"₹{x:,.0f}")
    display_df['Wealth Gain'] = display_df['Wealth Gain'].apply(lambda x: f"₹{x:,.0f}")
    
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    # SIP Tips
    st.subheader("💡 SIP Investment Tips")
    
    st.success("""
    **Smart SIP Strategies:**
    - Start early to benefit from compound growth
    - Increase SIP amount annually (step-up SIP)
    - Stay invested for long term (10+ years)
    - Don't stop SIP during market volatility
    - Review and rebalance your portfolio annually
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    **Disclaimer:** This application provides analysis for educational purposes only. 
    Stock predictions and SIP calculations are based on technical indicators and assumed returns 
    but should not be considered as financial advice. Always do your own research and consult 
    with a financial advisor before making investment decisions.
    """
)