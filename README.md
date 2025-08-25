# Multi-Market Stock Screener

## Overview

A comprehensive stock analysis application built with Streamlit that provides technical analysis and AI-powered predictions for both Indian and US stock markets. The system combines real-time market data, technical indicators, and machine learning insights to help users make informed trading decisions.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web interface and interactive components
- **Visualization**: Plotly for advanced charting and technical analysis plots
- **State Management**: Session state for maintaining user selections across market switches
- **Layout**: Wide layout with expandable sidebar for market/stock selection

### Backend Architecture
- **Modular Design**: Utility-based architecture with separated concerns:
  - `StockDataManager`: Handles data retrieval and caching
  - `TechnicalIndicators`: Calculates various technical analysis metrics
  - `AIPredictor`: Provides ML-based trading signals and predictions
  - `market_data`: Static configuration for supported stocks

### Data Processing Pipeline
- **Real-time Data**: Yahoo Finance API through yfinance library for live market data
- **Caching Strategy**: 5-minute TTL caching using Streamlit's built-in caching decorators
- **Technical Analysis**: Comprehensive indicator calculations including RSI, MACD, Bollinger Bands, Stochastic Oscillator, and moving averages
- **Time Series Processing**: Pandas for data manipulation with proper datetime indexing

### AI Integration
- **LLM Integration**: Hugging Face Inference API for natural language processing
- **Backup Models**: Multiple fallback models (DialoGPT, BlenderBot) for reliability
- **Signal Generation**: Technical data preparation for AI-powered trading signal analysis

### Multi-Market Support
- **Indian Market**: NSE-listed stocks with .NS suffix notation
- **US Market**: NASDAQ/NYSE stocks with standard symbols
- **Timezone Handling**: pytz for proper market time zone management
- **Market-specific Logic**: Separate stock lists and market hours consideration

## External Dependencies

### Data Providers
- **Yahoo Finance**: Primary data source via yfinance library for real-time and historical stock data
- **Supported Markets**: Indian NSE and US NASDAQ/NYSE exchanges

### AI Services
- **Hugging Face**: Inference API for AI-powered stock analysis
- **Models Used**: Microsoft DialoGPT and Facebook BlenderBot for natural language processing

### Python Libraries
- **Core Framework**: Streamlit for web application framework
- **Data Processing**: pandas and numpy for data manipulation and analysis
- **Visualization**: plotly for interactive charts and technical analysis plots
- **Market Data**: yfinance for Yahoo Finance API integration
- **Time Handling**: datetime and pytz for timezone-aware operations
- **HTTP Requests**: requests library for API communications

### Configuration Dependencies
- **Environment Variables**: Hugging Face API key for AI model access
- **Stock Universe**: Predefined lists of popular stocks for both Indian and US markets
- **Caching**: Built-in Streamlit caching mechanisms for performance optimization
