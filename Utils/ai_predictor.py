import pandas as pd
import numpy as np
import streamlit as st
import requests
import json
import os
from typing import Tuple

class AIPredictor:
    """AI-powered stock signal prediction using free LLM models"""
    
    def __init__(self):
        # Using Hugging Face Inference API with suitable models for financial analysis
        self.primary_model = "microsoft/DialoGPT-medium"
        self.api_url = f"https://api-inference.huggingface.co/models/{self.primary_model}"
        self.headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}"}
        self.backup_models = [
            "microsoft/DialoGPT-small",
            "facebook/blenderbot-400M-distill",
            "gpt2",
            "distilgpt2"
        ]
        
        # Check if we have a valid API token
        self.has_valid_token = bool(os.getenv('HF_TOKEN', '').strip())
        if self.has_valid_token:
            try:
                # Display token status in sidebar without showing the actual token
                st.sidebar.success("🤖 AI Predictions: ENABLED")
            except:
                pass  # Ignore if sidebar is not available during testing
        else:
            try:
                st.sidebar.info("🤖 AI Predictions: Rule-based analysis")
            except:
                pass  # Ignore if sidebar is not available during testing
    
    def prepare_technical_data(self, stock_data: pd.DataFrame, indicators: pd.DataFrame) -> dict:
        """Prepare technical analysis data for AI processing"""
        try:
            latest_data = stock_data.iloc[-1]
            latest_indicators = indicators.iloc[-1]
            
            # Calculate price changes
            price_change_1d = ((latest_data['Close'] - stock_data['Close'].iloc[-2]) / stock_data['Close'].iloc[-2]) * 100
            price_change_5d = ((latest_data['Close'] - stock_data['Close'].iloc[-6]) / stock_data['Close'].iloc[-6]) * 100 if len(stock_data) > 5 else 0
            
            # Volume analysis
            avg_volume_series = stock_data['Volume'].rolling(20).mean()
            avg_volume = avg_volume_series.iloc[-1] if len(avg_volume_series) > 0 else 1
            volume_ratio = latest_data['Volume'] / avg_volume if avg_volume > 0 else 1
            
            return {
                'current_price': latest_data['Close'],
                'price_change_1d': price_change_1d,
                'price_change_5d': price_change_5d,
                'volume_ratio': volume_ratio,
                'rsi': latest_indicators.get('RSI', 50),
                'macd': latest_indicators.get('MACD', 0),
                'macd_signal': latest_indicators.get('MACD_Signal', 0),
                'bb_position': self.calculate_bb_position(latest_data['Close'], latest_indicators),
                'sma_20_position': ((latest_data['Close'] - latest_indicators.get('SMA_20', latest_data['Close'])) / latest_indicators.get('SMA_20', latest_data['Close'])) * 100,
                'sma_50_position': ((latest_data['Close'] - latest_indicators.get('SMA_50', latest_data['Close'])) / latest_indicators.get('SMA_50', latest_data['Close'])) * 100,
                'williams_r': latest_indicators.get('Williams_R', -50),
                'stoch_k': latest_indicators.get('Stoch_K', 50)
            }
        except Exception as e:
            st.error(f"Error preparing technical data: {str(e)}")
            return {}
    
    def calculate_bb_position(self, current_price: float, indicators: pd.Series) -> float:
        """Calculate position within Bollinger Bands"""
        try:
            bb_upper = indicators.get('BB_Upper', current_price)
            bb_lower = indicators.get('BB_Lower', current_price)
            if bb_upper is not None and bb_lower is not None and bb_upper != bb_lower:
                return ((current_price - bb_lower) / (bb_upper - bb_lower)) * 100
            return 50.0
        except:
            return 50.0
    
    def generate_rule_based_prediction(self, tech_data: dict) -> Tuple[str, float, str]:
        """Generate prediction using rule-based approach as fallback"""
        
        buy_signals = 0
        sell_signals = 0
        signal_explanations = []
        
        # RSI Analysis
        if tech_data.get('rsi', 50) < 30:
            buy_signals += 2
            signal_explanations.append("RSI indicates oversold conditions (strong buy signal)")
        elif tech_data.get('rsi', 50) > 70:
            sell_signals += 2
            signal_explanations.append("RSI indicates overbought conditions (strong sell signal)")
        elif 30 <= tech_data.get('rsi', 50) <= 45:
            buy_signals += 1
            signal_explanations.append("RSI shows potential buying opportunity")
        elif 55 <= tech_data.get('rsi', 50) <= 70:
            sell_signals += 1
            signal_explanations.append("RSI suggests taking profits")
        
        # MACD Analysis
        macd = tech_data.get('macd', 0)
        macd_signal = tech_data.get('macd_signal', 0)
        if macd > macd_signal and macd > 0:
            buy_signals += 2
            signal_explanations.append("MACD shows strong bullish momentum")
        elif macd > macd_signal and macd <= 0:
            buy_signals += 1
            signal_explanations.append("MACD showing bullish crossover")
        elif macd < macd_signal and macd < 0:
            sell_signals += 2
            signal_explanations.append("MACD shows strong bearish momentum")
        elif macd < macd_signal and macd >= 0:
            sell_signals += 1
            signal_explanations.append("MACD showing bearish crossover")
        
        # Moving Average Analysis
        sma_20_pos = tech_data.get('sma_20_position', 0)
        sma_50_pos = tech_data.get('sma_50_position', 0)
        
        if sma_20_pos > 2 and sma_50_pos > 2:
            buy_signals += 1
            signal_explanations.append("Price above key moving averages (bullish trend)")
        elif sma_20_pos < -2 and sma_50_pos < -2:
            sell_signals += 1
            signal_explanations.append("Price below key moving averages (bearish trend)")
        
        # Bollinger Bands Analysis
        bb_pos = tech_data.get('bb_position', 50)
        if bb_pos < 20:
            buy_signals += 1
            signal_explanations.append("Price near lower Bollinger Band (potential bounce)")
        elif bb_pos > 80:
            sell_signals += 1
            signal_explanations.append("Price near upper Bollinger Band (potential pullback)")
        
        # Williams %R Analysis
        williams_r = tech_data.get('williams_r', -50)
        if williams_r < -80:
            buy_signals += 1
            signal_explanations.append("Williams %R indicates oversold conditions")
        elif williams_r > -20:
            sell_signals += 1
            signal_explanations.append("Williams %R indicates overbought conditions")
        
        # Volume Analysis
        volume_ratio = tech_data.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            if buy_signals > sell_signals:
                buy_signals += 1
                signal_explanations.append("High volume supports bullish sentiment")
            elif sell_signals > buy_signals:
                sell_signals += 1
                signal_explanations.append("High volume supports bearish sentiment")
        
        # Price momentum
        price_change_1d = tech_data.get('price_change_1d', 0)
        price_change_5d = tech_data.get('price_change_5d', 0)
        
        if price_change_1d > 2 and price_change_5d > 5:
            if buy_signals > sell_signals:
                buy_signals += 1
                signal_explanations.append("Strong positive momentum supports uptrend")
        elif price_change_1d < -2 and price_change_5d < -5:
            if sell_signals > buy_signals:
                sell_signals += 1
                signal_explanations.append("Strong negative momentum supports downtrend")
        
        # Determine final signal
        total_signals = buy_signals + sell_signals
        if total_signals == 0:
            signal = "HOLD"
            confidence = 50.0
        else:
            if buy_signals > sell_signals:
                signal = "BUY"
                confidence = min(95, 50 + (buy_signals - sell_signals) * 10)
            elif sell_signals > buy_signals:
                signal = "SELL"
                confidence = min(95, 50 + (sell_signals - buy_signals) * 10)
            else:
                signal = "HOLD"
                confidence = 50.0
        
        explanation = f"""
        **Technical Analysis Summary:**
        
        **Buy Signals:** {buy_signals} | **Sell Signals:** {sell_signals}
        
        **Key Factors:**
        """ + "\n".join([f"• {exp}" for exp in signal_explanations[:5]]) + f"""
        
        **Current Metrics:**
        • RSI: {tech_data.get('rsi', 0):.1f}
        • Price vs SMA(20): {tech_data.get('sma_20_position', 0):+.1f}%
        • Bollinger Band Position: {tech_data.get('bb_position', 50):.1f}%
        • Volume Ratio: {tech_data.get('volume_ratio', 1):.1f}x
        """
        
        return signal, confidence, explanation
    
    def query_huggingface_model(self, payload: dict, model_url: str | None = None) -> str:
        """Query Hugging Face model API"""
        try:
            url = model_url or self.api_url
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '')
                elif isinstance(result, dict):
                    return result.get('generated_text', '')
            
            return ""
            
        except Exception as e:
            st.warning(f"AI model query failed: {str(e)}")
            return ""
    
    def predict_signal(self, stock_data: pd.DataFrame, indicators: pd.DataFrame) -> Tuple[str, float, str]:
        """
        Predict buy/sell/hold signal using AI analysis
        
        Args:
            stock_data: Historical stock data
            indicators: Technical indicators
            
        Returns:
            Tuple of (signal, confidence, explanation)
        """
        
        # Prepare technical data
        tech_data = self.prepare_technical_data(stock_data, indicators)
        
        if not tech_data:
            return "HOLD", 50.0, "Unable to analyze - insufficient data"
        
        # Try AI prediction first (if API key is available)
        ai_prediction = None
        if self.has_valid_token:
            try:
                # Create enhanced prompt for AI analysis
                prompt = f"""Stock Technical Analysis:
RSI: {tech_data.get('rsi', 50):.1f} (Oversold<30, Overbought>70)
MACD: {tech_data.get('macd', 0):.4f} vs Signal: {tech_data.get('macd_signal', 0):.4f}
Price vs 20-day average: {tech_data.get('sma_20_position', 0):+.1f}%
Bollinger position: {tech_data.get('bb_position', 50):.1f}% (Low<20%, High>80%)
Volume: {tech_data.get('volume_ratio', 1):.1f}x average
Price change: {tech_data.get('price_change_1d', 0):+.1f}% today

Based on technical analysis, this stock shows"""
                
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_length": 200,
                        "temperature": 0.8,
                        "return_full_text": False,
                        "do_sample": True,
                        "top_p": 0.9
                    }
                }
                
                ai_response = self.query_huggingface_model(payload)
                
                # Try backup models if primary fails
                if not ai_response:
                    for backup_model in self.backup_models[:2]:
                        try:
                            backup_url = f"https://api-inference.huggingface.co/models/{backup_model}"
                            ai_response = self.query_huggingface_model(payload, backup_url)
                            if ai_response:
                                break
                        except:
                            continue
                
                if ai_response:
                    # Parse AI response and provide more detailed analysis
                    response_upper = ai_response.upper()
                    response_clean = ai_response.strip()
                    
                    # Determine signal based on keywords and sentiment
                    buy_keywords = ['BUY', 'BULLISH', 'POSITIVE', 'UPWARD', 'STRONG', 'GOOD', 'OVERSOLD']
                    sell_keywords = ['SELL', 'BEARISH', 'NEGATIVE', 'DOWNWARD', 'WEAK', 'POOR', 'OVERBOUGHT']
                    
                    buy_score = sum(1 for keyword in buy_keywords if keyword in response_upper)
                    sell_score = sum(1 for keyword in sell_keywords if keyword in response_upper)
                    
                    if buy_score > sell_score and buy_score > 0:
                        confidence = min(85.0, 60.0 + buy_score * 8)
                        ai_prediction = ("BUY", confidence, f"🤖 **AI Market Analysis:**\n\n{response_clean}\n\n*AI detected {buy_score} bullish signals in technical data.*")
                    elif sell_score > buy_score and sell_score > 0:
                        confidence = min(85.0, 60.0 + sell_score * 8)
                        ai_prediction = ("SELL", confidence, f"🤖 **AI Market Analysis:**\n\n{response_clean}\n\n*AI detected {sell_score} bearish signals in technical data.*")
                    else:
                        ai_prediction = ("HOLD", 65.0, f"🤖 **AI Market Analysis:**\n\n{response_clean}\n\n*AI suggests cautious approach with mixed signals.*")
                        
            except Exception as e:
                st.warning(f"AI prediction failed, using rule-based analysis: {str(e)}")
        
        # Use rule-based prediction as fallback or primary method
        rule_signal, rule_confidence, rule_explanation = self.generate_rule_based_prediction(tech_data)
        
        # If AI prediction is available, combine with rule-based
        if ai_prediction:
            ai_signal, ai_confidence, ai_explanation = ai_prediction
            
            # If both agree, increase confidence
            if ai_signal == rule_signal:
                final_confidence = min(95, (ai_confidence + rule_confidence) / 2 + 10)
            else:
                # If they disagree, use rule-based with moderate confidence
                final_confidence = rule_confidence * 0.8
            
            combined_explanation = f"""
            **🤖 AI Analysis:**
            {ai_explanation.replace('AI Analysis: ', '')}
            
            ---
            
            {rule_explanation}
            """
            
            return rule_signal, final_confidence, combined_explanation
        
        else:
            return rule_signal, rule_confidence, rule_explanation
