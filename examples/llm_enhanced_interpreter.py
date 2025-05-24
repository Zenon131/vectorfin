"""
LLM-Enhanced Financial Prediction Interpreter

This script demonstrates how to combine VectorFin predictions with LLM analysis
to provide more insightful and context-aware financial predictions.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import requests
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path so we can import the vectorfin package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import VectorFin components
from vectorfin.src.models.vectorfin import VectorFinSystem


class LLMFinancialInterpreter:
    """
    A class that enhances VectorFin predictions with LLM interpretation.
    """
    
    def __init__(self, model_path="./trained_models", llm_provider="openai", api_key=None):
        """
        Initialize the interpreter with a VectorFin model and LLM settings.
        
        Args:
            model_path: Path to the trained VectorFin model
            llm_provider: LLM provider to use ('openai', 'anthropic', etc.)
            api_key: API key for the LLM provider (if None, will look for environment variable)
        """
        self.model_path = model_path
        self.llm_provider = llm_provider
        
        # Set up API key
        if api_key:
            self.api_key = api_key
        else:
            # Try to get from environment
            if llm_provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY")
            elif llm_provider == "anthropic":
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        # Load VectorFin model
        print(f"Loading VectorFin model from {model_path}...")
        self.model = VectorFinSystem.load_models(
            VectorFinSystem,  # Pass the class as first parameter
            directory=model_path,
            vector_dim=128,
            sentiment_dim=16,
            fusion_dim=128,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
    def interpret_prediction(self, prediction_results, market_context, news_context, user_question=None):
        """
        Use an LLM to interpret VectorFin prediction results.
        
        Args:
            prediction_results: Results from VectorFin prediction
            market_context: Summary of market data used for prediction
            news_context: Summary of news data used for prediction
            user_question: Optional specific question from the user
            
        Returns:
            Interpretation text from the LLM
        """
        # Construct the prompt
        system_prompt = (
            "You are a financial analyst assistant that specializes in interpreting algorithmic "
            "trading predictions in a balanced, informative way. You provide nuanced analysis "
            "based on model predictions, market data, and news sentiment. You clearly identify "
            "the level of certainty, potential factors driving predictions, and appropriate "
            "caveats. You avoid hyperbole and never make definitive claims about future market "
            "movements."
        )
        
        base_prompt = f"""
        Based on algorithmic analysis of financial data and news, I need you to interpret the following prediction:
        
        PREDICTION RESULTS:
        {json.dumps(prediction_results, indent=2)}
        
        MARKET CONTEXT:
        {market_context}
        
        NEWS CONTEXT:
        {news_context}
        
        Please provide:
        1. A concise interpretation of what this prediction suggests
        2. Key factors that might be influencing this prediction based on the provided context
        3. The level of confidence/uncertainty in this prediction
        4. Potential scenarios that could play out
        5. What investors might consider when evaluating this information
        """
        
        # Add user question if provided
        if user_question:
            base_prompt += f"\n\nADDITIONAL USER QUESTION: {user_question}"
            
        # Call appropriate LLM API based on provider
        if self.llm_provider == "openai":
            return self._call_openai_api(system_prompt, base_prompt)
        elif self.llm_provider == "anthropic":
            return self._call_anthropic_api(system_prompt, base_prompt)
        else:
            # Default mock response for demonstration
            return self._mock_llm_response(system_prompt, base_prompt)
    
    def _call_openai_api(self, system_prompt, user_prompt):
        """Call the OpenAI API to get an interpretation."""
        if not self.api_key:
            return "Error: OpenAI API key not provided. Please set OPENAI_API_KEY environment variable or pass api_key to constructor."
            
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": "gpt-4-turbo", 
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3
                }
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error: API returned status code {response.status_code}: {response.text}"
                
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"
    
    def _call_anthropic_api(self, system_prompt, user_prompt):
        """Call the Anthropic API to get an interpretation."""
        if not self.api_key:
            return "Error: Anthropic API key not provided. Please set ANTHROPIC_API_KEY environment variable or pass api_key to constructor."
            
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-3-opus-20240229",
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.3
                }
            )
            
            if response.status_code == 200:
                return response.json()["content"][0]["text"]
            else:
                return f"Error: API returned status code {response.status_code}: {response.text}"
                
        except Exception as e:
            return f"Error calling Anthropic API: {str(e)}"
    
    def _mock_llm_response(self, system_prompt, user_prompt):
        """Generate a mock LLM response for demonstration purposes."""
        # In a real implementation, you would call an LLM API
        # This is just a placeholder for demonstration
        return """
        ## Financial Prediction Analysis
        
        ### Interpretation
        Based on the model's prediction, there appears to be a moderate bullish signal for the near term (next 5 trading days). The direction probability of 0.62 suggests a 62% chance of upward price movement, while the magnitude prediction of 0.023 indicates an expected price increase of approximately 2.3%. The volatility measure of 0.015 suggests relatively low expected price fluctuations during this period.
        
        ### Key Influencing Factors
        1. Recent positive news sentiment regarding product announcements and expansion plans
        2. Technical indicators showing upward momentum in recent trading sessions
        3. Overall market context appears supportive, with major indices showing stability
        4. The combination of news and price action suggests institutional interest remains positive
        
        ### Confidence Assessment
        The prediction carries moderate confidence. The 62% directional probability indicates meaningful but not overwhelming conviction. The relatively low volatility prediction suggests the model sees a somewhat stable path rather than erratic movement, which typically corresponds to higher prediction reliability.
        
        ### Potential Scenarios
        - **Base case (most likely)**: Gradual appreciation of approximately 2-2.5% over the 5-day horizon
        - **Bull case**: If positive catalyst emerges, could see gains exceeding 4%
        - **Bear case**: If market sentiment shifts or negative news emerges, could see flat or slightly negative performance
        
        ### Investment Considerations
        1. This prediction may be suitable for short-term trading strategies rather than long-term investment decisions
        2. The moderate confidence level suggests position sizing should be conservative
        3. Setting appropriate stop-loss levels is advisable given the inherent uncertainty in short-term predictions
        4. Consider the broader market environment and sector trends before making decisions
        5. This algorithmic prediction should be one of several inputs in your investment process, not the sole determining factor
        
        Remember that all market predictions carry inherent uncertainty, and past performance is not indicative of future results.
        """


def demonstrate_llm_interpretation():
    """Demonstrate the LLM interpretation capabilities."""
    # Initialize the interpreter
    # Note: In a real implementation, you'd provide a valid API key
    interpreter = LLMFinancialInterpreter(
        model_path="./trained_models", 
        llm_provider="openai"
    )
    
    # Example prediction results (would come from your model in real use)
    prediction_results = {
        "date": "2023-05-15",
        "prediction_horizon": 5,
        "predictions": {
            "direction": 0.62,  # Probability of upward movement
            "magnitude": 0.023,  # Expected percentage change
            "volatility": 0.015  # Expected volatility
        },
        "confidence_score": 0.78
    }
    
    # Example market context
    market_context = """
    - AAPL: Last closing price $187.43, up 1.2% over the past 5 days, with average volume 15% above the 30-day mean.
    - AAPL technical indicators: RSI at 58, MACD showing positive momentum, 50-day MA recently crossed above 200-day MA.
    - Broader market: S&P 500 up 0.8% over the past week, tech sector outperforming with 1.5% gains.
    - VIX at 16.4, indicating relatively low market volatility expectations.
    """
    
    # Example news context
    news_context = """
    Recent news highlights:
    - May 14, 2023: "Apple announces expansion of AI capabilities across product lineup" (Sentiment: Positive)
    - May 13, 2023: "iPhone sales in emerging markets exceed analyst expectations" (Sentiment: Positive)
    - May 12, 2023: "Supply chain concerns may impact Apple's production capacity" (Sentiment: Negative)
    - May 10, 2023: "Tech sector faces increased regulatory scrutiny in EU" (Sentiment: Negative)
    - May 09, 2023: "Apple developer conference announced for early June" (Sentiment: Neutral)
    
    Overall news sentiment: Moderately positive (60% positive, 20% neutral, 20% negative)
    """
    
    # Example user questions
    user_questions = [
        "How should I interpret this prediction for a short-term trading strategy?",
        "What potential risks should I be aware of with this prediction?",
        "How might this prediction change if the Federal Reserve increases interest rates?"
    ]
    
    # Get basic interpretation
    basic_interpretation = interpreter.interpret_prediction(
        prediction_results,
        market_context,
        news_context
    )
    
    print("=== BASIC INTERPRETATION ===")
    print(basic_interpretation)
    print("\n" + "="*50 + "\n")
    
    # Get interpretation with user question
    for question in user_questions:
        print(f"=== USER QUESTION: {question} ===")
        interpretation = interpreter.interpret_prediction(
            prediction_results,
            market_context,
            news_context,
            user_question=question
        )
        print(interpretation)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    demonstrate_llm_interpretation()
