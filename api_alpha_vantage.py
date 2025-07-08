"""
VectorFin API with Alpha Vantage adapter.

This is a modified version of the main API file that uses Alpha Vantage
instead of NewsAPI for fetching news data.
"""

import os
import sys
import json
import uuid
import secrets
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Header, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import VectorFin components
from vectorfin.src.models.vectorfin import VectorFinSystem
from vectorfin.src.utils.config import Config
from vectorfin.src.utils.common import setup_logging
from vectorfin.src.prediction_interpretation.prediction import InterpretationSystem

# Import Alpha Vantage adapter
from vectorfin.src.data.alpha_vantage_adapter import fetch_news, setup_alpha_vantage_adapter

# Set up logging
logger = logging.getLogger(__name__)
setup_logging()

# Initialize Config
config = Config()

# Initialize the API
app = FastAPI(
    title="VectorFin API (Alpha Vantage)",
    description="Financial prediction API powered by VectorFin using Alpha Vantage for news data",
    version="0.2.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up API rate limiting
API_RATE_LIMITS = {
    "default": {"day": 100},  # Default: 100 requests per day
    "admin": {"day": 1000}    # Admin: 1000 requests per day
}

# User rate tracking
user_request_counts = {}

# Generate a secure admin API key if not provided in environment
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", secrets.token_urlsafe(32))
if "ADMIN_API_KEY" not in os.environ:
    logger.info(f"Generated admin API key: {ADMIN_API_KEY}")

# Initialize the VectorFin system
system = None
try:
    # Try to initialize system at startup
    logger.info("Initializing VectorFin system...")
    system = VectorFinSystem(config)
    logger.info("VectorFin system initialized successfully")
except Exception as e:
    logger.error(f"Error initializing VectorFin system: {str(e)}")
    logger.info("System will attempt to initialize on first API call")

# Initialize the interpretation system
interpreter = None
try:
    # Try to initialize interpreter at startup
    logger.info("Initializing interpretation system...")
    interpreter = InterpretationSystem(config)
    logger.info("Interpretation system initialized successfully")
except Exception as e:
    logger.error(f"Error initializing interpretation system: {str(e)}")
    logger.info("Interpreter will attempt to initialize on first API call")

# Set up Alpha Vantage adapter
setup_alpha_vantage_adapter()

# API Models
class PredictionRequest(BaseModel):
    """Request model for making predictions."""
    tickers: List[str] = Field(
        default=["AAPL", "MSFT", "GOOGL"],
        description="List of ticker symbols to predict"
    )
    prediction_horizon: int = Field(
        default=5,
        description="Number of days to predict ahead"
    )
    news_days: int = Field(
        default=10,
        description="Number of days of news to analyze"
    )
    include_interpretation: bool = Field(
        default=True,
        description="Whether to include LLM interpretation of results"
    )
    alpha_vantage_api_key: Optional[str] = Field(
        default=None,
        description="API key for Alpha Vantage (will override environment variable)"
    )
    llm_api_url: Optional[str] = Field(
        default=None,
        description="URL for LLM API (for interpretation)"
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        description="API key for LLM API"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="LLM model name to use for interpretation"
    )


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    request_id: str = Field(description="Unique ID for this request")
    timestamp: str = Field(description="Timestamp of the prediction")
    tickers: List[str] = Field(description="List of ticker symbols predicted")
    prediction_horizon: int = Field(description="Number of days predicted ahead")
    predictions: Dict[str, Dict[str, Any]] = Field(description="Prediction results")
    interpretation: Optional[str] = Field(description="LLM interpretation of results")
    config_summary: Dict[str, Any] = Field(description="Summary of configuration used")


class ConfigResponse(BaseModel):
    """Response model for configuration."""
    config: Dict[str, Any] = Field(description="Current configuration")


class StatusResponse(BaseModel):
    """Response model for API status."""
    status: str = Field(description="API status")
    version: str = Field(description="API version")
    system_initialized: bool = Field(description="Whether the VectorFin system is initialized")
    interpreter_initialized: bool = Field(description="Whether the interpretation system is initialized")


# Helper Functions
def get_api_key(x_api_key: str = Header(None)) -> str:
    """Validate API key from header."""
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key required")
    
    # Admin key has unlimited access
    if x_api_key == ADMIN_API_KEY:
        return "admin"
    
    # For demo purposes, allow any key but apply rate limiting
    return x_api_key


def check_rate_limit(api_key: str = Depends(get_api_key), request: Request = None) -> str:
    """Check if the request is within rate limits."""
    # Get user's rate limit
    user_type = "admin" if api_key == "admin" else "default"
    limits = API_RATE_LIMITS[user_type]
    
    # Initialize user in tracking if needed
    if api_key not in user_request_counts:
        user_request_counts[api_key] = {"day": 0, "day_start": datetime.now()}
    
    # Reset counters if day has changed
    if (datetime.now() - user_request_counts[api_key]["day_start"]).days > 0:
        user_request_counts[api_key]["day"] = 0
        user_request_counts[api_key]["day_start"] = datetime.now()
    
    # Check daily limit
    if user_request_counts[api_key]["day"] >= limits["day"]:
        raise HTTPException(status_code=429, detail="Daily rate limit exceeded")
    
    # Increment counter
    user_request_counts[api_key]["day"] += 1
    
    return api_key


def fetch_market_data(tickers: List[str], days: int = 30) -> pd.DataFrame:
    """Fetch market data for the given tickers."""
    logger.info(f"Fetching market data for {tickers}...")
    
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # For this example, we'll generate synthetic data
    # In a real app, you'd use a market data API like Alpha Vantage's TIME_SERIES_DAILY
    logger.info("Using synthetic market data")
    data = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current_date <= end:
        if current_date.weekday() < 5:  # Only weekdays
            for ticker in tickers:
                # Generate random but somewhat realistic price data
                base_price = 100 + hash(ticker) % 400  # Different base price for each ticker
                price = base_price + (hash(str(current_date) + ticker) % 20) - 10
                data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'open': price - 1,
                    'high': price + 2,
                    'low': price - 2,
                    'close': price,
                    'volume': 1000000 + hash(str(current_date) + ticker) % 5000000
                })
        current_date += timedelta(days=1)
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df


def fetch_recent_news(tickers: List[str], days: int = 10) -> pd.DataFrame:
    """Fetch recent news for the given tickers using Alpha Vantage."""
    logger.info(f"Fetching news for {tickers} over the past {days} days...")
    try:
        return fetch_news(tickers, days)
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['date', 'headline', 'ticker', 'source'])


def prepare_data_for_prediction(news_data: pd.DataFrame, market_data: pd.DataFrame):
    """Prepare data for prediction by extracting features and aligning timestamps."""
    # For this API version, we'll keep this simple
    # In a real app, you'd do more sophisticated preprocessing
    return news_data, market_data


# API Endpoints
@app.get("/", response_model=StatusResponse)
def get_status(api_key: str = Depends(check_rate_limit)):
    """Get API status."""
    return {
        "status": "running",
        "version": "0.2.0",
        "system_initialized": system is not None,
        "interpreter_initialized": interpreter is not None
    }


@app.get("/config", response_model=ConfigResponse)
def get_config(api_key: str = Depends(check_rate_limit)):
    """Get current configuration (safe version)."""
    # Get config but redact sensitive values
    conf = config.get_all()
    redacted_config = redact_config(conf)
    return {"config": redacted_config}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, api_key: str = Depends(check_rate_limit)):
    """Make predictions for the given tickers."""
    logger.info(f"Prediction request received for {request.tickers}")
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    
    # Store original environment values
    old_alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if request.alpha_vantage_api_key:
        os.environ["ALPHA_VANTAGE_API_KEY"] = request.alpha_vantage_api_key
        # Also set NEWS_API_KEY for compatibility
        os.environ["NEWS_API_KEY"] = request.alpha_vantage_api_key
    
    old_llm_api_url = os.environ.get("LLM_API_URL")
    if request.llm_api_url:
        os.environ["LLM_API_URL"] = request.llm_api_url
    
    old_llm_api_key = os.environ.get("LLM_API_KEY")
    if request.llm_api_key:
        os.environ["LLM_API_KEY"] = request.llm_api_key
    
    old_model_name = os.environ.get("INTERPRETATION_MODEL")
    if request.model_name:
        os.environ["INTERPRETATION_MODEL"] = request.model_name
    
    global system, interpreter
    try:
        # Initialize system if needed
        if system is None:
            logger.info("Initializing VectorFin system...")
            system = VectorFinSystem(config)
        
        # Initialize interpreter if needed
        if interpreter is None and request.include_interpretation:
            logger.info("Initializing interpretation system...")
            interpreter = InterpretationSystem(config)
        
        # Fetch data
        news_data = fetch_recent_news(request.tickers, days=request.news_days)
        market_data = fetch_market_data(request.tickers, days=30)
        
        # Prepare data
        processed_news, aligned_market_data = prepare_data_for_prediction(news_data, market_data)
        
        # Make predictions
        predictions = {}
        
        for ticker in request.tickers:
            # Filter data for this ticker
            ticker_news = processed_news[processed_news['ticker'] == ticker] if not processed_news.empty else pd.DataFrame()
            ticker_market = aligned_market_data[aligned_market_data['ticker'] == ticker]
            
            # Skip if we don't have enough data
            if ticker_market.empty:
                logger.warning(f"Skipping {ticker}: Insufficient market data")
                continue
            
            try:
                result = system.predict(
                    ticker_news, 
                    ticker_market, 
                    prediction_horizon=request.prediction_horizon
                )
                
                direction = "UP" if result.get('direction', 0) > 0.5 else "DOWN"
                confidence = result.get('confidence', 0.5)
                
                predictions[ticker] = {
                    "direction": direction,
                    "confidence": confidence
                }
                
                logger.info(f"Prediction for {ticker}: {direction} (Confidence: {confidence:.2f})")
            except Exception as e:
                logger.error(f"Error predicting for {ticker}: {str(e)}")
                predictions[ticker] = {
                    "error": str(e)
                }
        
        # Generate interpretation if requested
        interpretation = None
        if request.include_interpretation and interpreter is not None and predictions:
            try:
                # Build the prompt
                tickers_str = ", ".join(request.tickers)
                prediction_str = "\n".join([f"{t}: {p['direction']} (Confidence: {p['confidence']:.2f})" 
                                          for t, p in predictions.items() if 'direction' in p])
                
                news_str = "No recent news available."
                if not news_data.empty:
                    recent_news = news_data.sort_values('date', ascending=False).head(5)
                    news_str = "\n".join([f"[{row['date'].strftime('%Y-%m-%d')}] {row['headline']} (Ticker: {row['ticker']})" 
                                      for _, row in recent_news.iterrows()])
                
                prompt = f"""
                You are a financial analyst assistant. I need you to interpret the following stock prediction results.
                
                Prediction horizon: {request.prediction_horizon} days
                
                Predicted price movements:
                {prediction_str}
                
                Recent relevant news:
                {news_str}
                
                Based on this information, please provide:
                1. A brief overall market outlook for these stocks
                2. Specific insights for each ticker, connecting the predictions to recent news
                3. Potential risks or factors that could affect these predictions
                
                Limit your response to 300-400 words.
                """
                
                interpretation = interpreter.generate_interpretation(prompt)
                logger.info("Generated interpretation successfully")
            except Exception as e:
                logger.error(f"Error generating interpretation: {str(e)}")
                interpretation = f"Error generating interpretation: {str(e)}"
        
        # Restore environment variables
        if request.alpha_vantage_api_key:
            if old_alpha_vantage_api_key:
                os.environ["ALPHA_VANTAGE_API_KEY"] = old_alpha_vantage_api_key
                os.environ["NEWS_API_KEY"] = old_alpha_vantage_api_key
            else:
                os.environ.pop("ALPHA_VANTAGE_API_KEY", None)
                os.environ.pop("NEWS_API_KEY", None)
        
        if request.llm_api_url:
            if old_llm_api_url:
                os.environ["LLM_API_URL"] = old_llm_api_url
            else:
                os.environ.pop("LLM_API_URL", None)
        
        if request.llm_api_key:
            if old_llm_api_key:
                os.environ["LLM_API_KEY"] = old_llm_api_key
            else:
                os.environ.pop("LLM_API_KEY", None)
        
        if request.model_name:
            if old_model_name:
                os.environ["INTERPRETATION_MODEL"] = old_model_name
            else:
                os.environ.pop("INTERPRETATION_MODEL", None)
        
        # Get safe config for response
        conf_summary = redact_config(config.get_all())
        
        # Return the results
        return {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "tickers": request.tickers,
            "prediction_horizon": request.prediction_horizon,
            "predictions": predictions,
            "interpretation": interpretation,
            "config_summary": {
                "using_alpha_vantage": True,
                "news_days": request.news_days,
                "model_info": conf_summary.get("models_dir", "default")
            }
        }
    
    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def redact_config(conf: Dict) -> Dict:
    """Redact sensitive information from config."""
    redacted = conf.copy()
    
    # Redact API keys
    if "alpha_vantage_api" in redacted:
        redacted["alpha_vantage_api"]["api_key"] = "***REDACTED***"
    
    if "news_api" in redacted and "api_key" in redacted["news_api"]:
        redacted["news_api"]["api_key"] = "***REDACTED***"
    
    if "llm" in redacted and "api_key" in redacted["llm"]:
        redacted["llm"]["api_key"] = "***REDACTED***"
    
    return redacted


if __name__ == "__main__":
    import uvicorn
    # If run directly, start the server
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_alpha_vantage:app", host="0.0.0.0", port=port, reload=False)
