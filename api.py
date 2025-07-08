"""
VectorFin API Server

This script provides a REST API for the VectorFin financial prediction system.
It allows users to make predictions, interpret them, and configure the system.
"""

import os
import sys
import json
import logging
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Add the parent directory to the path for importing the vectorfin package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Header, Query, status, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import uvicorn

# Import VectorFin components
from vectorfin.src.models.vectorfin import VectorFinSystem
from vectorfin.src.data.data_loader import MarketData, FinancialTextData
from vectorfin.src.utils.config import config

# Import the prediction functionality
from examples.interact_with_model import (
    load_trained_model,
    fetch_recent_news,
    fetch_market_data,
    prepare_data_for_prediction,
    query_llm_for_interpretation,
    make_prediction,
    generate_fallback_interpretation
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vectorfin-api")

# Create FastAPI app
app = FastAPI(
    title="VectorFin API",
    description="API for financial market predictions and interpretations",
    version="1.0.0",
    docs_url=None,  # We will create a custom docs endpoint with authentication
    redoc_url=None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Store for API keys (in a real production system, this would be in a database)
# For demo purposes, we'll use an in-memory dictionary with a default admin key
API_KEYS = {
    "admin": {
        "key": os.getenv("ADMIN_API_KEY", secrets.token_urlsafe(32)),
        "is_admin": True,
        "rate_limit": 1000,  # Requests per day
        "requests_today": 0,
        "last_reset": datetime.now().date()
    }
}

# Global model cache
MODEL_CACHE = {
    "system": None,
    "loaded_at": None,
    "models_dir": None
}

# Request rate limiting and API key validation
def get_api_key(api_key: str = Header(None, alias=API_KEY_NAME)) -> Dict:
    """
    Validate the API key and check rate limits.
    
    Args:
        api_key: The API key from the request header
        
    Returns:
        The API key info if valid
        
    Raises:
        HTTPException: If the API key is invalid or rate limit exceeded
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing"
        )
    
    # Find the API key in our store
    key_info = None
    for user, info in API_KEYS.items():
        if secrets.compare_digest(api_key, info["key"]):
            key_info = info
            break
    
    if key_info is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    # Reset counter if it's a new day
    today = datetime.now().date()
    if key_info["last_reset"] != today:
        key_info["requests_today"] = 0
        key_info["last_reset"] = today
    
    # Check rate limit
    if key_info["requests_today"] >= key_info["rate_limit"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    # Increment request counter
    key_info["requests_today"] += 1
    
    return key_info


# Model for LLM configuration
class LlmConfig(BaseModel):
    api_url: str = Field(..., description="URL of the LLM API endpoint")
    model_name: str = Field(..., description="Name of the model to use")
    api_key: Optional[str] = Field(None, description="API key for the LLM service (if required)")
    connect_timeout: int = Field(10, description="Connection timeout in seconds")
    read_timeout: int = Field(120, description="Read timeout in seconds")
    temperature: float = Field(0.3, description="Temperature for LLM generation")


# Model for prediction request
class PredictionRequest(BaseModel):
    tickers: List[str] = Field(
        default_factory=lambda: config.get("default_tickers"),
        description="List of stock tickers to analyze"
    )
    prediction_horizon: int = Field(
        default=5,
        description="Number of days to predict into the future",
        ge=1,
        le=30
    )
    news_days: int = Field(
        default=10,
        description="Number of days of news to analyze",
        ge=1,
        le=30
    )
    market_data_days: int = Field(
        default=30,
        description="Number of days of market data to analyze",
        ge=5,
        le=365
    )
    llm_config: Optional[LlmConfig] = Field(
        None, 
        description="Configuration for the LLM interpreter (if not provided, system defaults will be used)"
    )
    news_api_key: Optional[str] = Field(
        None,
        description="NewsAPI key (if not provided, system default will be used)"
    )
    alpha_vantage_api_key: Optional[str] = Field(
        None,
        description="Alpha Vantage API key (if not provided, system default will be used)"
    )


# Model for prediction response
class PredictionResponse(BaseModel):
    request_id: str = Field(..., description="Unique ID for this prediction request")
    date: str = Field(..., description="Date of the prediction")
    prediction_horizon: int = Field(..., description="Number of days predicted into the future")
    tickers: List[str] = Field(..., description="List of stock tickers analyzed")
    predictions: Dict[str, float] = Field(..., description="Prediction values for direction, magnitude, and volatility")
    interpretation: Optional[str] = Field(None, description="LLM interpretation of the prediction")
    created_at: str = Field(..., description="Timestamp when this prediction was created")


# Model for API key creation request
class ApiKeyRequest(BaseModel):
    username: str = Field(..., description="Username for the API key")
    is_admin: bool = Field(False, description="Whether this key has admin privileges")
    rate_limit: int = Field(100, description="Daily rate limit for this key")


# Model for API key response
class ApiKeyResponse(BaseModel):
    username: str = Field(..., description="Username for the API key")
    key: str = Field(..., description="The API key")
    is_admin: bool = Field(..., description="Whether this key has admin privileges")
    rate_limit: int = Field(..., description="Daily rate limit for this key")
    created_at: str = Field(..., description="Timestamp when this key was created")


# Function to load or get cached model
def get_model_system():
    """
    Load the VectorFin model system or get it from cache.
    
    Returns:
        The loaded VectorFinSystem instance
    """
    models_dir = config.get("models_dir")
    
    # Check if we need to (re)load the model
    if (MODEL_CACHE["system"] is None or 
        MODEL_CACHE["models_dir"] != models_dir or
        (MODEL_CACHE["loaded_at"] is not None and 
         (datetime.now() - MODEL_CACHE["loaded_at"]).days >= 1)):
        
        # Load the model
        MODEL_CACHE["system"] = load_trained_model(models_dir)
        MODEL_CACHE["models_dir"] = models_dir
        MODEL_CACHE["loaded_at"] = datetime.now()
    
    return MODEL_CACHE["system"]


# Routes
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information."""
    return {
        "app": "VectorFin API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/docs", include_in_schema=False)
async def get_docs(api_key_info: Dict = Depends(get_api_key)):
    """Swagger UI documentation with API key authentication."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="VectorFin API",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


@app.post(
    "/predict", 
    response_model=PredictionResponse, 
    summary="Make a financial prediction",
    description="Analyze market data and news to make financial predictions with optional custom LLM configuration."
)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    api_key_info: Dict = Depends(get_api_key)
):
    """
    Make a financial prediction based on market data and news.
    
    Args:
        request: The prediction request parameters
        background_tasks: FastAPI background tasks
        api_key_info: API key information
        
    Returns:
        The prediction results and interpretation
    """
    try:
        # Set temporary environment variables if provided in the request
        old_news_api_key = os.environ.get("NEWS_API_KEY")
        old_alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
        
        # Set API keys if provided
        if request.news_api_key:
            os.environ["NEWS_API_KEY"] = request.news_api_key
            
        if request.alpha_vantage_api_key:
            os.environ["ALPHA_VANTAGE_API_KEY"] = request.alpha_vantage_api_key
        
        # Build the LLM configuration for this request
        llm_config = None
        if request.llm_config:
            llm_config = {
                "api_url": request.llm_config.api_url,
                "model_name": request.llm_config.model_name,
                "api_key": request.llm_config.api_key,
                "connect_timeout": request.llm_config.connect_timeout,
                "read_timeout": request.llm_config.read_timeout,
                "temperature": request.llm_config.temperature,
                "max_retries": int(os.getenv("LLM_MAX_RETRIES", "2"))
            }
            logger.info(f"Using custom LLM configuration: {request.llm_config.api_url}, {request.llm_config.model_name}")
        
        # Get the model
        system = get_model_system()
        
        # Fetch data
        logger.info(f"Fetching data for tickers: {request.tickers}")
        news_data = fetch_recent_news(request.tickers, days=request.news_days)
        market_data = fetch_market_data(request.tickers, days=request.market_data_days)
        
        # Prepare data for prediction
        processed_news, aligned_market_data = prepare_data_for_prediction(news_data, market_data)
        
        # Make prediction
        prediction_results = make_prediction(
            system, 
            processed_news, 
            aligned_market_data, 
            prediction_horizon=request.prediction_horizon
        )
        
        # Get interpretation with custom LLM config if provided
        try:
            interpretation = query_llm_for_interpretation(
                prediction_results, 
                aligned_market_data, 
                processed_news, 
                request.prediction_horizon,
                llm_config=llm_config
            )
        except Exception as llm_error:
            logger.error(f"LLM interpretation failed: {str(llm_error)}", exc_info=True)
            # Use fallback interpretation but include error information
            interpretation = generate_fallback_interpretation(prediction_results, aligned_market_data, processed_news)
            interpretation += f"\n\n### Note\nThe LLM-based interpretation failed with error: {str(llm_error)}\nThis is an automated fallback interpretation."
        
        # Create response
        response = {
            "request_id": secrets.token_urlsafe(16),
            "date": prediction_results["date"],
            "prediction_horizon": prediction_results["prediction_horizon"],
            "tickers": request.tickers,
            "predictions": prediction_results["predictions"],
            "interpretation": interpretation,
            "created_at": datetime.now().isoformat()
        }
        
        # Schedule background task to store the prediction in the history
        background_tasks.add_task(
            save_prediction_to_history, 
            response
        )
        
        return response
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
    finally:
        # Restore original environment variables
        if request.news_api_key:
            if old_news_api_key:
                os.environ["NEWS_API_KEY"] = old_news_api_key
            else:
                os.environ.pop("NEWS_API_KEY", None)
                
        if request.alpha_vantage_api_key:
            if old_alpha_vantage_api_key:
                os.environ["ALPHA_VANTAGE_API_KEY"] = old_alpha_vantage_api_key
            else:
                os.environ.pop("ALPHA_VANTAGE_API_KEY", None)


@app.post(
    "/api-keys", 
    response_model=ApiKeyResponse, 
    summary="Create a new API key",
    description="Create a new API key for accessing the VectorFin API. Requires admin privileges."
)
async def create_api_key(
    request: ApiKeyRequest,
    api_key_info: Dict = Depends(get_api_key)
):
    """
    Create a new API key.
    
    Args:
        request: The API key creation request
        api_key_info: API key information for authentication
        
    Returns:
        The newly created API key
    """
    # Check admin privileges
    if not api_key_info.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    # Check if username already exists
    if request.username in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{request.username}' already exists"
        )
    
    # Create new API key
    new_key = secrets.token_urlsafe(32)
    API_KEYS[request.username] = {
        "key": new_key,
        "is_admin": request.is_admin,
        "rate_limit": request.rate_limit,
        "requests_today": 0,
        "last_reset": datetime.now().date()
    }
    
    return {
        "username": request.username,
        "key": new_key,
        "is_admin": request.is_admin,
        "rate_limit": request.rate_limit,
        "created_at": datetime.now().isoformat()
    }


@app.get(
    "/config", 
    summary="Get current configuration",
    description="Get the current system configuration. Requires admin privileges."
)
async def get_config(api_key_info: Dict = Depends(get_api_key)):
    """
    Get the current system configuration.
    
    Args:
        api_key_info: API key information for authentication
        
    Returns:
        The current configuration
    """
    # Check admin privileges
    if not api_key_info.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    # Get configuration (excluding sensitive information)
    conf = config.all
    
    # Remove API keys
    if "llm" in conf and "api_key" in conf["llm"]:
        conf["llm"]["api_key"] = "***REDACTED***"
    if "news_api" in conf and "api_key" in conf["news_api"]:
        conf["news_api"]["api_key"] = "***REDACTED***"
    
    return conf


@app.post(
    "/config", 
    summary="Update configuration",
    description="Update the system configuration. Requires admin privileges."
)
async def update_config(
    request: Dict[str, Any],
    api_key_info: Dict = Depends(get_api_key)
):
    """
    Update the system configuration.
    
    Args:
        request: The configuration update request
        api_key_info: API key information for authentication
        
    Returns:
        Success message
    """
    # Check admin privileges
    if not api_key_info.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    # Update configuration
    for key_path, value in flatten_dict(request).items():
        config.set(key_path, value)
    
    # Save configuration to file
    config.save_to_file()
    
    return {"status": "success", "message": "Configuration updated"}


# Background task to save prediction to history
def save_prediction_to_history(prediction: Dict):
    """
    Save a prediction to the prediction history file.
    
    Args:
        prediction: The prediction to save
    """
    try:
        # Create prediction outputs directory if it doesn't exist
        outputs_dir = Path("prediction_outputs")
        outputs_dir.mkdir(exist_ok=True)
        
        # Save prediction JSON
        date_str = datetime.now().strftime("%Y%m%d")
        prediction_file = outputs_dir / f"prediction_{date_str}.json"
        with open(prediction_file, "w") as f:
            json.dump(prediction, f, indent=2)
        
        # Save interpretation
        if prediction.get("interpretation"):
            interpretation_file = outputs_dir / f"interpretation_{date_str}.txt"
            with open(interpretation_file, "w") as f:
                f.write(prediction["interpretation"])
        
        # Update prediction history CSV
        history_file = outputs_dir / "prediction_history.csv"
        
        # Create header if file doesn't exist
        if not history_file.exists():
            with open(history_file, "w") as f:
                f.write("date,direction,magnitude,volatility,tickers\n")
        
        # Append prediction to history
        with open(history_file, "a") as f:
            tickers_str = "-".join(prediction["tickers"])
            f.write(f"{prediction['date']},"
                    f"{prediction['predictions']['direction']},"
                    f"{prediction['predictions']['magnitude']},"
                    f"{prediction['predictions']['volatility']},"
                    f"{tickers_str}\n")
        
        logger.info(f"Prediction saved to history: {prediction_file}")
    except Exception as e:
        logger.error(f"Error saving prediction to history: {str(e)}")


# Helper function to flatten nested dictionary
def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    """
    Flatten a nested dictionary.
    
    Args:
        d: The dictionary to flatten
        parent_key: The parent key prefix
        sep: The separator to use between keys
        
    Returns:
        Flattened dictionary with dot-separated keys
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "message": "An internal server error occurred"}
    )


# Main entry point
if __name__ == "__main__":
    # Display the admin API key
    admin_key = API_KEYS["admin"]["key"]
    print(f"Admin API Key: {admin_key}")
    print("Keep this key secure. It provides full access to the API.")
    
    # Start the server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
