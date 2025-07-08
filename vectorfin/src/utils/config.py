"""
Configuration management for VectorFin.

This module provides centralized configuration handling for API keys,
endpoints, model parameters, and other settings.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging
from dotenv import load_dotenv

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration manager for VectorFin."""
    
    # Default configuration
    _defaults = {
        "models_dir": "./trained_models",
        "prediction_horizon": 5,
        "default_tickers": ["AAPL", "MSFT", "GOOGL"],
        "llm": {
            "api_url": "http://10.102.138.33:6223/v1/chat/completions",
            "model_name": "gemma-3-4b-it-qat",
            "connect_timeout": 10,
            "read_timeout": 120,
            "max_retries": 2,
            "temperature": 0.3,
        },
        "news_api": {
            "url": "https://www.alphavantage.co/query",
            "function": "NEWS_SENTIMENT",
            "days_to_fetch": 10,
            "limit": 1000
        },
        "market_data": {
            "days_to_fetch": 30
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }
    
    # Instance of the configuration
    _instance = None
    
    # The actual configuration
    _config = {}
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from environment and files."""
        # Start with defaults
        self._config = self._defaults.copy()
        
        # Override with environment variables
        self._load_from_env()
        
        # Override with config file if it exists
        config_path = os.getenv("VECTORFIN_CONFIG", str(Path.home() / ".vectorfin" / "config.json"))
        if os.path.exists(config_path):
            self._load_from_file(config_path)
        
        # Setup logging based on config
        self._setup_logging()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # LLM configuration
        if os.getenv("LLM_API_URL"):
            self._config["llm"]["api_url"] = os.getenv("LLM_API_URL")
        if os.getenv("INTERPRETATION_MODEL"):
            self._config["llm"]["model_name"] = os.getenv("INTERPRETATION_MODEL")
        if os.getenv("LLM_API_KEY"):
            self._config["llm"]["api_key"] = os.getenv("LLM_API_KEY")
        if os.getenv("LLM_CONNECT_TIMEOUT"):
            self._config["llm"]["connect_timeout"] = int(os.getenv("LLM_CONNECT_TIMEOUT"))
        if os.getenv("LLM_READ_TIMEOUT"):
            self._config["llm"]["read_timeout"] = int(os.getenv("LLM_READ_TIMEOUT"))
        
        # News API configuration - use Alpha Vantage API key if available
        if os.getenv("ALPHA_VANTAGE_API_KEY"):
            self._config["news_api"]["api_key"] = os.getenv("ALPHA_VANTAGE_API_KEY")
        elif os.getenv("NEWS_API_KEY"):
            self._config["news_api"]["api_key"] = os.getenv("NEWS_API_KEY")
        
        # Model directory
        if os.getenv("VECTORFIN_MODELS_DIR"):
            self._config["models_dir"] = os.getenv("VECTORFIN_MODELS_DIR")
    
    def _load_from_file(self, config_path):
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                # Deep merge the configurations
                self._deep_merge(self._config, file_config)
        except Exception as e:
            logger.warning(f"Failed to load configuration from {config_path}: {e}")
    
    def _deep_merge(self, target, source):
        """Deep merge two dictionaries."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def _setup_logging(self):
        """Setup logging based on configuration."""
        logging.basicConfig(
            level=getattr(logging, self._config["logging"]["level"]),
            format=self._config["logging"]["format"]
        )
    
    def get(self, key_path: str, default=None) -> Any:
        """
        Get a configuration value by its key path.
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., "llm.api_url")
            default: Default value to return if the key is not found
            
        Returns:
            The configuration value or default if not found
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value by its key path.
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., "llm.api_url")
            value: The value to set
        """
        keys = key_path.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
    
    def save_to_file(self, config_path=None) -> bool:
        """
        Save the current configuration to a file.
        
        Args:
            config_path: Path to save the configuration to. If None, uses default path.
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not config_path:
            config_path = os.getenv("VECTORFIN_CONFIG", str(Path.home() / ".vectorfin" / "config.json"))
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(self._config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            return False
    
    @property
    def all(self) -> Dict:
        """Get a copy of the entire configuration."""
        return self._config.copy()


# Create a singleton instance for easy importing
config = Config()

# Export the singleton instance
__all__ = ['config']
