"""
Vercel entry point for the VectorFin API
"""

import os
import sys
import logging
import secrets
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path for importing the vectorfin package
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the FastAPI app from api.py
from api import app as api_app
from api import API_KEYS

# Setup logging for Vercel
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vercel-api")

# Generate and log admin API key only on cold starts
if "VERCEL" in os.environ and "admin" in API_KEYS:
    admin_key = API_KEYS["admin"]["key"]
    logger.info(f"Admin API Key (save this securely): {admin_key}")
    logger.info(f"API initialized at: {datetime.now().isoformat()}")

# Export the FastAPI app for Vercel
app = api_app
