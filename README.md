# VectorFin

VectorFin is a multimodal financial analysis system that combines numerical market data and sentiment/language analysis into a unified vector space. Unlike traditional systems that analyze these data types separately, VectorFin transforms all inputs into compatible vector representations that can be mathematically combined to provide holistic market insights.

## Overview

This system represents a novel approach to financial analysis by:

1. **Unifying Text and Numbers**: Transforms both financial text (news, social media) and numerical market data into the same vector space
2. **Multi-modal Learning**: Leverages cross-attention mechanisms to allow text and numerical data to influence each other
3. **Explainable Predictions**: Provides attention-based explanations for which inputs influenced predictions
4. **Vector Space Operations**: Enables semantic navigation of financial concepts
5. **Production-Ready API**: Exposes predictions through a RESTful API with custom LLM integration

## Core Components

### 1. Text Vectorization Module

Transforms financial text into sentiment-enriched vector representations using:

- Finance-tuned transformer models (FinBERT)
- Sentiment augmentation
- Dimension reduction for the shared vector space

### 2. Numerical Data Module

Transforms market metrics into meaningful vector representations via:

- Comprehensive market feature extraction
- Autoencoder architecture
- Market regime awareness

### 3. Alignment and Integration Layer

Creates a unified vector space through:

- Cross-modal attention mechanisms
- Contrastive learning
- Calibration for text vs. numerical influence

### 4. Prediction and Interpretation Module

Converts unified vectors into actionable insights through:

- Multiple prediction heads (direction, magnitude, volatility, timing)
- Attention-based explainability
- Semantic navigation of the vector space

### 5. API Server

Exposes VectorFin capabilities through a RESTful API:

- Authentication and rate limiting
- Custom LLM integration for interpretations
- Configurable prediction parameters
- Robust error handling and fallbacks

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vectorfin.git
cd vectorfin

# Install core requirements
pip install -r requirements.txt

# For API functionality, install API requirements
pip install -r requirements-api.txt
```

## Prerequisites

Before using VectorFin, you'll need:

1. **NewsAPI Key**: Required for fetching real financial news. Sign up at [newsapi.org](https://newsapi.org/).
2. **LLM API**: For enhanced prediction interpretation. You can use:
   - Local LLM setup (default configuration)
   - OpenAI API (requires API key)
   - Any other LLM provider with an OpenAI-compatible endpoint

## Environment Setup

Create a `.env` file in the root directory with the following variables:

```
# Required for news data
NEWS_API_KEY=your_newsapi_key_here

# LLM configuration (optional, defaults provided)
LLM_API_URL=your_llm_api_url_here
INTERPRETATION_MODEL=your_model_name_here
LLM_API_KEY=your_llm_api_key_here
LLM_CONNECT_TIMEOUT=10
LLM_READ_TIMEOUT=120

# For API server (optional)
ADMIN_API_KEY=your_custom_admin_key_here
```

## Usage Options

VectorFin can be used in multiple ways:

1. **Python Library**: Import and use VectorFin components in your Python code
2. **Command-line Tool**: Run the `interact_with_model.py` script for predictions
3. **API Server**: Use the RESTful API for integration with other systems

### Option 1: Python Library Usage

```python
from vectorfin.src.models.vectorfin import VectorFinSystem
import pandas as pd

# Create VectorFin system
system = VectorFinSystem(vector_dim=128)

# Example financial texts
texts = [
    "The company reported better than expected earnings, raising their guidance for the next quarter.",
    "The stock plummeted after the CEO announced his resignation amid fraud allegations."
]

# Example market data
market_data = pd.DataFrame({
    'open': [150.0, 152.0],
    'high': [155.0, 153.0],
    'low': [149.0, 145.0],
    'close': [153.0, 146.0],
    'volume': [1000000, 1500000]
})

# Analyze text and market data together
analysis = system.analyze_text_and_market(texts, market_data)

# Print market predictions and explanations
for i, interp in enumerate(analysis["interpretations"]):
    print(f"\nSample {i}:")
    print(interp["summary"])
```

### Option 2: Command-line Tool

For real-time predictions with live market and news data, you can use the `interact_with_model.py` script:

```bash
# Basic usage with default settings
python examples/interact_with_model.py

# With custom parameters
python examples/interact_with_model.py --tickers AAPL MSFT META --horizon 7 --llm-api-url "https://api.openai.com/v1/chat/completions" --llm-model "gpt-3.5-turbo"
```

This script will:
1. Load a pre-trained VectorFin model
2. Fetch recent news for specified tickers using NewsAPI
3. Fetch recent market data for those tickers
4. Generate predictions based on the combined data
5. Use an LLM to interpret the predictions and provide insights

### Option 3: API Server

Start the API server:

```bash
# Using the provided script
./run_api_server.sh

# Or manually
python api.py
```

The server will start on `http://0.0.0.0:8000`. The admin API key will be displayed in the console when you start the server. Make sure to save this key securely as it provides full access to the API.

Access the API documentation at `http://localhost:8000/docs` using your API key.

#### Making a Prediction via API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "prediction_horizon": 5
  }'
```

## Customizing VectorFin

### Using a Different LLM Provider

VectorFin allows you to use any LLM provider for prediction interpretation:

#### Option 1: Through Command-line Arguments

```bash
python examples/interact_with_model.py --llm-api-url "https://api.openai.com/v1/chat/completions" --llm-model "gpt-3.5-turbo"
```

#### Option 2: Through Environment Variables

```bash
export LLM_API_URL="https://api.openai.com/v1/chat/completions"
export INTERPRETATION_MODEL="gpt-3.5-turbo"
export LLM_API_KEY="your-openai-api-key"

python examples/interact_with_model.py
```

#### Option 3: Through the API

```python
import requests

# Your VectorFin API key
api_key = "your_vectorfin_api_key"

# Custom LLM configuration
llm_config = {
    "api_url": "https://api.openai.com/v1/chat/completions",
    "model_name": "gpt-3.5-turbo",
    "api_key": "your_openai_api_key",
    "temperature": 0.2,
    "connect_timeout": 15,
    "read_timeout": 180
}

# Make prediction request with custom LLM
response = requests.post(
    "http://localhost:8000/predict",
    headers={
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    },
    json={
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "prediction_horizon": 5,
        "llm_config": llm_config
    }
)

# Process response
if response.status_code == 200:
    result = response.json()
    print(f"Prediction: {result['predictions']}")
    print(f"Interpretation: {result['interpretation']}")
else:
    print(f"Error: {response.text}")
```

### Supported LLM APIs

The system is designed to work with any LLM provider that follows the OpenAI-compatible chat completions API format, including:

- OpenAI (ChatGPT)
- Anthropic Claude (with the appropriate endpoint)
- Llama models (via Ollama)
- Local models (LM Studio, etc.)
- Custom self-hosted models

### LLM Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `api_url` | URL of the LLM API endpoint | `http://10.102.138.33:6223/v1/chat/completions` |
| `model_name` | Name of the model to use | gemma-3-4b-it-qat |
| `api_key` | API key for authentication (if required) | None |
| `connect_timeout` | Connection timeout in seconds | 10 |
| `read_timeout` | Read timeout in seconds | 120 |
| `temperature` | Temperature for text generation (0-1) | 0.3 |

## Automating Daily Predictions

To automate predictions on a daily basis, you can use a cron job or task scheduler.

### Setting Up a Cron Job (Unix/Linux/macOS)

1. Use the provided shell script or create one:

```bash
# Make the script executable
chmod +x run_daily_prediction.sh

# Open crontab for editing
crontab -e

# Add this line to run daily at 8:00 AM
0 8 * * * /path/to/vectorfin/run_daily_prediction.sh
```

## API Documentation

For detailed API documentation, see the [API Documentation](docs/api_documentation.md) or access the interactive docs at `http://localhost:8000/docs` when the API server is running.

### Key API Endpoints

- **POST /predict**: Make a financial prediction
- **GET /config**: Get current configuration (admin only)
- **POST /config**: Update configuration (admin only)
- **POST /api-keys**: Create a new API key (admin only)

### API Security Considerations

- The API uses API key authentication
- API keys are stored in memory by default (consider using a database for production)
- Admin API keys provide full access to the system configuration
- All API calls should be made over HTTPS in production

## Making the API Internet-Accessible

By default, the API runs locally on `0.0.0.0:8000`, making it available:
- On your local machine at `http://localhost:8000`
- On your local network at your machine's IP address

To make it accessible across the internet:

### Option 1: Deploy to Vercel (Recommended for Easy Setup)

VectorFin can be deployed to Vercel's serverless platform:

1. **Push your code to GitHub/GitLab/Bitbucket**
2. **Connect to Vercel and import your repository**
3. **Add your environment variables**:

   ```bash
   # Required
   NEWS_API_KEY=your_newsapi_key_here
   
   # Optional (defaults will be used if not set)
   LLM_API_URL=your_llm_api_url_here  # Optional
   INTERPRETATION_MODEL=your_model_name_here  # Optional
   ```
   
4. **Deploy and access your API at your-project.vercel.app**

For detailed instructions, see [Vercel Deployment Guide](docs/vercel_deployment_guide.md).

### Option 2: Traditional Cloud Deployment

1. **Deploy to a cloud provider**:
   - AWS (EC2, ECS, Lambda)
   - Google Cloud (GCE, GKE, Cloud Run)
   - Azure (VM, AKS, App Service)

2. **Set up security**:
   - Use HTTPS (SSL/TLS certificates)
   - Implement more robust authentication
   - Consider a WAF (Web Application Firewall)

3. **Configure networking**:
   - Set up a domain name
   - Configure DNS records
   - Set up proper firewall rules

## Troubleshooting

### Common Issues and Solutions

#### 1. NewsAPI Rate Limiting

**Issue**: "You have made too many requests recently" error from NewsAPI.

**Solution**: NewsAPI free tier has a limit of 100 requests per day. You can:

- Use a paid plan for more requests
- Implement caching to reduce API calls
- Reduce the number of tickers analyzed

#### 2. LLM Endpoint Connection Errors

**Issue**: "Connection refused" or "Connection error" when calling LLM endpoint.

**Solution**:

- Verify your LLM server is running
- Check the URL and port in your configuration
- If using a remote server, ensure proper network access

#### 3. API Authentication Issues

**Issue**: "Invalid API key" or "API key is missing" errors when using the API.

**Solution**:

- Ensure you're including the `X-API-Key` header in your requests
- Verify the API key is correct
- For new deployments, check the console output for the generated admin API key

## Examples and Tools

- `examples/api_client.py`: Python client for the VectorFin API
- `examples/interact_with_model.py`: Command-line tool for predictions
- `examples/visualize_predictions.py`: Visualization tools for predictions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
