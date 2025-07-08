# VectorFin API Documentation

## Overview

The VectorFin API provides a RESTful interface to the VectorFin financial prediction system. It allows users to make predictions, interpret them, and configure the system.

## Authentication

All API endpoints require an API key, which should be provided in the `X-API-Key` header.

```http
X-API-Key: your_api_key_here
```

## Endpoints

### GET /

Returns basic information about the API.

**Response:**

```json
{
  "app": "VectorFin API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

### GET /docs

Shows interactive API documentation (Swagger UI). Requires API key authentication.

### POST /predict

Make a financial prediction based on market data and news.

**Request Body:**

```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "prediction_horizon": 5,
  "news_days": 10,
  "market_data_days": 30,
  "llm_config": {
    "api_url": "http://your-llm-api-url/v1/chat/completions",
    "model_name": "your-model-name",
    "api_key": "your-llm-api-key",
    "connect_timeout": 10,
    "read_timeout": 120,
    "temperature": 0.3
  },
  "news_api_key": "your-news-api-key"
}
```

#### Request Field Details

| Field | Type | Description |
|-------|------|-------------|
| `tickers` | array | List of stock ticker symbols to analyze |
| `prediction_horizon` | integer | Number of days to predict into the future |
| `news_days` | integer | Number of days of news to analyze |
| `market_data_days` | integer | Number of days of market data to analyze |
| `news_api_key` | string | Optional API key for news service |
| `llm_config` | object | Optional configuration for the LLM interpreter |

#### LLM Configuration Options

The `llm_config` object allows you to customize the LLM used for interpreting predictions:

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `api_url` | string | URL of the LLM API endpoint | Default internal LLM |
| `model_name` | string | Name of the model to use | gemma-3-4b-it-qat |
| `api_key` | string | API key for authentication (if required) | None |
| `connect_timeout` | integer | Connection timeout in seconds | 10 |
| `read_timeout` | integer | Read timeout in seconds | 120 |
| `temperature` | float | Temperature for text generation (0-1) | 0.3 |

All request fields are optional except `tickers`. If not provided, the system will use defaults.

**Response:**

```json
{
  "request_id": "unique-request-id",
  "date": "2025-07-07",
  "prediction_horizon": 5,
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "predictions": {
    "direction": 0.65,
    "magnitude": 1.2,
    "volatility": 0.8
  },
  "interpretation": "Market analysis and recommendation...",
  "created_at": "2025-07-08T12:30:45"
}
```

### POST /api-keys

Create a new API key. Requires admin privileges.

**Request Body:**

```json
{
  "username": "new_user",
  "is_admin": false,
  "rate_limit": 100
}
```

**Response:**

```json
{
  "username": "new_user",
  "key": "generated-api-key",
  "is_admin": false,
  "rate_limit": 100,
  "created_at": "2025-07-08T12:30:45"
}
```

### GET /config

Get the current system configuration. Requires admin privileges.

**Response:**

```json
{
  "models_dir": "./trained_models",
  "prediction_horizon": 5,
  "default_tickers": ["AAPL", "MSFT", "GOOGL"],
  "llm": {
    "api_url": "http://10.102.138.33:6223/v1/chat/completions",
    "model_name": "gemma-3-4b-it-qat",
    "api_key": "***REDACTED***",
    "connect_timeout": 10,
    "read_timeout": 120,
    "max_retries": 2,
    "temperature": 0.3
  },
  "news_api": {
    "url": "https://newsapi.org/v2/everything",
    "days_to_fetch": 10,
    "language": "en",
    "sort_by": "publishedAt",
    "page_size": 100,
    "api_key": "***REDACTED***"
  }
}
```

### POST /config

Update the system configuration. Requires admin privileges.

**Request Body:**

```json
{
  "llm": {
    "api_url": "http://new-llm-api-url/v1/chat/completions",
    "model_name": "new-model-name"
  },
  "default_tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"]
}
```

**Response:**

```json
{
  "status": "success",
  "message": "Configuration updated"
}
```

## Error Handling

All errors return a JSON response with a status code and message:

```json
{
  "status": "error",
  "message": "Error message details"
}
```

Common error codes:

- `401`: Unauthorized (missing or invalid API key)
- `403`: Forbidden (insufficient permissions)
- `429`: Too Many Requests (rate limit exceeded)
- `500`: Internal Server Error

## Rate Limiting

Each API key has a daily rate limit. If you exceed this limit, you will receive a `429` error.

## Examples

### Making a Prediction

**curl:**

```bash
curl -X POST "http://your-api-host:8000/predict" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "prediction_horizon": 5,
    "llm_config": {
      "api_url": "http://your-llm-api/v1/chat/completions",
      "model_name": "your-model-name",
      "api_key": "your-llm-api-key"
    }
  }'
```

**Python:**

```python
import requests
import json

url = "http://your-api-host:8000/predict"
headers = {
    "X-API-Key": "your-api-key",
    "Content-Type": "application/json"
}
data = {
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "prediction_horizon": 5,
    "llm_config": {
        "api_url": "http://your-llm-api/v1/chat/completions",
        "model_name": "your-model-name",
        "api_key": "your-llm-api-key"
    }
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
print(json.dumps(result, indent=2))
```
