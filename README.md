# VectorFin

VectorFin is a multimodal financial analysis system that combines numerical market data and sentiment/language analysis into a unified vector space. Unlike traditional systems that analyze these data types separately, VectorFin transforms all inputs into compatible vector representations that can be mathematically combined to provide holistic market insights.

## Overview

This system represents a novel approach to financial analysis by:

1. **Unifying Text and Numbers**: Transforms both financial text (news, social media) and numerical market data into the same vector space
2. **Multi-modal Learning**: Leverages cross-attention mechanisms to allow text and numerical data to influence each other
3. **Explainable Predictions**: Provides attention-based explanations for which inputs influenced predictions
4. **Vector Space Operations**: Enables semantic navigation of financial concepts

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

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vectorfin.git
cd vectorfin

# Install requirements
pip install -r requirements.txt
```

## Prerequisites

Before using VectorFin, you'll need:

1. **NewsAPI Key**: Required for fetching real financial news. Sign up at [newsapi.org](https://newsapi.org/).
2. **Local LLM Endpoint (Optional)**: For enhanced interpretation, you can use a local language model server. The default setup expects an endpoint at `http://192.168.68.122:6223/v1/chat/completions` using a Gemma model, but this can be modified.

## Environment Setup

Create a `.env` file in the root directory with the following variables:

```
NEWS_API_KEY=your_newsapi_key_here
LLM_API_URL=your_llm_api_url_here  # Optional, defaults to local endpoint
```

## Usage Guide

### Basic Usage

VectorFin can be used to analyze financial data and generate predictions:

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

### Making Real-Time Predictions

For real-time predictions with live market and news data, you can use the `interact_with_model.py` script in the `examples` directory:

```bash
# Run the example script
python examples/interact_with_model.py
```

This script will:
1. Load a pre-trained VectorFin model
2. Fetch recent news for specified tickers using NewsAPI
3. Fetch recent market data for those tickers
4. Generate predictions based on the combined data
5. Use an LLM to interpret the predictions and provide insights

### Customizing the Prediction Process

You can customize the prediction process by modifying the following parameters in `interact_with_model.py`:

- `tickers`: List of stock tickers to analyze
- `models_dir`: Directory containing trained models
- `prediction_horizon`: Number of days to predict ahead

```python
# Define parameters
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']  # Add any tickers of interest
models_dir = "./trained_models"
prediction_horizon = 7  # Extended to 7 days
```

### Understanding VectorFin Outputs

VectorFin generates three key predictions:

1. **Direction**: Probability (0-1) of upward price movement
2. **Magnitude**: Expected percentage change in price
3. **Volatility**: Expected market volatility

The LLM interpretation provides:
- A concise summary of the prediction
- Key influencing factors based on news and market data
- Risk assessment
- Recommended action (buy, hold, sell)

## Automating Daily Predictions

To automate predictions on a daily basis, you can use a cron job or task scheduler.

### Setting Up a Cron Job (Unix/Linux/macOS)

1. Create a wrapper shell script for the prediction task:

```bash
#!/bin/bash
# Save as /Users/jonathanwallace/vectorfin/run_daily_prediction.sh

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Set working directory
cd /Users/jonathanwallace/vectorfin

# Run the prediction script
python examples/interact_with_model.py > logs/prediction_$(date +\%Y\%m\%d).log 2>&1
```

2. Make the script executable:

```bash
chmod +x /Users/jonathanwallace/vectorfin/run_daily_prediction.sh
```

3. Add the cron job to run daily (e.g., at 8:00 AM):

```bash
# Open crontab for editing
crontab -e

# Add this line to run daily at 8:00 AM
0 8 * * * /Users/jonathanwallace/vectorfin/run_daily_prediction.sh
```

### Setting Up a Scheduled Task (Windows)

1. Create a batch script for the prediction task:

```batch
@echo off
REM Save as C:\path\to\vectorfin\run_daily_prediction.bat

REM Activate virtual environment if needed
REM call C:\path\to\your\venv\Scripts\activate.bat

REM Set working directory
cd C:\path\to\vectorfin

REM Run the prediction script
python examples\interact_with_model.py > logs\prediction_%date:~-4,4%%date:~-7,2%%date:~-10,2%.log 2>&1
```

2. Open Task Scheduler and create a new task:
   - Trigger: Daily at 8:00 AM
   - Action: Start a program
   - Program/script: `C:\path\to\vectorfin\run_daily_prediction.bat`

## Extending VectorFin

### Adding Custom Prediction Models

You can extend VectorFin by adding custom prediction models:

1. Create a new prediction head class in `vectorfin/src/prediction_interpretation/`
2. Register your new prediction head in the `PredictionInterpreter` class
3. Update the interpretation logic to handle your new prediction type

### Using a Different LLM Provider

To use a different LLM provider for interpretations:

1. Update the `query_llm_for_interpretation` function in `interact_with_model.py`:

```python
def query_llm_for_interpretation(prediction_results, market_data, news_data, prediction_horizon):
    """Send prediction results to an LLM API for interpretation."""
    # Create prompt with prediction data
    prompt = f"""
    Based on financial data and news analysis, interpret the following prediction:
    
    Prediction Horizon: {prediction_horizon} days
    
    Market Data Summary:
    {market_data_summary(market_data)}
    
    Recent News Headlines:
    {format_recent_news(news_data, 5)}
    
    Model Prediction Results:
    {json.dumps(prediction_results, indent=2)}
    
    Please provide:
    1. A concise interpretation of the prediction
    2. Key factors that might be influencing this prediction
    3. Potential risks or uncertainties to consider
    4. A recommendation based on this prediction (buy, hold, sell, etc.)
    """
    
    # Change this section to use your preferred LLM provider
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",  # Or other provider
        headers={
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4",  # Change model as needed
            "messages": [
                {"role": "system", "content": "You are a financial analysis assistant that interprets market predictions."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
    )
    
    # Parse response based on the API format
    interpretation = response.json()["choices"][0]["message"]["content"]
    
    return interpretation
```

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

#### 3. Data Alignment Issues

**Issue**: "KeyError" or "TypeError" when aligning market and news data.
**Solution**:
- Ensure both data sources have proper date formatting
- Check that tickers exist in both datasets
- Verify that date ranges overlap

## Training

VectorFin can be trained on financial data in multiple ways. For detailed training documentation, see the `docs/training_guide.md` file.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Financial NLP research from Stanford and MIT
- FinBERT project for financial language models
- The open-source financial analysis community

## Contact

For support or contributions, please open an issue on GitHub.
