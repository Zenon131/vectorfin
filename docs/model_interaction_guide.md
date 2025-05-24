# Interacting with the Trained VectorFin Model

This guide explains how to interact with your trained VectorFin model in various ways.

## Prerequisites

Before running any of the examples, ensure you have:

1. A trained VectorFin model (in the `./trained_models` directory)
2. All dependencies installed

```bash
pip install torch pandas numpy flask requests yfinance
```

## 1. Basic Model Interaction

The simplest way to interact with your trained model is using the `interact_with_model.py` script:

```bash
python examples/interact_with_model.py
```

This script:

- Loads a trained VectorFin model
- Fetches (or generates synthetic) financial news data
- Fetches market data for specified tickers
- Makes predictions using the model
- Uses a mock LLM interpretation (can be connected to actual LLM APIs)

## 2. LLM-Enhanced Interpretation

For more advanced interpretation of model predictions, use the `llm_enhanced_interpreter.py` script:

```bash
python examples/llm_enhanced_interpreter.py
```

To use this with an actual LLM:

1. Set up your API key as an environment variable:

   ```bash
   # For OpenAI
   export OPENAI_API_KEY=your_key_here

   # For Anthropic
   export ANTHROPIC_API_KEY=your_key_here
   ```

2. Modify the script to use your preferred provider:
   ```python
   interpreter = LLMFinancialInterpreter(
       model_path="./trained_models",
       llm_provider="openai"  # or "anthropic"
   )
   ```

This example demonstrates:

- Creating a specialized financial prediction interpreter
- Combining AI predictions with LLM analysis
- How to structure prompts for optimal financial analysis
- Handling different types of user questions

## 3. Web Interface

For a user-friendly interface, use the web application:

```bash
python examples/web_interface.py
```

Then open your browser to [http://localhost:5000](http://localhost:5000)

The web interface provides:

- A form to enter ticker symbols and news
- Options for prediction horizon
- Integration with LLM interpretation
- Visualized prediction results

## Customization

### Using Your Own News Data

To use real news data instead of the synthetic data in the examples:

```python
# Load news from a CSV file
news_data = FinancialTextData.load_news_data(
    filepath="path/to/your/news.csv",
    date_column="date",
    text_column="headline"
)
```

### Working with Different Prediction Horizons

You can adjust the prediction horizon (in days) in any of the examples:

```python
# Make predictions for 10 days ahead
prediction_results = make_prediction(system, news_data, market_data, prediction_horizon=10)
```

### Integrating with Other LLMs

The `LLMFinancialInterpreter` class can be extended to work with other LLM providers:

```python
def _call_your_llm_api(self, system_prompt, user_prompt):
    # Your code to call your preferred LLM API
    pass
```

## Model Output Format

The model produces predictions with the following structure:

```python
{
    'date': '2023-05-15',            # Date of prediction
    'prediction_horizon': 5,         # Days ahead
    'predictions': {
        'direction': 0.62,           # Probability of upward movement (>0.5 means up)
        'magnitude': 0.023,          # Expected percentage change
        'volatility': 0.015          # Expected volatility
    },
    'confidence_score': 0.78         # Overall confidence
}
```

## LLM Interpretation Structure

When using LLM interpretation, you typically get analysis covering:

1. Overall interpretation of the prediction
2. Key factors potentially influencing the prediction
3. Confidence assessment
4. Potential scenarios that could play out
5. Investment considerations

## Advanced Usage

### Programmatically Accessing Model Components

You can access individual components of the VectorFin system:

```python
# Access the text vectorizer
text_vectors = system.text_vectorizer.vectorize_texts(texts)

# Access the numerical vectorizer
num_vectors = system.num_vectorizer.vectorize(market_data)

# Access the alignment layer
aligned_text, aligned_num = system.alignment(text_vectors, num_vectors)
```

### Adding Custom Transforms

You can add custom transformations to the data processing pipeline:

```python
def custom_transform(sample):
    # Transform the sample data
    return transformed_sample

# Apply the transform when loading the dataset
dataset = AlignedFinancialDataset(
    text_data=news_data,
    market_data=market_data,
    transform=custom_transform
)
```

---

For more information, refer to the VectorFin documentation or the source code.
