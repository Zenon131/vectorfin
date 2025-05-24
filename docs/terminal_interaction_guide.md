# Using the VectorFin Terminal Interface

This guide provides instructions for interacting with trained VectorFin models using the command-line interface. The terminal interface is the most straightforward way to test predictions and analyze financial data.

## Prerequisites

Before using the terminal interface, ensure that:

1. The VectorFin package is properly installed
2. You have trained models OR have created test models
3. You have Python 3.8+ installed with the necessary dependencies

## Step 1: Create Test Models (If Needed)

If you haven't trained VectorFin models yet, you can create test models to demonstrate the functionality:

```bash
cd /Users/jonathanwallace/vectorfin
python examples/create_test_models.py
```

This will create synthetic model files in the `trained_models` directory, which can be used to test the interaction scripts.

## Step 2: Run the Terminal Interface

The basic terminal interface is provided by `interact_with_model.py`. To run it:

```bash
cd /Users/jonathanwallace/vectorfin
python examples/interact_with_model.py
```

By default, this will:

1. Load the model from the `trained_models` directory
2. Generate synthetic news data for demonstration (for AAPL, MSFT, GOOGL)
3. Fetch market data for these tickers
4. Make a 5-day prediction using the model
5. Display the prediction results and an LLM-based interpretation

## Customizing the Interaction

To customize the interaction, you can modify the `main()` function in `interact_with_model.py`:

1. Change the tickers list to include different stocks
2. Adjust the prediction_horizon for different time frames
3. Modify the models_dir parameter if your models are stored elsewhere

For example:

```python
def main():
    # Define parameters
    tickers = ['TSLA', 'NVDA', 'AMD']  # Different tickers
    models_dir = "./my_custom_models"  # Custom model directory
    prediction_horizon = 10  # Longer prediction window

    # Rest of the function remains the same
    ...
```

## Advanced Usage: LLM-Enhanced Interpreter

For more advanced interpretation of model results, you can use the LLM-enhanced interpreter:

```bash
python examples/llm_enhanced_interpreter.py
```

This requires setting up an API key for the LLM provider (OpenAI or Anthropic) either in your environment or by passing it as an argument.

## Troubleshooting

### No Models Found

If you encounter errors about missing model files:

1. Check that the models exist in the expected directory
2. Run `examples/create_test_models.py` to create test models
3. Ensure the model_path parameter correctly points to your models

### Parameter Errors

If you see parameter errors in the VectorFinSystem initialization:

1. Ensure you're passing the class itself as the first parameter to `load_models`
2. Make sure the vector_dim, sentiment_dim, and fusion_dim match the dimensions used during training

### Data Loading Errors

If market data fails to load:

1. Check your internet connection
2. Verify the tickers are valid
3. Try with a shorter date range

## Next Steps

After getting familiar with the terminal interface, you might want to:

1. Train your own models with actual financial data
2. Integrate real news sources instead of synthetic data
3. Explore the web interface for a more visual experience
4. Develop automated trading strategies based on predictions
