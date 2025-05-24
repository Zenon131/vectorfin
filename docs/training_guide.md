# VectorFin Training Guide

This guide provides comprehensive instructions for training the VectorFin system, a multimodal financial analysis framework that combines numerical market data and sentiment/language analysis in a unified vector space.

## Table of Contents

1. [Understanding the VectorFin Architecture](#understanding-the-vectorfin-architecture)
2. [Data Preparation](#data-preparation)
3. [Quick Start Training](#quick-start-training)
4. [Component-wise Training](#component-wise-training)
5. [End-to-End Training](#end-to-end-training)
6. [Training Parameters and Hyperparameters](#training-parameters-and-hyperparameters)
7. [Evaluation and Model Selection](#evaluation-and-model-selection)
8. [Saving and Loading Models](#saving-and-loading-models)
9. [Advanced Training Techniques](#advanced-training-techniques)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

## Understanding the VectorFin Architecture

Before diving into training, it's important to understand the modular architecture of VectorFin:

1. **Text Vectorization Module**: Transforms financial text into sentiment-enriched vector representations

   - Built on FinBERT transformer models
   - Incorporates sentiment augmentation
   - Projects text into a shared vector space

2. **Numerical Data Module**: Transforms market metrics into meaningful vector representations

   - Uses an autoencoder architecture
   - Processes technical indicators and market features
   - Projects numerical data into the same vector space as text

3. **Alignment and Integration Layer**: Creates a unified vector space

   - Uses cross-modal attention mechanisms
   - Implements contrastive learning
   - Aligns text and numerical vectors

4. **Prediction and Interpretation Module**: Generates market predictions and insights
   - Multiple prediction heads (direction, magnitude, volatility, timing)
   - Attention-based explainability
   - Semantic navigation capabilities

This modular design allows for both component-wise training (training each module separately) and end-to-end training (optimizing the entire system together).

## Data Preparation

### Market Data

VectorFin uses historical market data with OHLCV (Open, High, Low, Close, Volume) values:

```python
from vectorfin.src.data.data_loader import MarketData

# Fetch market data for multiple tickers
market_data = MarketData.fetch_market_data(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2020-01-01",
    end_date="2023-12-31"
)

# Preview the data
for ticker, data in market_data.items():
    print(f"{ticker} data shape: {data.shape}")
```

### Financial Text Data

For text data, you can either:

- Use a CSV file with financial news/social media data
- Generate synthetic news data for experimentation

```python
from vectorfin.src.data.data_loader import FinancialTextData

# Option 1: Load from CSV file
news_data = FinancialTextData.load_news_data("path/to/news_data.csv")

# Option 2: Generate synthetic news
# (see examples/training_pipeline.py for a detailed implementation)
```

### Data Alignment

To combine text and market data, temporal alignment is necessary:

```python
from vectorfin.src.data.data_loader import AlignedFinancialDataset

# Create an aligned dataset
dataset = AlignedFinancialDataset(
    text_data=processed_news,
    market_data=aligned_market_data,
    text_column='headline',
    date_column='date',
    prediction_horizon=5,  # Predict 5 days ahead
    max_texts_per_day=10   # Maximum texts per day to include
)
```

## Quick Start Training

The easiest way to train VectorFin is with the provided shell script:

```bash
# Make the script executable
chmod +x train.sh

# Run with default settings
./train.sh
```

This script:

1. Checks requirements and installs if necessary
2. Downloads market data for major tech companies
3. Generates synthetic news data
4. Trains all components end-to-end
5. Saves models to `./trained_models`

### Customizing the Quick Start

You can edit the `train.sh` script to modify default parameters:

```bash
# Edit these variables in train.sh
TICKERS="AAPL,MSFT,GOOGL,AMZN,META"  # Tickers to use
START_DATE="2022-01-01"              # Training data start date
END_DATE="2023-01-01"                # Training data end date
MODELS_DIR="./trained_models"        # Where to save models
EPOCHS=5                             # Number of training epochs
BATCH_SIZE=16                        # Batch size for training
```

## Component-wise Training

VectorFin supports training individual components for more control over the process.

### 1. Training the Text Vectorization Module

This component transforms financial text into sentiment-enriched vectors:

```bash
python examples/training_pipeline.py \
    --tickers "AAPL,MSFT,GOOGL" \
    --start_date "2020-01-01" \
    --end_date "2023-12-31" \
    --models_dir "./models" \
    --train_text_only
```

The text vectorizer is fine-tuned on financial texts with sentiment labels.

### 2. Training the Numerical Data Module

This trains the autoencoder for market data:

```bash
python examples/training_pipeline.py \
    --tickers "AAPL,MSFT,GOOGL" \
    --start_date "2020-01-01" \
    --end_date "2023-12-31" \
    --models_dir "./models" \
    --train_numerical_only
```

The numerical vectorizer learns to compress market features into a lower-dimensional space while preserving important information.

### 3. Training the Alignment Integration Layer

This trains the layer that aligns text and numerical vectors:

```bash
python examples/training_pipeline.py \
    --tickers "AAPL,MSFT,GOOGL" \
    --start_date "2020-01-01" \
    --end_date "2023-12-31" \
    --models_dir "./models" \
    --train_alignment_only
```

The alignment layer uses contrastive learning to ensure text and numerical vectors are compatible.

### 4. Training the Prediction Heads

This trains the different prediction mechanisms:

```bash
python examples/training_pipeline.py \
    --tickers "AAPL,MSFT,GOOGL" \
    --start_date "2020-01-01" \
    --end_date "2023-12-31" \
    --models_dir "./models" \
    --train_prediction_only
```

The prediction heads are trained to convert unified vectors into different types of market predictions.

## End-to-End Training

For optimal performance, end-to-end training is recommended:

```bash
python examples/training_pipeline.py \
    --tickers "AAPL,MSFT,GOOGL" \
    --start_date "2020-01-01" \
    --end_date "2023-12-31" \
    --models_dir "./models" \
    --epochs 10 \
    --batch_size 32 \
    --use_synthetic_news
```

End-to-end training allows all components to be optimized together, creating a more cohesive system.

### Programmatic End-to-End Training

You can also train VectorFin programmatically:

```python
from vectorfin.src.models.vectorfin import VectorFinSystem, VectorFinTrainer
from vectorfin.src.data.data_loader import AlignedFinancialDataset

# Create system
system = VectorFinSystem(
    vector_dim=128,
    sentiment_dim=16,
    fusion_dim=128
)

# Create trainer
trainer = VectorFinTrainer(
    system=system,
    learning_rate=1e-4,
    weight_decay=1e-5
)

# Train on your dataset
history = trainer.train_end_to_end(
    dataset=your_aligned_dataset,
    num_epochs=10,
    batch_size=32,
    save_dir="./models"
)

# Plot training history
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(history["total_loss"], label="Total Loss")
plt.legend()
plt.grid(True)
plt.title("VectorFin Training Progress")
plt.show()
```

## Training Parameters and Hyperparameters

VectorFin's training behavior can be customized with various parameters:

### Core Model Parameters

- `vector_dim` (default: 128): Dimension of vectors in the shared space
- `sentiment_dim` (default: 16): Dimension of sentiment features
- `fusion_dim` (default: 128): Dimension of fused vectors after alignment

### Training Control Parameters

- `epochs` (default: 10): Number of training epochs
- `batch_size` (default: 32): Batch size for training
- `learning_rate` (default: 1e-4): Learning rate for optimizers
- `weight_decay` (default: 1e-5): Weight decay for regularization

### Data Parameters

- `tickers`: Comma-separated list of ticker symbols
- `start_date`: Start date for market data (YYYY-MM-DD)
- `end_date`: End date for market data (YYYY-MM-DD)
- `train_test_split` (default: 0.8): Proportion of data to use for training
- `prediction_horizon` (default: 5): Number of days ahead to predict

### Component Training Flags

- `--train_text_only`: Train only the text vectorization module
- `--train_numerical_only`: Train only the numerical data module
- `--train_alignment_only`: Train only the alignment integration layer
- `--train_prediction_only`: Train only the prediction interpretation module

## Evaluation and Model Selection

After training, evaluate your models to select the best one:

```python
from vectorfin.src.models.vectorfin import VectorFinSystem
import pandas as pd

# Load trained system
system = VectorFinSystem.load_models(
    directory="./models",
    vector_dim=128,
    sentiment_dim=16,
    fusion_dim=128
)

# Evaluate on test data
metrics = evaluate_system(system, test_loader)

print(f"Direction Prediction Accuracy: {metrics['direction_accuracy']:.4f}")
print(f"Magnitude Prediction MSE: {metrics['magnitude_mse']:.6f}")
print(f"Volatility Prediction MSE: {metrics['volatility_mse']:.6f}")
```

### Key Metrics to Monitor

- **Direction Accuracy**: How well the model predicts price movement direction
- **Magnitude MSE**: Mean squared error for price change magnitude predictions
- **Volatility MSE**: Mean squared error for volatility predictions

## Saving and Loading Models

VectorFin models can be saved and loaded for future use:

```python
# Save model
system.save_models("./my_trained_models")

# Load model
system = VectorFinSystem.load_models(
    directory="./my_trained_models",
    vector_dim=128,
    sentiment_dim=16,
    fusion_dim=128
)
```

## Advanced Training Techniques

### Curriculum Learning

Train on simpler patterns first, then increase complexity:

1. First train on clear bull/bear markets
2. Then train on more nuanced market conditions

### Transfer Learning

1. Pre-train individual components on large datasets
2. Fine-tune the integrated system on specific tickers or market conditions

### Custom Loss Functions

For specialized prediction tasks, consider implementing custom loss functions:

```python
# Example: Custom asymmetric loss function that penalizes
# missing upside moves more than downside moves
def asymmetric_direction_loss(predictions, targets, up_weight=2.0):
    standard_loss = nn.BCELoss(reduction='none')(predictions, targets)

    # Increase weight for upward moves (where target is 1)
    weights = torch.ones_like(targets)
    weights[targets > 0.5] = up_weight

    return (standard_loss * weights).mean()
```

### Market Regime-Aware Training

Detect market regimes and train specialized models:

1. Identify different market regimes (trending, volatile, range-bound)
2. Train separate models or add regime awareness to the system

## Troubleshooting

### Common Issues and Solutions

#### Numerical Instability

**Issue**: Training diverges or produces NaN values.

**Solution**:

- Reduce learning rate
- Add gradient clipping:
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```

#### Data Imbalance

**Issue**: Model biased towards more frequent market conditions.

**Solution**:

- Use weighted sampling to balance different market regimes
- Implement class weights in loss functions

#### Poor Text-Market Alignment

**Issue**: Text and market data not properly aligned.

**Solution**:

- Ensure proper date matching in `AlignedFinancialDataset`
- Adjust the temporal windows for relating text to market movements

#### Out of Memory Errors

**Issue**: Training crashes due to memory limitations.

**Solution**:

- Reduce batch size
- Use gradient accumulation for large models

## Best Practices

### Data Quality

- **Diverse Tickers**: Include a diverse set of tickers across different sectors
- **Time Span**: Include both bull and bear markets for robustness
- **Data Cleaning**: Properly handle missing values and outliers

### Model Configuration

- **Vector Dimensions**: Start with smaller dimensions (64-128) and increase if needed
- **Learning Rate**: Start with 1e-4 and adjust based on training stability
- **Batch Size**: Use the largest batch size that fits in memory

### Training Strategy

- **Progressive Training**: Train components in order (text → numerical → alignment → prediction)
- **Checkpoint Saving**: Save models regularly during training
- **Validation**: Use a separate validation set to monitor for overfitting

### Inference

- **Ensemble Methods**: Consider averaging predictions from models trained with different seeds
- **Calibration**: Calibrate prediction probabilities on recent data
- **Continuous Learning**: Periodically retrain on new data to adapt to changing market conditions

---

By following this training guide, you'll be able to effectively train and optimize the VectorFin system for your specific financial analysis needs. Remember that financial markets are complex and stochastic, so continuous evaluation and refinement of your models is essential.

For additional assistance or to report issues, please file a ticket in the VectorFin repository.
