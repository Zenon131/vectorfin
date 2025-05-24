"""
VectorFin Training Pipeline

This script demonstrates a complete end-to-end training process for the VectorFin system,
including data preparation, model configuration, and training.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, random_split
import sys
from torch.utils.data._utils.collate import default_collate

# Add parent directory to path so we can import the vectorfin package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import VectorFin components
from vectorfin.src.models.vectorfin import VectorFinSystem, VectorFinTrainer
from vectorfin.src.data.data_loader import FinancialTextData, MarketData, AlignedFinancialDataset
from vectorfin.src.text_vectorization import FinancialTextProcessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train the VectorFin system.')
    
    # Data parameters
    parser.add_argument('--tickers', type=str, default='AAPL,MSFT,GOOGL,AMZN,META',
                        help='Comma-separated list of ticker symbols')
    parser.add_argument('--start_date', type=str, default='2020-01-01',
                        help='Start date for market data (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-12-31',
                        help='End date for market data (YYYY-MM-DD)')
    parser.add_argument('--news_data', type=str, default=None,
                        help='Path to CSV file with financial news data')
    parser.add_argument('--use_synthetic_news', action='store_true',
                        help='Generate synthetic news if no news data is provided')
                        
    # Model parameters
    parser.add_argument('--vector_dim', type=int, default=128,
                        help='Dimension of vectors in the shared space')
    parser.add_argument('--sentiment_dim', type=int, default=16,
                        help='Dimension of sentiment features')
    parser.add_argument('--fusion_dim', type=int, default=128,
                        help='Dimension of fused vectors')
                        
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for optimizers')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizers')
    parser.add_argument('--train_test_split', type=float, default=0.8,
                        help='Proportion of data to use for training')
    parser.add_argument('--prediction_horizon', type=int, default=5,
                        help='Number of days ahead to predict')
                        
    # Output parameters
    parser.add_argument('--models_dir', type=str, default='./models',
                        help='Directory to save trained models')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='How often to log training progress (in batches)')
                        
    # Device parameters
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for training (cpu, cuda, or None for auto)')
                        
    # Component-wise training flags
    parser.add_argument('--train_text_only', action='store_true',
                        help='Train only the text vectorization module')
    parser.add_argument('--train_numerical_only', action='store_true',
                        help='Train only the numerical data module')
    parser.add_argument('--train_alignment_only', action='store_true',
                        help='Train only the alignment integration layer')
    parser.add_argument('--train_prediction_only', action='store_true',
                        help='Train only the prediction interpretation module')
    
    return parser.parse_args()


def fetch_market_data(args):
    """Fetch market data for training."""
    print(f"Fetching market data for {args.tickers.split(',')}")
    
    # Fetch data for each ticker
    market_data = MarketData.fetch_market_data(
        tickers=args.tickers.split(','),
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    print(f"Fetched market data for {len(market_data)} tickers")
    
    # Preview the data
    for ticker, data in market_data.items():
        print(f"{ticker} data shape: {data.shape}")
    
    return market_data


def load_or_generate_news_data(args):
    """Load existing news data or generate synthetic news data."""
    if args.news_data and os.path.exists(args.news_data):
        # Load news data from file
        print(f"Loading news data from {args.news_data}")
        news_data = FinancialTextData.load_news_data(args.news_data)
        print(f"Loaded {len(news_data)} news items")
    elif args.use_synthetic_news:
        # Generate synthetic news data
        print("Generating synthetic news data")
        news_data = generate_synthetic_news(args)
        print(f"Generated {len(news_data)} synthetic news items")
    else:
        raise ValueError("Either provide --news_data or use --use_synthetic_news")
        
    return news_data


def generate_synthetic_news(args):
    """Generate synthetic financial news data for training."""
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Generate dates
    num_days = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    # Templates for synthetic news
    positive_templates = [
        "{ticker} reports strong quarterly earnings, beating analyst expectations",
        "{ticker} announces new product line, stock rises",
        "Analysts upgrade {ticker} rating to 'buy', citing growth potential",
        "{ticker} expands into new markets, investors respond positively",
        "{ticker} exceeds revenue forecasts, shares rally",
        "Strong consumer demand boosts {ticker}'s sales figures",
        "{ticker} completes successful acquisition, integration ahead of schedule",
        "Cost-cutting measures improve {ticker}'s profit margins"
    ]
    
    negative_templates = [
        "{ticker} misses earnings expectations, shares decline",
        "{ticker} faces regulatory scrutiny over business practices",
        "Analysts downgrade {ticker} on growth concerns",
        "{ticker} recalls product due to safety issues",
        "Competition pressures {ticker}'s market share",
        "{ticker} cuts dividend, cites challenging market conditions",
        "Supply chain issues impact {ticker}'s production capacity",
        "{ticker} CEO steps down amid controversy"
    ]
    
    neutral_templates = [
        "{ticker} announces quarterly results in line with expectations",
        "{ticker} maintains current outlook despite market volatility",
        "Industry report highlights {ticker}'s stable position",
        "{ticker} reshuffles management team in planned transition",
        "{ticker} completes standard regulatory review",
        "Analysts maintain neutral stance on {ticker} stock",
        "{ticker} introduces minor updates to product lineup",
        "{ticker} holding annual shareholder meeting next month"
    ]
    
    # Create synthetic news
    tickers = args.tickers.split(',')
    news_items = []
    
    # Generate ~3 news items per day with different sentiments
    for date in dates:
        # Randomly select how many news items for this day (0-5)
        num_news = np.random.randint(0, 6)
        
        for _ in range(num_news):
            # Select random ticker
            ticker = np.random.choice(tickers)
            
            # Select sentiment (40% positive, 30% neutral, 30% negative)
            sentiment = np.random.choice(['positive', 'neutral', 'negative'], 
                                        p=[0.4, 0.3, 0.3])
            
            # Select template based on sentiment
            if sentiment == 'positive':
                template = np.random.choice(positive_templates)
            elif sentiment == 'neutral':
                template = np.random.choice(neutral_templates)
            else:
                template = np.random.choice(negative_templates)
            
            # Create headline
            headline = template.format(ticker=ticker)
            
            # Add to news items
            news_items.append({
                'date': date,
                'headline': headline,
                'ticker': ticker,
                'sentiment': sentiment,
                'source': 'synthetic'
            })
    
    # Convert to DataFrame
    news_data = pd.DataFrame(news_items)
    
    # Ensure date column is datetime
    news_data['date'] = pd.to_datetime(news_data['date'])
    
    # Sort by date
    news_data = news_data.sort_values('date')
    
    return news_data


def custom_collate_fn(batch):
    """
    Custom collate function that converts pandas Timestamps to strings.
    This prevents issues with the default collate function.
    """
    for item in batch:
        if 'date' in item and isinstance(item['date'], pd.Timestamp):
            item['date'] = str(item['date'])
        if 'target_date' in item and isinstance(item['target_date'], pd.Timestamp):
            item['target_date'] = str(item['target_date'])
    
    return default_collate(batch)


def prepare_dataset(market_data, news_data, args):
    """Prepare dataset for training."""
    print("Preparing dataset for training...")
    
    # Align market data
    aligned_market_data = MarketData.align_market_data(market_data)
    print(f"Aligned market data shape: {aligned_market_data.shape}")
    
    # Preprocess news data
    processed_news = FinancialTextData.preprocess_text_data(news_data)
    
    # Create dataset
    dataset = AlignedFinancialDataset(
        text_data=processed_news,
        market_data=aligned_market_data,
        text_column='headline',
        date_column='date',
        prediction_horizon=args.prediction_horizon,
        max_texts_per_day=10
    )
    
    print(f"Created dataset with {len(dataset)} samples")
    
    # Split into training and testing sets
    train_size = int(len(dataset) * args.train_test_split)
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Split into {len(train_dataset)} training and {len(test_dataset)} testing samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, test_loader


def create_and_train_system(train_loader, test_loader, args):
    """Create and train the VectorFin system."""
    print("Creating VectorFin system...")
    
    # Create system
    system = VectorFinSystem(
        vector_dim=args.vector_dim,
        sentiment_dim=args.sentiment_dim,
        fusion_dim=args.fusion_dim,
        device=args.device,
        models_dir=args.models_dir
    )
    
    # Create trainer
    trainer = VectorFinTrainer(
        system=system,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create models directory
    Path(args.models_dir).mkdir(exist_ok=True, parents=True)
    
    # Training based on selected components
    if args.train_text_only:
        train_text_vectorizer(trainer, train_loader, args)
        
    elif args.train_numerical_only:
        train_numerical_vectorizer(trainer, train_loader, args)
        
    elif args.train_alignment_only:
        train_alignment_layer(trainer, train_loader, args)
        
    elif args.train_prediction_only:
        train_prediction_heads(trainer, train_loader, test_loader, args)
        
    else:
        # Full end-to-end training
        print(f"Starting end-to-end training for {args.epochs} epochs...")
        history = train_end_to_end(trainer, train_loader, test_loader, args)
        
        # Visualize training history
        visualize_training_history(history, args)
    
    # Save the trained system
    system.save_models(args.models_dir)
    print(f"Saved trained models to {args.models_dir}")
    
    return system


def extract_batch_data(batch):
    """Extract text and market data from a batch."""
    texts = batch["texts"]
    
    # Flatten list of lists if necessary
    if isinstance(texts[0], list):
        texts = [item for sublist in texts for item in sublist]
    
    # Clean texts
    cleaned_texts = FinancialTextProcessor.batch_clean(texts)
    
    # Extract market data
    market_data = batch["market_data"]
    if isinstance(market_data, torch.Tensor):
        market_data = market_data.float()
    else:
        market_data = torch.tensor(market_data, dtype=torch.float32)
    
    # Extract targets if available
    targets = {}
    if "target" in batch:
        target_data = batch["target"]
        if isinstance(target_data, torch.Tensor):
            target_data = target_data.float()
        else:
            target_data = torch.tensor(target_data, dtype=torch.float32)
            
        # Create different target types (simplified for demonstration)
        # Direction: binary (up/down)
        targets["direction"] = (target_data[:, 3] > market_data[:, 3]).float().unsqueeze(1)
        
        # Magnitude: percentage change
        targets["magnitude"] = ((target_data[:, 3] - market_data[:, 3]) / market_data[:, 3]).unsqueeze(1)
        
        # Volatility: approximated by high-low range
        targets["volatility"] = ((target_data[:, 1] - target_data[:, 2]) / target_data[:, 3]).unsqueeze(1)
        
        # Timing: simplified to just use the prediction horizon day
        # In a real implementation, you would have a more sophisticated timing target
        horizon_day = torch.zeros(len(market_data), 30)
        for i in range(len(market_data)):
            horizon_day[i, 0] = 1  # Set day 0 as the target day
        targets["timing"] = horizon_day
        
    return cleaned_texts, market_data, targets


def train_text_vectorizer(trainer, train_loader, args):
    """Train just the text vectorization module."""
    print("Training text vectorization module...")
    
    # Collect text data and sentiment labels
    texts = []
    labels = []
    
    for batch in train_loader:
        batch_texts, _, _ = extract_batch_data(batch)
        texts.extend(batch_texts)
        
        # Generate synthetic sentiment labels for demonstration
        # In practice, you would use real sentiment labels
        batch_labels = torch.randint(0, 3, (len(batch_texts),)).tolist()
        labels.extend(batch_labels)
    
    # Train text vectorizer
    history = trainer.train_text_vectorizer(
        texts=texts,
        labels=labels,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print(f"Text vectorizer training complete. Final accuracy: {history['accuracy'][-1]:.4f}")


def train_numerical_vectorizer(trainer, train_loader, args):
    """Train just the numerical vectorization module."""
    print("Training numerical vectorization module...")
    
    # Collect numerical data
    all_market_data = []
    
    for batch in train_loader:
        _, market_data, _ = extract_batch_data(batch)
        all_market_data.append(market_data)
    
    # Combine all market data
    all_market_data = torch.cat(all_market_data, dim=0)
    
    # Create dataset and dataloader for autoencoder training
    market_dataset = torch.utils.data.TensorDataset(all_market_data)
    market_loader = torch.utils.data.DataLoader(
        market_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Train numerical vectorizer
    history = trainer.train_num_vectorizer(
        dataloader=market_loader,
        num_epochs=args.epochs
    )
    
    print(f"Numerical vectorizer training complete. Final loss: {history['loss'][-1]:.6f}")


def train_alignment_layer(trainer, train_loader, args):
    """Train just the alignment integration layer."""
    print("Training alignment integration layer...")
    
    # Process batches and collect vectors
    text_vectors_list = []
    num_vectors_list = []
    
    for batch in train_loader:
        # Extract data
        texts, market_data, _ = extract_batch_data(batch)
        
        # Process text through text vectorizer
        text_vectors = trainer.system.process_text(texts)
        
        # Process market data through numerical vectorizer
        num_vectors, _ = trainer.system.process_market_data(
            pd.DataFrame(market_data.cpu().numpy())
        )
        
        # Store vectors
        text_vectors_list.append(text_vectors)
        num_vectors_list.append(num_vectors)
   