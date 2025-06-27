#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VectorFin Training Example

This script provides a step-by-step demonstration of training the VectorFin system,
from data preparation to model evaluation, with detailed explanations.
"""

import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add parent directory to path so we can import the vectorfin package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import VectorFin components
from vectorfin.src.models.vectorfin import VectorFinSystem, VectorFinTrainer
from vectorfin.src.data.data_loader import FinancialTextData, MarketData, AlignedFinancialDataset
from vectorfin.src.text_vectorization import FinancialTextProcessor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Example training for the VectorFin system.")
    
    # Training mode
    parser.add_argument("--mode", type=str, default="end-to-end",
                        choices=["text-only", "numerical-only", "alignment-only", 
                                 "prediction-only", "end-to-end", "progressive"],
                        help="Training mode")
    
    # Data parameters
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT,GOOGL",
                        help="Comma-separated list of ticker symbols")
    parser.add_argument("--start_date", type=str, default="2020-01-01",
                        help="Start date for market data (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2023-01-01",
                        help="End date for market data (YYYY-MM-DD)")
    parser.add_argument("--news_data", type=str, default=None,
                        help="Path to CSV file with financial news data")
    parser.add_argument("--use_synthetic_news", action="store_true",
                        help="Generate synthetic news if no news data is provided")
    
    # Model parameters
    parser.add_argument("--vector_dim", type=int, default=128,
                        help="Dimension of vectors in the shared space")
    parser.add_argument("--sentiment_dim", type=int, default=16,
                        help="Dimension of sentiment features")
    parser.add_argument("--fusion_dim", type=int, default=128,
                        help="Dimension of fused vectors")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for optimizers")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for optimizers")
    parser.add_argument("--train_test_split", type=float, default=0.8,
                        help="Proportion of data to use for training")
    parser.add_argument("--prediction_horizon", type=int, default=5,
                        help="Number of days ahead to predict")
    
    # Output parameters
    parser.add_argument("--models_dir", type=str, default="./trained_models",
                        help="Directory to save trained models")
    parser.add_argument("--plot_training", action="store_true",
                        help="Plot training history")
    
    return parser.parse_args()

def fetch_and_prepare_data(args):
    """
    Fetch and prepare market and news data for training.
    
    Returns:
        tuple: (market_data, news_data, train_loader, test_loader)
    """
    print("Step 1: Fetching and preparing data")
    print("===================================")
    
    # 1. Fetch market data
    print("\nFetching market data...")
    tickers = args.tickers.split(',')
    market_data = MarketData.fetch_market_data(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # 2. Load or generate news data
    if args.news_data and Path(args.news_data).exists():
        print("\nLoading news data...")
        news_data = FinancialTextData.load_news_data(args.news_data)
    elif args.use_synthetic_news:
        print("\nGenerating synthetic news data...")
        news_data = generate_synthetic_news(args)
    else:
        raise ValueError("Either provide --news_data or use --use_synthetic_news")
    
    # 3. Prepare dataset
    print("\nPreparing aligned dataset...")
    
    # Align market data (combine data from different tickers)
    aligned_market_data = MarketData.align_market_data(market_data)
    
    # Preprocess news data
    processed_news = FinancialTextData.preprocess_text_data(news_data)
    
    # Create aligned dataset
    dataset = AlignedFinancialDataset(
        text_data=processed_news,
        market_data=aligned_market_data,
        text_column='headline',
        date_column='date',
        prediction_horizon=args.prediction_horizon,
        max_texts_per_day=10
    )
    
    # 4. Split into train and test sets
    print("\nSplitting into training and testing sets...")
    train_size = int(len(dataset) * args.train_test_split)
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    # 5. Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    return market_data, news_data, train_loader, test_loader

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
    
    print(f"Generated {len(news_data)} synthetic news items")
    return news_data

def create_vectorfin_system(args):
    """Create and initialize the VectorFin system."""
    print("\nStep 2: Creating VectorFin system")
    print("===============================")
    
    # Create the system with specified parameters
    system = VectorFinSystem(
        vector_dim=args.vector_dim,
        sentiment_dim=args.sentiment_dim,
        fusion_dim=args.fusion_dim,
        device=None,  # Auto-detect
        models_dir=args.models_dir
    )
    
    # Create the trainer
    trainer = VectorFinTrainer(
        system=system,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create models directory
    Path(args.models_dir).mkdir(exist_ok=True, parents=True)
    
    return system, trainer

def train_text_vectorization_module(trainer, train_loader, args):
    """Train the text vectorization module."""
    print("\nStep 3a: Training Text Vectorization Module")
    print("=========================================")
    
    # Collect text data and sentiment labels
    texts = []
    labels = []
    
    print("Extracting text samples from training data...")
    for i, batch in enumerate(train_loader):
        # Extract texts from batch
        batch_texts = batch["texts"]
        
        # Flatten list of lists if necessary
        if isinstance(batch_texts[0], list):
            batch_texts = [item for sublist in batch_texts for item in sublist]
        
        # Clean texts
        cleaned_texts = FinancialTextProcessor.batch_clean(batch_texts)
        texts.extend(cleaned_texts)
        
        # Generate sentiment labels based on synthetic data
        # In a real implementation, you would use actual sentiment labels
        batch_labels = []
        for text in cleaned_texts:
            # Simple heuristic: check for positive and negative words
            text_lower = text.lower()
            if any(word in text_lower for word in ['strong', 'beat', 'rise', 'positive', 'growth']):
                batch_labels.append(2)  # Positive
            elif any(word in text_lower for word in ['miss', 'decline', 'down', 'negative', 'concern']):
                batch_labels.append(0)  # Negative
            else:
                batch_labels.append(1)  # Neutral
        
        labels.extend(batch_labels)
        
        if i >= 30:  # Limit samples for demonstration
            break
    
    print(f"Collected {len(texts)} text samples for training")
    
    # Train text vectorizer
    print("\nTraining text vectorizer...")
    history = trainer.train_text_vectorizer(
        texts=texts,
        labels=labels,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Plot training history if requested
    if args.plot_training and 'accuracy' in history:
        plt.figure(figsize=(10, 5))
        plt.plot(history['accuracy'], label='Accuracy')
        plt.plot(history['loss'], label='Loss')
        plt.title('Text Vectorizer Training')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{args.models_dir}/text_vectorizer_training.png")
    
    print(f"Text vectorizer training complete. Final accuracy: {history['accuracy'][-1]:.4f}")
    return history

def train_numerical_vectorization_module(trainer, train_loader, args):
    """Train the numerical vectorization module."""
    print("\nStep 3b: Training Numerical Vectorization Module")
    print("=============================================")
    
    # Collect numerical data
    all_market_data = []
    
    print("Extracting market data from training data...")
    for batch in train_loader:
        # Extract market data
        market_data = batch["market_data"]
        if isinstance(market_data, torch.Tensor):
            market_data = market_data.float()
        else:
            market_data = torch.tensor(market_data, dtype=torch.float32)
        
        all_market_data.append(market_data)
    
    # Combine all market data
    all_market_data = torch.cat(all_market_data, dim=0)
    
    print(f"Collected {len(all_market_data)} market data samples for training")
    
    # Create dataset and dataloader for autoencoder training
    market_dataset = torch.utils.data.TensorDataset(all_market_data)
    market_loader = torch.utils.data.DataLoader(
        market_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Train numerical vectorizer
    print("\nTraining numerical vectorizer...")
    history = trainer.train_num_vectorizer(
        dataloader=market_loader,
        num_epochs=args.epochs
    )
    
    # Plot training history if requested
    if args.plot_training and 'loss' in history:
        plt.figure(figsize=(10, 5))
        plt.plot(history['loss'], label='Reconstruction Loss')
        plt.title('Numerical Vectorizer Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{args.models_dir}/num_vectorizer_training.png")
    
    print(f"Numerical vectorizer training complete. Final loss: {history['loss'][-1]:.6f}")
    return history

def train_alignment_integration_layer(trainer, train_loader, args):
    """Train the alignment integration layer."""
    print("\nStep 3c: Training Alignment Integration Layer")
    print("==========================================")
    
    # Process batches and collect vectors
    text_vectors_list = []
    num_vectors_list = []
    
    print("Generating text and numerical vectors...")
    for batch in train_loader:
        # Extract texts
        texts = batch["texts"]
        if isinstance(texts[0], list):
            texts = [item for sublist in texts for item in sublist]
        cleaned_texts = FinancialTextProcessor.batch_clean(texts)
        
        # Process text through text vectorizer
        text_vectors = trainer.system.process_text(cleaned_texts)
        
        # Extract market data
        market_data = batch["market_data"]
        if isinstance(market_data, torch.Tensor):
            market_data = market_data.float()
        else:
            market_data = torch.tensor(market_data, dtype=torch.float32)
        
        # Process market data through numerical vectorizer
        num_vectors, _ = trainer.system.process_market_data(
            pd.DataFrame(market_data.cpu().numpy())
        )
        
        # Store vectors
        text_vectors_list.append(text_vectors)
        num_vectors_list.append(num_vectors)
    
    # Combine all vectors
    all_text_vectors = torch.cat(text_vectors_list, dim=0)
    all_num_vectors = torch.cat(num_vectors_list, dim=0)
    
    print(f"Collected {len(all_text_vectors)} text vectors and {len(all_num_vectors)} numerical vectors")
    
    # Train alignment layer
    print("\nTraining alignment layer...")
    history = trainer.train_alignment_layer(
        text_vectors=all_text_vectors,
        num_vectors=all_num_vectors,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Plot training history if requested
    if args.plot_training and 'loss' in history:
        plt.figure(figsize=(10, 5))
        plt.plot(history['loss'], label='Contrastive Loss')
        plt.title('Alignment Layer Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{args.models_dir}/alignment_layer_training.png")
    
    print(f"Alignment layer training complete. Final loss: {history['loss'][-1]:.6f}")
    return history

def train_prediction_heads(trainer, train_loader, args):
    """Train the prediction heads."""
    print("\nStep 3d: Training Prediction Heads")
    print("===============================")
    
    # Process batches and collect unified vectors and targets
    unified_vectors_list = []
    targets_list = []
    
    print("Generating unified vectors and extracting targets...")
    for batch in train_loader:
        # Extract texts
        texts = batch["texts"]
        if isinstance(texts[0], list):
            texts = [item for sublist in texts for item in sublist]
        cleaned_texts = FinancialTextProcessor.batch_clean(texts)
        
        # Process text through text vectorizer
        text_vectors = trainer.system.process_text(cleaned_texts)
        
        # Extract market data
        market_data = batch["market_data"]
        if isinstance(market_data, torch.Tensor):
            market_data = market_data.float()
        else:
            market_data = torch.tensor(market_data, dtype=torch.float32)
        
        # Process market data through numerical vectorizer
        num_vectors, _ = trainer.system.process_market_data(
            pd.DataFrame(market_data.cpu().numpy())
        )
        
        # Combine vectors
        unified_vectors = trainer.system.combine_vectors(text_vectors, num_vectors)
        
        # Extract targets if available
        targets = {}
        if "target" in batch:
            target_data = batch["target"]
            if isinstance(target_data, torch.Tensor):
                target_data = target_data.float()
            else:
                target_data = torch.tensor(target_data, dtype=torch.float32)
                
            # Create different target types
            # Direction: binary (up/down)
            targets["direction"] = (target_data[:, 3] > market_data[:, 3]).float().unsqueeze(1)
            
            # Magnitude: percentage change
            targets["magnitude"] = ((target_data[:, 3] - market_data[:, 3]) / market_data[:, 3]).unsqueeze(1)
            
            # Volatility: approximated by high-low range
            targets["volatility"] = ((target_data[:, 1] - target_data[:, 2]) / target_data[:, 3]).unsqueeze(1)
            
            # Timing: simplified to just use the prediction horizon day
            horizon_day = torch.zeros(len(market_data), 30)
            for i in range(len(market_data)):
                horizon_day[i, 0] = 1  # Set day 0 as the target day
            targets["timing"] = horizon_day
            
            # Store vectors and targets
            unified_vectors_list.append(unified_vectors)
            targets_list.append(targets)
    
    # Combine all unified vectors
    all_unified_vectors = torch.cat(unified_vectors_list, dim=0)
    
    # Combine all targets
    all_targets = {
        "direction": torch.cat([t["direction"] for t in targets_list], dim=0),
        "magnitude": torch.cat([t["magnitude"] for t in targets_list], dim=0),
        "volatility": torch.cat([t["volatility"] for t in targets_list], dim=0),
        "timing": torch.cat([t["timing"] for t in targets_list], dim=0)
    }
    
    print(f"Collected {len(all_unified_vectors)} unified vectors with targets")
    
    # Train prediction heads
    print("\nTraining prediction heads...")
    history = trainer.train_prediction_heads(
        unified_vectors=all_unified_vectors,
        labels=all_targets,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Plot training history if requested
    if args.plot_training:
        plt.figure(figsize=(12, 8))
        for key in history:
            if history[key]:
                plt.plot(history[key], label=key)
        plt.title('Prediction Heads Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{args.models_dir}/prediction_heads_training.png")
    
    print("Prediction heads training complete.")
    for key in history:
        if history[key]:
            print(f"Final {key}: {history[key][-1]:.6f}")
    
    return history

def train_end_to_end(trainer, train_loader, test_loader, args):
    """Train the entire system end-to-end."""
    print("\nStep 3: Training End-to-End")
    print("=========================")
    
    # Create dataset wrapper for end-to-end training
    class AlignedDatasetWrapper(torch.utils.data.Dataset):
        """Wrapper for dataloader to be used as a dataset."""
        
        def __init__(self, dataloader):
            self.dataloader = dataloader
            self.data = []
            
            # Cache all data
            print("Preparing dataset for end-to-end training...")
            for batch in dataloader:
                for i in range(len(batch["texts"])):
                    sample = {key: batch[key][i] for key in batch if isinstance(batch[key], (list, torch.Tensor))}
                    self.data.append(sample)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    # Create dataset wrapper
    train_dataset = AlignedDatasetWrapper(train_loader)
    
    print(f"Prepared dataset with {len(train_dataset)} samples")
    
    # Train end-to-end
    print("\nStarting end-to-end training...")
    history = trainer.train_end_to_end(
        dataset=train_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.models_dir
    )
    
    # Plot training history if requested
    if args.plot_training:
        # Plot total loss
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(history["total_loss"])
        plt.title("Total Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        
        # Plot component losses
        plt.subplot(2, 2, 2)
        plt.plot(history["text_loss"], label="Text")
        plt.plot(history["num_loss"], label="Numerical")
        plt.plot(history["alignment_loss"], label="Alignment")
        plt.title("Component Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Plot prediction loss
        plt.subplot(2, 2, 3)
        plt.plot(history["prediction_loss"])
        plt.title("Prediction Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{args.models_dir}/end_to_end_training.png")
    
    print("End-to-end training complete!")
    return history

def progressive_training(trainer, train_loader, test_loader, args):
    """Perform progressive training of all components."""
    print("\nStep 3: Progressive Training")
    print("==========================")
    print("Training components in sequence for optimal integration")
    
    # 1. Train text vectorizer
    text_history = train_text_vectorization_module(trainer, train_loader, args)
    
    # 2. Train numerical vectorizer
    num_history = train_numerical_vectorization_module(trainer, train_loader, args)
    
    # 3. Train alignment layer
    alignment_history = train_alignment_integration_layer(trainer, train_loader, args)
    
    # 4. Train prediction heads
    prediction_history = train_prediction_heads(trainer, train_loader, args)
    
    # 5. Final end-to-end fine-tuning with reduced learning rate
    original_lr = args.learning_rate
    args.learning_rate = original_lr / 5  # Reduce learning rate for fine-tuning
    
    # Update trainer with new learning rate
    trainer = VectorFinTrainer(
        system=trainer.system,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    print("\nFine-tuning with end-to-end training...")
    final_history = train_end_to_end(trainer, train_loader, test_loader, args)
    
    # Restore original learning rate
    args.learning_rate = original_lr
    
    return {
        "text_history": text_history,
        "num_history": num_history,
        "alignment_history": alignment_history,
        "prediction_history": prediction_history,
        "final_history": final_history
    }

def evaluate_system(system, test_loader):
    """Evaluate the trained system."""
    print("\nStep 4: Evaluating Trained System")
    print("==============================")
    
    # Metrics
    direction_correct = 0
    direction_total = 0
    magnitude_mse = 0
    volatility_mse = 0
    
    print("Evaluating on test data...")
    
    # Process each batch
    for batch in test_loader:
        # Extract texts
        texts = batch["texts"]
        if isinstance(texts[0], list):
            texts = [item for sublist in texts for item in sublist]
        cleaned_texts = FinancialTextProcessor.batch_clean(texts)
        
        # Extract market data
        market_data = batch["market_data"]
        if isinstance(market_data, torch.Tensor):
            market_data = market_data.float().cpu().numpy()
        
        # Analyze with the system
        analysis = system.analyze_text_and_market(cleaned_texts, pd.DataFrame(market_data))
        
        # Extract predictions
        predictions = analysis["predictions"]
        
        # Extract targets if available
        if "target" in batch:
            target_data = batch["target"]
            if isinstance(target_data, torch.Tensor):
                target_data = target_data.float()
            else:
                target_data = torch.tensor(target_data, dtype=torch.float32)
            
            # Create different target types
            # Direction: binary (up/down)
            direction_true = (target_data[:, 3] > torch.tensor(market_data[:, 3])).float().unsqueeze(1)
            
            # Magnitude: percentage change
            magnitude_true = ((target_data[:, 3] - torch.tensor(market_data[:, 3])) / 
                             torch.tensor(market_data[:, 3])).unsqueeze(1)
            
            # Volatility: approximated by high-low range
            volatility_true = ((target_data[:, 1] - target_data[:, 2]) / 
                              target_data[:, 3]).unsqueeze(1)
            
            # Calculate direction accuracy
            direction_pred = predictions["direction"].cpu() > 0.5
            direction_true = direction_true.cpu() > 0.5
            direction_correct += (direction_pred == direction_true).sum().item()
            direction_total += direction_true.size(0)
            
            # Calculate magnitude MSE
            magnitude_mse += torch.nn.functional.mse_loss(
                predictions["magnitude"].cpu(), 
                magnitude_true.cpu()
            ).item() * len(cleaned_texts)
            
            # Calculate volatility MSE
            volatility_mse += torch.nn.functional.mse_loss(
                predictions["volatility"].cpu(), 
                volatility_true.cpu()
            ).item() * len(cleaned_texts)
    
    # Calculate final metrics
    direction_accuracy = direction_correct / direction_total if direction_total > 0 else 0
    magnitude_mse /= direction_total if direction_total > 0 else 1
    volatility_mse /= direction_total if direction_total > 0 else 1
    
    print(f"Direction Prediction Accuracy: {direction_accuracy:.4f}")
    print(f"Magnitude Prediction MSE: {magnitude_mse:.6f}")
    print(f"Volatility Prediction MSE: {volatility_mse:.6f}")
    
    return {
        "direction_accuracy": direction_accuracy,
        "magnitude_mse": magnitude_mse,
        "volatility_mse": volatility_mse
    }

def main():
    """Main function."""
    print("VectorFin Training Example")
    print("=========================")
    
    # Parse arguments
    args = parse_args()
    
    # Fetch and prepare data
    market_data, news_data, train_loader, test_loader = fetch_and_prepare_data(args)
    
    # Create VectorFin system
    system, trainer = create_vectorfin_system(args)
    
    # Train based on selected mode
    if args.mode == "text-only":
        train_text_vectorization_module(trainer, train_loader, args)
    
    elif args.mode == "numerical-only":
        train_numerical_vectorization_module(trainer, train_loader, args)
    
    elif args.mode == "alignment-only":
        train_alignment_integration_layer(trainer, train_loader, args)
    
    elif args.mode == "prediction-only":
        train_prediction_heads(trainer, train_loader, args)
    
    elif args.mode == "progressive":
        progressive_training(trainer, train_loader, test_loader, args)
    
    else:  # end-to-end
        train_end_to_end(trainer, train_loader, test_loader, args)
    
    # Save the trained system
    system.save_models(args.models_dir)
    print(f"\nSaved trained models to {args.models_dir}")
    
    # Evaluate system
    metrics = evaluate_system(system, test_loader)
    
    print("\nTraining complete!")
    print(f"Model saved to: {args.models_dir}")
    print("\nNext steps:")
    print("1. Use the trained model for inference with examples/inference.py")
    print("2. Fine-tune specific components with component-wise training")
    print("3. Explore the vector space with the SemanticNavigator component")

if __name__ == "__main__":
    main()
