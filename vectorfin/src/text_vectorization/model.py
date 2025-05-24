"""
Text Vectorization Module for VectorFin

This module transforms financial text data into sentiment-enriched vector
representations using finance-tuned transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Union, Optional, Tuple


class FinancialTextVectorizer(nn.Module):
    """
    Transforms financial text into sentiment-enriched vector representations.
    
    This module uses a finance-tuned transformer model (like FinBERT) to extract
    contextual embeddings, augments them with sentiment information, and projects
    them into a shared vector space compatible with the numerical data vectors.
    """
    
    def __init__(
        self,
        pretrained_model_name: str = "yiyanghkust/finbert-tone",
        vector_dim: int = 128,
        max_length: int = 512,
        sentiment_dim: int = 16,
        device: Optional[str] = None
    ):
        """
        Initialize the financial text vectorizer.
        
        Args:
            pretrained_model_name: The name of the pretrained transformer model
            vector_dim: The dimension of the output vector
            max_length: The maximum length of input text
            sentiment_dim: The dimension of the explicit sentiment features
            device: The device to use (cpu or cuda)
        """
        super().__init__()
        
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained model and tokenizer
        self.transformer = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        
        # Get the transformer's embedding dimension
        self.transformer_dim = self.transformer.config.hidden_size
        
        # Sentiment augmentation layer
        self.sentiment_layer = nn.Linear(self.transformer_dim, sentiment_dim)
        
        # Dimension reduction layer for projecting into the shared vector space
        self.projection_layer = nn.Sequential(
            nn.Linear(self.transformer_dim + sentiment_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, vector_dim)
        )
        
        # Configuration
        self.max_length = max_length
        self.vector_dim = vector_dim
        self.sentiment_dim = sentiment_dim
        
        # Move model to device
        self.to(self.device)
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Transform financial texts into vector representations.
        
        Args:
            texts: A list of financial texts to vectorize
            
        Returns:
            A tensor of shape (batch_size, vector_dim) containing the vector 
            representations of the input texts
        """
        # Tokenize the texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Get transformer embeddings
        with torch.no_grad():
            outputs = self.transformer(**inputs)
        
        # Extract [CLS] token embedding (sentence representation)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Generate sentiment features
        sentiment_features = self.sentiment_layer(cls_embeddings)
        
        # Concatenate embeddings and sentiment features
        combined_features = torch.cat([cls_embeddings, sentiment_features], dim=1)
        
        # Project to the shared vector space
        vectors = self.projection_layer(combined_features)
        
        # Normalize vectors
        normalized_vectors = F.normalize(vectors, p=2, dim=1)
        
        return normalized_vectors
    
    def extract_sentiment(self, texts: List[str]) -> torch.Tensor:
        """
        Extract explicit sentiment features from texts.
        
        Args:
            texts: A list of financial texts
            
        Returns:
            A tensor of shape (batch_size, sentiment_dim) containing 
            the sentiment features
        """
        # Tokenize the texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Get transformer embeddings
        with torch.no_grad():
            outputs = self.transformer(**inputs)
        
        # Extract [CLS] token embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Generate sentiment features
        sentiment_features = self.sentiment_layer(cls_embeddings)
        
        return sentiment_features
    
    def fine_tune(
        self,
        texts: List[str],
        labels: List[int],
        learning_rate: float = 1e-5,
        num_epochs: int = 3,
        batch_size: int = 16
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the text vectorizer on a sentiment classification task.
        
        Args:
            texts: A list of financial texts
            labels: A list of sentiment labels (0: negative, 1: neutral, 2: positive)
            learning_rate: The learning rate for optimization
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary containing training history (loss and accuracy)
        """
        # Create classification head
        classification_head = nn.Linear(self.vector_dim, 3).to(self.device)
        
        # Create optimizer
        params = list(self.transformer.parameters()) + list(self.parameters())
        optimizer = torch.optim.AdamW(params, lr=learning_rate)
        
        # Convert labels to tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        # Training history
        history = {
            "loss": [],
            "accuracy": []
        }
        
        # Set model to training mode
        self.train()
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                end_idx = min(i + batch_size, len(texts))
                
                batch_texts = texts[i:end_idx]
                batch_labels = labels_tensor[i:end_idx]
                
                # Forward pass
                vectors = self(batch_texts)
                logits = classification_head(vectors)
                
                # Calculate loss
                loss = F.cross_entropy(logits, batch_labels)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == batch_labels).sum().item()
                total += batch_labels.size(0)
            
            # Record average epoch loss and accuracy
            avg_loss = epoch_loss / (len(texts) / batch_size)
            accuracy = correct / total if total > 0 else 0
            
            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
        
        # Set model back to evaluation mode
        self.eval()
        
        return history


class FinancialTextProcessor:
    """
    Processes financial text data for analysis.
    
    This class provides methods for cleaning, normalizing, and preparing
    financial text data for vectorization.
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize a text.
        
        Args:
            text: The text to clean
            
        Returns:
            The cleaned text
        """
        # Basic cleaning
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def batch_clean(texts: List[str]) -> List[str]:
        """
        Clean a batch of texts.
        
        Args:
            texts: A list of texts to clean
            
        Returns:
            A list of cleaned texts
        """
        return [FinancialTextProcessor.clean_text(text) for text in texts]


# Example usage
if __name__ == "__main__":
    # Sample texts
    texts = [
        "The company reported better than expected earnings, raising their guidance for the next quarter.",
        "The stock plummeted after the CEO announced his resignation amid fraud allegations.",
        "The market remained stable despite concerns about inflation."
    ]
    
    # Create vectorizer
    vectorizer = FinancialTextVectorizer()
    
    # Clean texts
    cleaned_texts = FinancialTextProcessor.batch_clean(texts)
    
    # Get vector representations
    vectors = vectorizer(cleaned_texts)
    print(f"Text vectors shape: {vectors.shape}")
    
    # Extract sentiment
    sentiment = vectorizer.extract_sentiment(cleaned_texts)
    print(f"Sentiment features shape: {sentiment.shape}")
