"""
Alignment and Integration Layer for VectorFin

This module creates a unified vector space where text and numerical vectors
can be meaningfully combined and processed together.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Union, Optional


class CrossModalAttention(nn.Module):
    """
    Implements cross-modal attention between text and numerical vectors.
    
    This module allows text vectors to attend to relevant numerical features
    and numerical vectors to be influenced by sentiment context.
    """
    
    def __init__(
        self,
        vector_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        device: Optional[str] = None
    ):
        """
        Initialize the cross-modal attention module.
        
        Args:
            vector_dim: Dimension of the input vectors
            num_heads: Number of attention heads
            dropout: Dropout probability
            device: Device to use (cpu or cuda)
        """
        super().__init__()
        
        # Check that vector_dim is divisible by num_heads
        if vector_dim % num_heads != 0:
            raise ValueError(f"vector_dim ({vector_dim}) must be divisible by num_heads ({num_heads})")
            
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store parameters
        self.vector_dim = vector_dim
        self.num_heads = num_heads
        self.head_dim = vector_dim // num_heads
        
        # Projection layers for text modality
        self.text_query = nn.Linear(vector_dim, vector_dim)
        self.text_key = nn.Linear(vector_dim, vector_dim)
        self.text_value = nn.Linear(vector_dim, vector_dim)
        
        # Projection layers for numerical modality
        self.num_query = nn.Linear(vector_dim, vector_dim)
        self.num_key = nn.Linear(vector_dim, vector_dim)
        self.num_value = nn.Linear(vector_dim, vector_dim)
        
        # Output projections
        self.text_out = nn.Linear(vector_dim, vector_dim)
        self.num_out = nn.Linear(vector_dim, vector_dim)
        
        # Normalization layers
        self.text_norm1 = nn.LayerNorm(vector_dim)
        self.text_norm2 = nn.LayerNorm(vector_dim)
        self.num_norm1 = nn.LayerNorm(vector_dim)
        self.num_norm2 = nn.LayerNorm(vector_dim)
        
        # Feedforward networks
        self.text_ffn = nn.Sequential(
            nn.Linear(vector_dim, vector_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vector_dim * 4, vector_dim)
        )
        
        self.num_ffn = nn.Sequential(
            nn.Linear(vector_dim, vector_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vector_dim * 4, vector_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Move to device
        self.to(self.device)
    
    def forward(
        self,
        text_vectors: torch.Tensor,
        num_vectors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention between text and numerical vectors.
        
        Args:
            text_vectors: Text vectors of shape (batch_size, vector_dim)
            num_vectors: Numerical vectors of shape (batch_size, vector_dim)
            
        Returns:
            Tuple of updated (text_vectors, num_vectors) after attention
        """
        batch_size = text_vectors.shape[0]
        
        # === Text attending to numerical features ===
        # Reshape for multi-head attention
        def reshape_for_multihead(x):
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Text queries attend to numerical keys
        text_q = reshape_for_multihead(self.text_query(text_vectors))
        num_k = reshape_for_multihead(self.num_key(num_vectors))
        num_v = reshape_for_multihead(self.num_value(num_vectors))
        
        # Calculate attention scores (text attending to numerical)
        text_num_attn = torch.matmul(text_q, num_k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        text_num_attn = F.softmax(text_num_attn, dim=-1)
        text_num_attn = self.dropout(text_num_attn)
        
        # Apply attention to values
        text_output = torch.matmul(text_num_attn, num_v)
        text_output = text_output.transpose(1, 2).contiguous().view(batch_size, -1, self.vector_dim)
        text_output = self.text_out(text_output).squeeze(1)
        
        # Residual connection and layer norm
        text_vectors = self.text_norm1(text_vectors + text_output)
        text_ff_output = self.text_ffn(text_vectors)
        text_vectors = self.text_norm2(text_vectors + text_ff_output)
        
        # === Numerical features attending to text ===
        # Numerical queries attend to text keys
        num_q = reshape_for_multihead(self.num_query(num_vectors))
        text_k = reshape_for_multihead(self.text_key(text_vectors))
        text_v = reshape_for_multihead(self.text_value(text_vectors))
        
        # Calculate attention scores (numerical attending to text)
        num_text_attn = torch.matmul(num_q, text_k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        num_text_attn = F.softmax(num_text_attn, dim=-1)
        num_text_attn = self.dropout(num_text_attn)
        
        # Apply attention to values
        num_output = torch.matmul(num_text_attn, text_v)
        num_output = num_output.transpose(1, 2).contiguous().view(batch_size, -1, self.vector_dim)
        num_output = self.num_out(num_output).squeeze(1)
        
        # Residual connection and layer norm
        num_vectors = self.num_norm1(num_vectors + num_output)
        num_ff_output = self.num_ffn(num_vectors)
        num_vectors = self.num_norm2(num_vectors + num_ff_output)
        
        return text_vectors, num_vectors


class VectorFusionModule(nn.Module):
    """
    Combines text and numerical vectors into a unified representation.
    
    This module learns to create a joint representation that preserves
    the important information from both modalities.
    """
    
    def __init__(
        self,
        vector_dim: int,
        fusion_dim: int = None,
        dropout: float = 0.1,
        device: Optional[str] = None
    ):
        """
        Initialize the vector fusion module.
        
        Args:
            vector_dim: Dimension of input vectors
            fusion_dim: Dimension of fused vectors (defaults to vector_dim)
            dropout: Dropout probability
            device: Device to use (cpu or cuda)
        """
        super().__init__()
        
        # Set fusion dimension if not provided
        if fusion_dim is None:
            fusion_dim = vector_dim
            
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store parameters
        self.vector_dim = vector_dim
        self.fusion_dim = fusion_dim
        
        # Learnable weights for modality importance
        self.text_weight = nn.Parameter(torch.ones(1))
        self.num_weight = nn.Parameter(torch.ones(1))
        
        # Projection layers for fusion
        self.text_proj = nn.Linear(vector_dim, fusion_dim)
        self.num_proj = nn.Linear(vector_dim, fusion_dim)
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Move to device
        self.to(self.device)
    
    def forward(
        self,
        text_vectors: torch.Tensor,
        num_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine text and numerical vectors into a unified representation.
        
        Args:
            text_vectors: Text vectors of shape (batch_size, vector_dim)
            num_vectors: Numerical vectors of shape (batch_size, vector_dim)
            
        Returns:
            Fused vectors of shape (batch_size, fusion_dim)
        """
        # Calculate normalized weights
        weights = F.softmax(torch.stack([self.text_weight, self.num_weight]), dim=0)
        text_weight, num_weight = weights[0], weights[1]
        
        # Project both modalities to fusion dimension
        text_proj = self.text_proj(text_vectors) * text_weight
        num_proj = self.num_proj(num_vectors) * num_weight
        
        # Concatenate and fuse
        concat_vectors = torch.cat([text_proj, num_proj], dim=1)
        fused_vectors = self.fusion_layer(concat_vectors)
        
        # Normalize output vectors
        fused_vectors = F.normalize(fused_vectors, p=2, dim=1)
        
        return fused_vectors
    
    def get_modality_weights(self) -> Tuple[float, float]:
        """
        Get the current weights assigned to each modality.
        
        Returns:
            Tuple of (text_weight, num_weight) after normalization
        """
        weights = F.softmax(torch.stack([self.text_weight, self.num_weight]), dim=0)
        return weights[0].item(), weights[1].item()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for aligning related concepts in the vector space.
    
    This loss function brings related text and numerical vectors closer 
    together while pushing unrelated pairs apart.
    """
    
    def __init__(self, margin: float = 0.5, temperature: float = 0.07):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for contrastive loss
            temperature: Temperature parameter for scaling
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        text_vectors: torch.Tensor,
        num_vectors: torch.Tensor,
        positive_pairs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate contrastive loss between text and numerical vectors.
        
        Args:
            text_vectors: Text vectors of shape (batch_size, vector_dim)
            num_vectors: Numerical vectors of shape (batch_size, vector_dim)
            positive_pairs: Optional tensor indicating which pairs are positive
                           If None, assumes diagonal pairs are positive
                           
        Returns:
            Contrastive loss value
        """
        # Calculate similarity matrix
        similarity = torch.matmul(text_vectors, num_vectors.transpose(0, 1)) / self.temperature
        
        # If positive_pairs not provided, assume diagonal pairs are positive
        if positive_pairs is None:
            positive_pairs = torch.eye(text_vectors.shape[0], device=text_vectors.device)
        
        # Calculate positive and negative pair losses
        pos_loss = -torch.sum(similarity * positive_pairs) / positive_pairs.sum()
        
        # Create negative mask (all non-positive pairs)
        negative_pairs = 1.0 - positive_pairs
        
        # Calculate negative pair loss with margin
        neg_similarity = similarity - self.margin
        neg_similarity = torch.clamp(neg_similarity, min=0.0)
        neg_loss = torch.sum(neg_similarity * negative_pairs) / negative_pairs.sum()
        
        # Combined loss
        loss = pos_loss + neg_loss
        
        return loss


class AlignmentIntegrationLayer(nn.Module):
    """
    Main module for aligning and integrating text and numerical vectors.
    
    This module contains the cross-modal attention mechanism and the vector fusion
    to create a unified representation in the shared vector space.
    """
    
    def __init__(
        self,
        vector_dim: int = 128,
        fusion_dim: int = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        device: Optional[str] = None
    ):
        """
        Initialize the alignment and integration layer.
        
        Args:
            vector_dim: Dimension of input vectors
            fusion_dim: Dimension of fused vectors (defaults to vector_dim)
            num_heads: Number of attention heads
            dropout: Dropout probability
            device: Device to use (cpu or cuda)
        """
        super().__init__()
        
        # Set fusion dimension if not provided
        if fusion_dim is None:
            fusion_dim = vector_dim
            
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cross-modal attention module
        self.cross_attn = CrossModalAttention(
            vector_dim=vector_dim,
            num_heads=num_heads,
            dropout=dropout,
            device=device
        )
        
        # Vector fusion module
        self.fusion = VectorFusionModule(
            vector_dim=vector_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
            device=device
        )
        
        # Store configuration
        self.vector_dim = vector_dim
        self.fusion_dim = fusion_dim
        
        # Move to device
        self.to(self.device)
    
    def forward(
        self,
        text_vectors: torch.Tensor,
        num_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Process text and numerical vectors through the alignment and integration layer.
        
        Args:
            text_vectors: Text vectors of shape (batch_size, vector_dim)
            num_vectors: Numerical vectors of shape (batch_size, vector_dim)
            
        Returns:
            Unified vectors of shape (batch_size, fusion_dim)
        """
        # Apply cross-modal attention
        text_attn, num_attn = self.cross_attn(text_vectors, num_vectors)
        
        # Fuse vectors
        unified_vectors = self.fusion(text_attn, num_attn)
        
        return unified_vectors
    
    def get_modality_weights(self) -> Tuple[float, float]:
        """
        Get the current weights assigned to each modality.
        
        Returns:
            Tuple of (text_weight, num_weight) after normalization
        """
        return self.fusion.get_modality_weights()
    
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load(cls, path: str, **kwargs) -> 'AlignmentIntegrationLayer':
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from
            **kwargs: Additional arguments for initialization
            
        Returns:
            Loaded AlignmentIntegrationLayer model
        """
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location=model.device))
        return model


class SemanticNavigator:
    """
    Tool for exploring the unified vector space.
    
    This class provides methods to navigate and query the vector space,
    finding related concepts and performing vector operations.
    """
    
    def __init__(
        self,
        vector_dim: int,
        index_type: str = "flat",
        device: Optional[str] = None
    ):
        """
        Initialize the semantic navigator.
        
        Args:
            vector_dim: Dimension of vectors in the space
            index_type: Type of index to use ('flat' or 'hnsw')
            device: Device to use (cpu or cuda)
        """
        self.vector_dim = vector_dim
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.index_type = index_type
        
        # Initialize empty vectors and metadata storage
        self.vectors = None
        self.metadata = []
        self.index = None
    
    def build_index(self, vectors: torch.Tensor, metadata: List[Dict]) -> None:
        """
        Build search index from vectors.
        
        Args:
            vectors: Tensor of vectors to index
            metadata: List of metadata dictionaries for each vector
        """
        import faiss
        
        # Convert to numpy if needed
        if isinstance(vectors, torch.Tensor):
            vectors = vectors.detach().cpu().numpy()
            
        # Store vectors and metadata
        self.vectors = vectors
        self.metadata = metadata
        
        # Create index based on type
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.vector_dim)
            self.index.add(vectors)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.vector_dim, 32)  # 32 neighbors
            self.index.add(vectors)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def search(
        self,
        query_vector: torch.Tensor,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices, metadata)
        """
        # Convert to numpy if needed
        if isinstance(query_vector, torch.Tensor):
            query_vector = query_vector.detach().cpu().numpy().reshape(1, -1)
        
        # Perform search
        distances, indices = self.index.search(query_vector, k)
        
        # Get metadata for results
        result_metadata = [self.metadata[idx] for idx in indices[0] if idx < len(self.metadata)]
        
        return distances[0], indices[0], result_metadata
    
    def find_analogies(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Find analogies in the form of a:b::c:?
        
        Args:
            a: First vector in the analogy
            b: Second vector in the analogy
            c: Third vector in the analogy
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices, metadata) for analogy results
        """
        # Convert to numpy if needed
        vectors = [a, b, c]
        for i, vec in enumerate(vectors):
            if isinstance(vec, torch.Tensor):
                vectors[i] = vec.detach().cpu().numpy().flatten()
        
        # Calculate analogy query vector
        query_vector = vectors[1] - vectors[0] + vectors[2]
        query_vector = query_vector.reshape(1, -1)
        
        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
        
        # Search for results
        return self.search(query_vector, k)


# Example usage
if __name__ == "__main__":
    # Create sample vectors
    text_vectors = torch.randn(8, 128)
    num_vectors = torch.randn(8, 128)
    
    # Normalize vectors
    text_vectors = F.normalize(text_vectors, p=2, dim=1)
    num_vectors = F.normalize(num_vectors, p=2, dim=1)
    
    # Create alignment layer
    alignment = AlignmentIntegrationLayer(vector_dim=128)
    
    # Get unified vectors
    unified_vectors = alignment(text_vectors, num_vectors)
    print(f"Unified vectors shape: {unified_vectors.shape}")
    
    # Get modality weights
    text_weight, num_weight = alignment.get_modality_weights()
    print(f"Modality weights - Text: {text_weight:.4f}, Numerical: {num_weight:.4f}")