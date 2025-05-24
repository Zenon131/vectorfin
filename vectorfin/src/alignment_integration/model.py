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
        
        # Query, key, value projections for each head
        self.q_proj = nn.Linear(vector_dim, vector_dim)
        self.k_proj = nn.Linear(vector_dim, vector_dim)
        self.v_proj = nn.Linear(vector_dim, vector_dim)
        
        # Output projection
        self.output_proj = nn.Linear(vector_dim, vector_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Move to device
        self.to(self.device)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-modal attention between query and key-value pairs.
        
        Args:
            query: Tensor of shape (batch_size, seq_len_q, vector_dim)
            key: Tensor of shape (batch_size, seq_len_k, vector_dim)
            value: Tensor of shape (batch_size, seq_len_v, vector_dim)
            attention_mask: Optional mask of shape (batch_size, seq_len_q, seq_len_k)
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        batch_size = query.size(0)
        
        # Project query, key, value
        q = self.q_proj(query)  # (batch_size, seq_len_q, vector_dim)
        k = self.k_proj(key)    # (batch_size, seq_len_k, vector_dim)
        v = self.v_proj(value)  # (batch_size, seq_len_v, vector_dim)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        # (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for multiple heads
            expanded_mask = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(expanded_mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        # (batch_size, num_heads, seq_len_q, head_dim)
        context = torch.matmul(attn_weights, v)
        
        # Reshape back to (batch_size, seq_len_q, vector_dim)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.vector_dim)
        
        # Project to final output
        output = self.output_proj(context)
        
        return output, attn_weights


class AlignmentIntegrationLayer(nn.Module):
    """
    Aligns and integrates text and numerical vectors in a unified space.
    
    This module ensures that vectors from different modalities are properly
    aligned and can be meaningfully combined and compared.
    """
    
    def __init__(
        self,
        vector_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        device: Optional[str] = None
    ):
        """
        Initialize the alignment integration layer.
        
        Args:
            vector_dim: Dimension of the vectors
            hidden_dim: Dimension of hidden layers
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            device: Device to use (cpu or cuda)
        """
        super().__init__()
        
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store parameters
        self.vector_dim = vector_dim
        
        # Cross-modal attention
        self.text_to_num_attention = CrossModalAttention(
            vector_dim=vector_dim,
            num_heads=num_heads,
            dropout=dropout,
            device=device
        )
        
        self.num_to_text_attention = CrossModalAttention(
            vector_dim=vector_dim,
            num_heads=num_heads,
            dropout=dropout,
            device=device
        )
        
        # Feed-forward networks for alignment
        self.text_alignment_ffn = nn.Sequential(
            nn.Linear(vector_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vector_dim)
        )
        
        self.num_alignment_ffn = nn.Sequential(
            nn.Linear(vector_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vector_dim)
        )
        
        # Layer normalization
        self.text_norm1 = nn.LayerNorm(vector_dim)
        self.text_norm2 = nn.LayerNorm(vector_dim)
        self.num_norm1 = nn.LayerNorm(vector_dim)
        self.num_norm2 = nn.LayerNorm(vector_dim)
        
        # Move to device
        self.to(self.device)
    
    def forward(
        self,
        text_vectors: torch.Tensor,
        num_vectors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align and integrate text and numerical vectors.
        
        Args:
            text_vectors: Tensor of shape (batch_size, seq_len_text, vector_dim)
            num_vectors: Tensor of shape (batch_size, seq_len_num, vector_dim)
            
        Returns:
            Tuple of (aligned text vectors, aligned numerical vectors)
        """
        # Cross-modal attention: text attending to numerical
        text_num_context, _ = self.text_to_num_attention(
            query=text_vectors,
            key=num_vectors,
            value=num_vectors
        )
        
        # Residual connection and normalization
        text_vectors_updated = self.text_norm1(text_vectors + text_num_context)
        
        # Concatenate original text vectors with context
        text_combined = torch.cat([text_vectors_updated, text_num_context], dim=-1)
        
        # Feed-forward network
        text_ffn_output = self.text_alignment_ffn(text_combined)
        
        # Residual connection and normalization
        aligned_text_vectors = self.text_norm2(text_vectors_updated + text_ffn_output)
        
        # Cross-modal attention: numerical attending to text
        num_text_context, _ = self.num_to_text_attention(
            query=num_vectors,
            key=text_vectors,
            value=text_vectors
        )
        
        # Residual connection and normalization
        num_vectors_updated = self.num_norm1(num_vectors + num_text_context)
        
        # Concatenate original numerical vectors with context
        num_combined = torch.cat([num_vectors_updated, num_text_context], dim=-1)
        
        # Feed-forward network
        num_ffn_output = self.num_alignment_ffn(num_combined)
        
        # Residual connection and normalization
        aligned_num_vectors = self.num_norm2(num_vectors_updated + num_ffn_output)
        
        return aligned_text_vectors, aligned_num_vectors


class VectorFusionModule(nn.Module):
    """
    Fuses text and numerical vectors into a joint representation.
    
    This module combines aligned text and numerical vectors into a unified
    representation that captures information from both modalities.
    """
    
    def __init__(
        self,
        vector_dim: int = 128,
        fusion_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        fusion_method: str = 'attention',
        device: Optional[str] = None
    ):
        """
        Initialize the vector fusion module.
        
        Args:
            vector_dim: Dimension of the input vectors
            fusion_dim: Dimension of the fused vectors
            hidden_dim: Dimension of hidden layers
            dropout: Dropout probability
            fusion_method: Method to use for fusion ('concat', 'sum', 'attention')
            device: Device to use (cpu or cuda)
        """
        super().__init__()
        
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store parameters
        self.vector_dim = vector_dim
        self.fusion_dim = fusion_dim
        self.fusion_method = fusion_method
        
        # Fusion attention (for 'attention' method)
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=vector_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion projection (for all methods)
        if fusion_method == 'concat':
            self.fusion_projection = nn.Sequential(
                nn.Linear(vector_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, fusion_dim)
            )
        else:
            self.fusion_projection = nn.Sequential(
                nn.Linear(vector_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, fusion_dim)
            )
        
        # Move to device
        self.to(self.device)
    
    def forward(
        self,
        text_vectors: torch.Tensor,
        num_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse text and numerical vectors.
        
        Args:
            text_vectors: Tensor of shape (batch_size, seq_len_text, vector_dim)
            num_vectors: Tensor of shape (batch_size, seq_len_num, vector_dim)
            
        Returns:
            Tensor of shape (batch_size, fusion_dim) with fused vectors
        """
        if self.fusion_method == 'concat':
            # Average across sequence length
            text_avg = torch.mean(text_vectors, dim=1)
            num_avg = torch.mean(num_vectors, dim=1)
            
            # Concatenate
            combined = torch.cat([text_avg, num_avg], dim=1)
            
            # Project to fusion dimension
            fused_vectors = self.fusion_projection(combined)
            
        elif self.fusion_method == 'sum':
            # Average across sequence length
            text_avg = torch.mean(text_vectors, dim=1)
            num_avg = torch.mean(num_vectors, dim=1)
            
            # Sum
            combined = text_avg + num_avg
            
            # Project to fusion dimension
            fused_vectors = self.fusion_projection(combined)
            
        elif self.fusion_method == 'attention':
            # Concatenate along sequence dimension
            combined = torch.cat([text_vectors, num_vectors], dim=1)
            
            # Self-attention
            attn_output, _ = self.fusion_attention(combined, combined, combined)
            
            # Average across sequence length
            attn_avg = torch.mean(attn_output, dim=1)
            
            # Project to fusion dimension
            fused_vectors = self.fusion_projection(attn_avg)
        
        else:
            raise ValueError(f"Invalid fusion method: {self.fusion_method}")
        
        # Normalize vectors
        fused_vectors = F.normalize(fused_vectors, p=2, dim=1)
        
        return fused_vectors


class SemanticNavigator(nn.Module):
    """
    Enables navigation and search in the unified vector space.
    
    This module provides functionality for finding similar items, identifying
    correlations, and extracting insights from the unified vector space.
    """
    
    def __init__(
        self,
        vector_dim: int = 128,
        index_method: str = 'faiss',
        device: Optional[str] = None
    ):
        """
        Initialize the semantic navigator.
        
        Args:
            vector_dim: Dimension of the vectors
            index_method: Indexing method ('faiss', 'annoy', or 'exact')
            device: Device to use (cpu or cuda)
        """
        super().__init__()
        
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store parameters
        self.vector_dim = vector_dim
        self.index_method = index_method
        
        # Dictionary to store vectors and metadata
        self.vector_store = {
            'vectors': [],
            'metadata': [],
            'modality': []  # 'text', 'numerical', or 'fused'
        }
        
        # Index (initially None, created when needed)
        self.index = None
        
        # Move to device
        self.to(self.device)
    
    def add_vectors(
        self,
        vectors: torch.Tensor,
        metadata: List[Dict],
        modality: str
    ) -> None:
        """
        Add vectors to the semantic space.
        
        Args:
            vectors: Tensor of shape (batch_size, vector_dim)
            metadata: List of metadata dictionaries for each vector
            modality: The modality of the vectors ('text', 'numerical', or 'fused')
        """
        # Convert to numpy for storage
        vectors_np = vectors.detach().cpu().numpy()
        
        # Add to store
        self.vector_store['vectors'].append(vectors_np)
        self.vector_store['metadata'].extend(metadata)
        self.vector_store['modality'].extend([modality] * len(vectors_np))
        
        # Invalidate index (needs to be rebuilt)
        self.index = None
    
    def _build_index(self) -> None:
        """Build the vector index for efficient similarity search."""
        # Concatenate all vectors
        all_vectors = np.vstack(self.vector_store['vectors'])
        
        if self.index_method == 'faiss':
            try:
                import faiss
                
                # Create FAISS index
                self.index = faiss.IndexFlatL2(self.vector_dim)
                self.index.add(all_vectors.astype('float32'))
                
            except ImportError:
                print("FAISS not installed. Falling back to exact search.")
                self.index = all_vectors
                self.index_method = 'exact'
                
        elif self.index_method == 'annoy':
            try:
                from annoy import AnnoyIndex
                
                # Create Annoy index
                self.index = AnnoyIndex(self.vector_dim, 'angular')
                for i, vec in enumerate(all_vectors):
                    self.index.add_item(i, vec)
                    
                self.index.build(10)  # 10 trees
                
            except ImportError:
                print("Annoy not installed. Falling back to exact search.")
                self.index = all_vectors
                self.index_method = 'exact'
                
        else:  # 'exact' or fallback
            self.index = all_vectors
            self.index_method = 'exact'
    
    def search(
        self,
        query_vector: torch.Tensor,
        k: int = 5,
        filter_modality: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for similar vectors in the semantic space.
        
        Args:
            query_vector: Query vector of shape (vector_dim,)
            k: Number of results to return
            filter_modality: Optional filter for modality type
            
        Returns:
            List of dictionaries with results (metadata and similarity scores)
        """
        if not self.vector_store['vectors']:
            return []
            
        # Build index if needed
        if self.index is None:
            self._build_index()
            
        # Convert query to numpy
        query_np = query_vector.detach().cpu().numpy().reshape(1, -1)
        
        # Search based on indexing method
        if self.index_method == 'faiss':
            distances, indices = self.index.search(query_np.astype('float32'), k)
            distances, indices = distances[0], indices[0]
            
        elif self.index_method == 'annoy':
            indices, distances = zip(*self.index.get_nns_by_vector(
                query_np.flatten(), k, include_distances=True))
            
        else:  # 'exact'
            # Compute L2 distances
            distances = np.linalg.norm(self.index - query_np, axis=1)
            indices = np.argsort(distances)[:k]
            distances = distances[indices]
        
        # Prepare results
        results = []
        for i, (idx, dist) in enumerate(zip(indices, distances)):
            # Skip if filtering by modality and this doesn't match
            if (filter_modality is not None and 
                self.vector_store['modality'][idx] != filter_modality):
                continue
                
            # Convert distance to similarity score (1 = identical, 0 = completely different)
            similarity = 1.0 / (1.0 + float(dist))
            
            result = {
                'rank': i + 1,
                'metadata': self.vector_store['metadata'][idx],
                'similarity': similarity,
                'modality': self.vector_store['modality'][idx]
            }
            
            results.append(result)
            
        return results[:k]


# Example usage
if __name__ == "__main__":
    # Create sample vectors
    batch_size = 4
    seq_len_text = 3
    seq_len_num = 5
    vector_dim = 128
    
    # Random text and numerical vectors
    text_vectors = torch.randn(batch_size, seq_len_text, vector_dim)
    num_vectors = torch.randn(batch_size, seq_len_num, vector_dim)
    
    # Normalize
    text_vectors = F.normalize(text_vectors, p=2, dim=2)
    num_vectors = F.normalize(num_vectors, p=2, dim=2)
    
    # Create alignment integration layer
    align_layer = AlignmentIntegrationLayer(vector_dim=vector_dim)
    
    # Align vectors
    aligned_text_vectors, aligned_num_vectors = align_layer(text_vectors, num_vectors)
    
    print(f"Aligned text vectors shape: {aligned_text_vectors.shape}")
    print(f"Aligned numerical vectors shape: {aligned_num_vectors.shape}")
    
    # Create fusion module
    fusion_module = VectorFusionModule(vector_dim=vector_dim)
    
    # Fuse vectors
    fused_vectors = fusion_module(aligned_text_vectors, aligned_num_vectors)
    
    print(f"Fused vectors shape: {fused_vectors.shape}")
