"""
Text Vectorization Module for VectorFin

This module transforms financial text data into sentiment-enriched vector
representations using finance-tuned transformer models.
"""

from .model import FinancialTextVectorizer, FinancialTextProcessor

__all__ = ['FinancialTextVectorizer', 'FinancialTextProcessor']