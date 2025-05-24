"""
Numerical Data Module for VectorFin

This module transforms market metrics into meaningful vector representations
that capture their financial significance and can be used in the shared vector space.
"""

from .model import NumericalVectorizer, MarketDataProcessor, NumericalFeatureConfig

__all__ = ['NumericalVectorizer', 'MarketDataProcessor', 'NumericalFeatureConfig']