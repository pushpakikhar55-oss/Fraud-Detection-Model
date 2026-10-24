"""
Features module: Behavioral, velocity, temporal, and risk-signal feature engineering.
"""
from src.features.engineer import FeatureEngineer
from src.features.registry import FEATURE_METADATA

__all__ = ["FeatureEngineer", "FEATURE_METADATA"]
