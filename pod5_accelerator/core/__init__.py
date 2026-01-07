"""
Core module for POD5 file readers and signal processing utilities.

This module contains:
- AcceleratedPOD5Reader: Multi-threaded optimized reader
- BaselinePOD5Reader: Single-threaded baseline reader
- SignalProcessor: Nanopore signal processing utilities
- SyntheticPOD5Generator: Test data generation tools
"""

from pod5_accelerator.core.accelerated_reader import AcceleratedPOD5Reader
from pod5_accelerator.core.baseline_reader import BaselinePOD5Reader
from pod5_accelerator.core.signal_processor import SignalProcessor
from pod5_accelerator.core.synthetic_generator import SyntheticPOD5Generator

__all__ = [
    "AcceleratedPOD5Reader",
    "BaselinePOD5Reader",
    "SignalProcessor",
    "SyntheticPOD5Generator",
]
