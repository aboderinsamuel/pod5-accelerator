"""
POD5 Accelerator - High-performance POD5 file reader for Oxford Nanopore sequencing data.

This package provides optimized readers for POD5 files with multi-threading
and zero-copy operations to achieve 40%+ throughput improvements.

Components:
----------
- core: Core reader implementations and utilities
- benchmarks: Performance benchmarking suite
- tests: Comprehensive unit tests

Key Classes:
-----------
- AcceleratedPOD5Reader: Multi-threaded optimized reader
- BaselinePOD5Reader: Single-threaded baseline reader
- POD5Benchmark: Performance benchmarking and comparison
- SignalProcessor: Nanopore signal processing utilities
- SyntheticPOD5Generator: Test data generation
"""

__version__ = "0.1.0"
__author__ = "POD5 Accelerator Team"

from pod5_accelerator.core.accelerated_reader import AcceleratedPOD5Reader
from pod5_accelerator.core.baseline_reader import BaselinePOD5Reader
from pod5_accelerator.core.signal_processor import SignalProcessor
from pod5_accelerator.core.synthetic_generator import SyntheticPOD5Generator
from pod5_accelerator.benchmarks.benchmark import POD5Benchmark

__all__ = [
    "AcceleratedPOD5Reader",
    "BaselinePOD5Reader",
    "SignalProcessor",
    "SyntheticPOD5Generator",
    "POD5Benchmark",
]
