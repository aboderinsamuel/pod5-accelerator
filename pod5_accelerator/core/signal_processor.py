"""
Signal processing utilities for nanopore sequencing data preprocessing.

This module provides the SignalProcessor class with optimized methods for processing
raw electrical current signals from Oxford Nanopore sequencing devices. These signals
represent ionic current variations as DNA/RNA molecules pass through protein nanopores.

Signal Processing Relevance to DNA Sequencing:
----------------------------------------------
Oxford Nanopore sequencing measures electrical current as nucleotides pass through
nanopores. Raw signals require preprocessing to:

1. Normalization: Standardize current levels across different pores and conditions
   - Z-score normalization centers signals around mean=0, std=1
   - Essential for consistent basecalling and comparison across runs

2. Filtering: Remove high-frequency noise from analog current measurements
   - Butterworth low-pass filter removes electrical noise
   - Preserves biological signal features (current level changes)
   - Improves signal-to-noise ratio for basecalling algorithms

3. Statistics: Characterize signal quality and data distribution
   - Mean current indicates base composition (A/T/G/C have different levels)
   - Standard deviation measures noise and signal quality
   - Median provides robust central tendency measure

4. Event Detection: Identify significant current changes (state transitions)
   - Threshold-based detection finds potential base boundaries
   - Precursor to segmentation algorithms in basecalling pipelines

Optimization Strategy:
---------------------
All functions use:
- NumPy vectorized operations: Avoid Python loops (10-100x faster)
- In-place operations: Minimize memory copies where possible
- Efficient scipy implementations: Optimized C/Fortran kernels
- Broadcasting: Leverage NumPy's broadcasting for array operations

Time Complexity Analysis:
- normalize_signal: O(n) - two passes over array
- filter_signal: O(n log n) - FFT-based filtering
- compute_statistics: O(n) - single pass with vectorized ops
- detect_events: O(n) - single pass comparison

Usage Example:
--------------
>>> signal = read_data['signal']  # Raw nanopore signal
>>> normalized = SignalProcessor.normalize_signal(signal)
>>> filtered = SignalProcessor.filter_signal(normalized)
>>> stats = SignalProcessor.compute_statistics(filtered)
>>> events = SignalProcessor.detect_events(filtered, threshold=2.5)
"""

import numpy as np
from scipy import signal
from typing import Dict, Any
import time
from functools import wraps


def timing_decorator(func):
    """
    Decorator to measure and report function execution time.
    
    Useful for profiling signal processing operations and identifying
    performance bottlenecks in the preprocessing pipeline.
    
    Args:
        func: Function to time
    
    Returns:
        Wrapped function that prints execution time
    
    Example:
    --------
    >>> @timing_decorator
    ... def my_function(data):
    ...     # process data
    ...     pass
    >>> my_function(signal_data)
    my_function executed in 0.0123 seconds
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__name__} executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper


class SignalProcessor:
    """
    Static utility class for nanopore signal processing operations.
    
    Provides optimized methods for preprocessing raw electrical current signals
    from Oxford Nanopore sequencing. All methods are static and can be called
    without instantiation.
    
    Processing Pipeline:
    -------------------
    Typical preprocessing workflow:
    1. normalize_signal: Standardize current levels (z-score)
    2. filter_signal: Remove high-frequency noise (low-pass filter)
    3. compute_statistics: Quality control metrics
    4. detect_events: Identify significant transitions
    
    All operations use NumPy vectorization for optimal performance.
    
    Example:
    --------
    >>> # Process raw nanopore signal
    >>> raw_signal = read_record.signal
    >>> 
    >>> # Normalize to z-score
    >>> normalized = SignalProcessor.normalize_signal(raw_signal)
    >>> 
    >>> # Filter noise
    >>> filtered = SignalProcessor.filter_signal(normalized, cutoff_freq=0.1)
    >>> 
    >>> # Compute quality metrics
    >>> stats = SignalProcessor.compute_statistics(filtered)
    >>> print(f"Signal SNR: {stats['mean'] / stats['std']:.2f}")
    >>> 
    >>> # Detect events
    >>> events = SignalProcessor.detect_events(filtered, threshold=2.0)
    >>> print(f"Detected {len(events)} events")
    """
    
    @staticmethod
    def normalize_signal(raw_signal: np.ndarray) -> np.ndarray:
        """
        Normalize raw signal using z-score normalization (standardization).
        
        Z-score normalization transforms signal to have mean=0 and std=1:
            normalized = (raw - mean) / std
        
        This is crucial for nanopore data because:
        - Different pores have different baseline currents
        - Temperature affects absolute current levels
        - Standardization enables comparison across experiments
        - Basecalling models expect normalized inputs
        
        Optimization:
        ------------
        - Uses NumPy vectorized operations (no Python loops)
        - Single pass to compute mean and std (efficient)
        - Broadcasting for element-wise operations
        - Returns new array (preserves input)
        
        Time Complexity: O(n) where n is signal length
        Space Complexity: O(n) for output array
        
        Args:
            raw_signal (np.ndarray): Raw electrical current measurements.
                                    Shape: (n_samples,)
                                    Units: picoamperes (pA)
        
        Returns:
            np.ndarray: Normalized signal with mean≈0, std≈1.
                       Same shape as input.
        
        Edge Cases:
        ----------
        - Zero std (constant signal): Returns zero array
        - Empty array: Returns empty array
        - NaN values: Propagate through calculation
        
        Example:
        --------
        >>> raw = np.array([95.2, 100.1, 98.5, 102.3, 99.1])
        >>> normalized = SignalProcessor.normalize_signal(raw)
        >>> print(f"Mean: {normalized.mean():.2f}, Std: {normalized.std():.2f}")
        Mean: 0.00, Std: 1.00
        
        >>> # Verify normalization
        >>> assert abs(normalized.mean()) < 1e-10
        >>> assert abs(normalized.std() - 1.0) < 1e-10
        """
        # Handle edge case: empty array
        if len(raw_signal) == 0:
            return np.array([])
        
        # Compute mean and standard deviation in one pass
        mean = np.mean(raw_signal)
        std = np.std(raw_signal)
        
        # Handle edge case: zero standard deviation (constant signal)
        if std == 0 or np.isnan(std):
            return np.zeros_like(raw_signal)
        
        # Z-score normalization: (x - μ) / σ
        # Broadcasting automatically applies to all elements
        normalized = (raw_signal - mean) / std
        
        return normalized
    
    @staticmethod
    def filter_signal(
        raw_signal: np.ndarray,
        cutoff_freq: float = 0.1,
        order: int = 4
    ) -> np.ndarray:
        """
        Apply low-pass Butterworth filter to remove high-frequency noise.
        
        Butterworth filters provide maximally flat passband response, making them
        ideal for preserving biological signal features while removing electrical
        noise from nanopore measurements.
        
        Why Filtering Matters:
        ---------------------
        - Nanopore signals contain electrical noise (thermal, Johnson noise)
        - High-frequency noise obscures true current level changes
        - Low-pass filtering preserves base transitions (low freq)
        - Improves signal quality for downstream basecalling
        
        Filter Design:
        -------------
        - Type: Butterworth (maximally flat passband)
        - Order: 4 (good balance of sharpness and stability)
        - Cutoff: 0.1 (normalized frequency, 10% of Nyquist)
        - Implementation: scipy.signal.butter + filtfilt
        
        Optimization:
        ------------
        - Uses scipy's optimized C implementation
        - filtfilt: Zero-phase filtering (no signal delay)
        - FFT-based for long signals (efficient)
        
        Time Complexity: O(n log n) for FFT-based implementation
        Space Complexity: O(n) for output array
        
        Args:
            raw_signal (np.ndarray): Raw or normalized signal.
                                    Shape: (n_samples,)
            cutoff_freq (float): Normalized cutoff frequency (0-1).
                               0.1 = 10% of Nyquist frequency.
                               Lower = more aggressive filtering.
                               Default: 0.1
            order (int): Filter order (steepness of rolloff).
                        Higher = sharper cutoff but more ringing.
                        Default: 4
        
        Returns:
            np.ndarray: Filtered signal with high-frequency noise removed.
                       Same shape as input.
        
        Edge Cases:
        ----------
        - Signal too short: Returns original (need length > 3*order)
        - Empty array: Returns empty array
        - NaN values: May propagate or cause errors
        
        Example:
        --------
        >>> # Add synthetic noise
        >>> clean_signal = np.sin(np.linspace(0, 10, 1000))
        >>> noise = np.random.normal(0, 0.5, 1000)
        >>> noisy_signal = clean_signal + noise
        >>> 
        >>> # Filter noise
        >>> filtered = SignalProcessor.filter_signal(noisy_signal, cutoff_freq=0.1)
        >>> 
        >>> # Compare noise levels
        >>> noise_before = np.std(noisy_signal - clean_signal)
        >>> noise_after = np.std(filtered - clean_signal)
        >>> print(f"Noise reduction: {(1 - noise_after/noise_before)*100:.1f}%")
        """
        # Handle edge cases
        if len(raw_signal) == 0:
            return np.array([])
        
        # Need sufficient samples for filtering (filtfilt requires 3*order)
        if len(raw_signal) <= 3 * order:
            return raw_signal.copy()
        
        # Design Butterworth low-pass filter
        # cutoff_freq is normalized (0 to 1, where 1 is Nyquist frequency)
        b, a = signal.butter(order, cutoff_freq, btype='low', analog=False)  # type: ignore
        
        # Apply zero-phase filter (forward and backward pass)
        # This avoids phase distortion that would shift signal features
        filtered = signal.filtfilt(b, a, raw_signal)  # type: ignore
        
        return filtered
    
    @staticmethod
    def compute_statistics(signal_data: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive statistical summary of signal data.
        
        Provides key metrics for signal quality assessment and characterization:
        - Central tendency: mean, median
        - Dispersion: standard deviation, range
        - Extrema: minimum, maximum
        
        These statistics are used for:
        - Quality control: High std may indicate noisy data
        - Data validation: Check expected current ranges (70-130 pA)
        - Preprocessing decisions: Guide normalization parameters
        - Comparative analysis: Compare signals across runs
        
        Optimization:
        ------------
        - All metrics computed via NumPy vectorized operations
        - Single pass over data (where possible)
        - No Python loops
        - Efficient NumPy C implementations
        
        Time Complexity: O(n) - linear scan for all metrics
        Space Complexity: O(1) - only scalar outputs
        
        Args:
            signal_data (np.ndarray): Signal to analyze.
                                     Shape: (n_samples,)
                                     Can be raw or processed signal.
        
        Returns:
            Dict[str, float]: Statistical summary containing:
                - mean: Average signal value
                - std: Standard deviation (noise level)
                - min: Minimum value
                - max: Maximum value
                - median: Middle value (robust to outliers)
                - range: max - min (signal dynamics)
        
        Edge Cases:
        ----------
        - Empty array: Returns NaN for all metrics
        - Single value: std=0, mean=median=min=max
        - Constant array: std=0, range=0
        
        Example:
        --------
        >>> signal = read_record.signal
        >>> stats = SignalProcessor.compute_statistics(signal)
        >>> 
        >>> print(f"Mean current: {stats['mean']:.2f} pA")
        >>> print(f"Signal noise (std): {stats['std']:.2f} pA")
        >>> print(f"Dynamic range: {stats['range']:.2f} pA")
        >>> 
        >>> # Quality check
        >>> if stats['std'] > 10:
        ...     print("Warning: High noise level detected")
        >>> 
        >>> # Signal-to-noise ratio
        >>> snr = stats['mean'] / stats['std']
        >>> print(f"SNR: {snr:.2f}")
        """
        # Handle edge case: empty array
        if len(signal_data) == 0:
            return {
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'median': np.nan,
                'range': np.nan,
            }
        
        # Compute statistics using vectorized NumPy operations
        # All of these are O(n) or better
        stats = {
            'mean': float(np.mean(signal_data)),
            'std': float(np.std(signal_data)),
            'min': float(np.min(signal_data)),
            'max': float(np.max(signal_data)),
            'median': float(np.median(signal_data)),
            'range': float(np.ptp(signal_data)),  # peak-to-peak (max - min)
        }
        
        return stats
    
    @staticmethod
    def detect_events(
        signal_data: np.ndarray,
        threshold: float = 2.0
    ) -> np.ndarray:
        """
        Detect significant signal transitions using simple threshold method.
        
        Identifies points where signal exceeds a threshold in absolute z-score,
        indicating potential event boundaries (e.g., base transitions in DNA).
        
        Event Detection in Nanopore Sequencing:
        ---------------------------------------
        - DNA bases have different current levels (A/T/G/C signature)
        - Transitions between bases cause rapid current changes
        - Detecting these transitions is first step in segmentation
        - More sophisticated methods use HMMs, neural networks
        
        This is a simplified approach for demonstration:
        - Computes z-score: (signal - mean) / std
        - Finds points where |z-score| > threshold
        - Returns indices of significant deviations
        
        Optimization:
        ------------
        - Vectorized comparison (no loops)
        - Single pass over data
        - Boolean indexing for efficient filtering
        
        Time Complexity: O(n) - linear scan
        Space Complexity: O(k) where k is number of detected events
        
        Args:
            signal_data (np.ndarray): Signal to analyze (preferably normalized).
                                     Shape: (n_samples,)
            threshold (float): Z-score threshold for event detection.
                             Higher = fewer, more significant events.
                             Default: 2.0 (events beyond ±2 std)
        
        Returns:
            np.ndarray: Array of indices where events were detected.
                       Shape: (n_events,)
        
        Edge Cases:
        ----------
        - Constant signal (std=0): Returns empty array
        - Empty array: Returns empty array
        - All values below threshold: Returns empty array
        
        Example:
        --------
        >>> # Normalize signal first
        >>> normalized = SignalProcessor.normalize_signal(signal)
        >>> 
        >>> # Detect significant transitions
        >>> events = SignalProcessor.detect_events(normalized, threshold=2.5)
        >>> print(f"Detected {len(events)} events")
        >>> 
        >>> # Visualize events
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(normalized)
        >>> plt.scatter(events, normalized[events], color='red', marker='x')
        >>> plt.axhline(y=2.5, color='r', linestyle='--', alpha=0.3)
        >>> plt.axhline(y=-2.5, color='r', linestyle='--', alpha=0.3)
        >>> plt.show()
        """
        # Handle edge case: empty array
        if len(signal_data) == 0:
            return np.array([], dtype=int)
        
        # Compute z-scores (standardize signal)
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        
        # Handle edge case: zero standard deviation (constant signal)
        if std == 0 or np.isnan(std):
            return np.array([], dtype=int)
        
        z_scores = (signal_data - mean) / std
        
        # Find indices where absolute z-score exceeds threshold
        # This identifies significant deviations from baseline
        event_mask = np.abs(z_scores) > threshold
        event_indices = np.where(event_mask)[0]
        
        return event_indices


# Example timing-decorated convenience function
@timing_decorator
def process_signal_pipeline(
    raw_signal: np.ndarray,
    normalize: bool = True,
    filter_noise: bool = True,
    cutoff_freq: float = 0.1
) -> Dict[str, Any]:
    """
    Complete signal processing pipeline with timing.
    
    Applies full preprocessing workflow to raw nanopore signal:
    1. Normalization (optional)
    2. Noise filtering (optional)
    3. Statistical analysis
    4. Event detection
    
    Args:
        raw_signal (np.ndarray): Raw signal data
        normalize (bool): Apply z-score normalization. Default: True
        filter_noise (bool): Apply low-pass filtering. Default: True
        cutoff_freq (float): Filter cutoff frequency. Default: 0.1
    
    Returns:
        Dict[str, Any]: Processed results containing:
            - processed_signal: Final processed signal
            - statistics: Statistical summary
            - events: Detected event indices
            - processing_time: Total processing time
    
    Example:
    --------
    >>> raw_signal = read_record.signal
    >>> result = process_signal_pipeline(raw_signal)
    process_signal_pipeline executed in 0.0234 seconds
    >>> print(f"Detected {len(result['events'])} events")
    >>> print(f"Mean: {result['statistics']['mean']:.2f}")
    """
    start_time = time.time()
    
    # Start with raw signal
    processed = raw_signal.copy()
    
    # Apply normalization
    if normalize:
        processed = SignalProcessor.normalize_signal(processed)
    
    # Apply filtering
    if filter_noise:
        processed = SignalProcessor.filter_signal(processed, cutoff_freq=cutoff_freq)
    
    # Compute statistics
    stats = SignalProcessor.compute_statistics(processed)
    
    # Detect events
    events = SignalProcessor.detect_events(processed, threshold=2.0)
    
    processing_time = time.time() - start_time
    
    return {
        'processed_signal': processed,
        'statistics': stats,
        'events': events,
        'processing_time': processing_time,
    }
