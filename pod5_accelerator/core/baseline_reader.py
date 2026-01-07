"""
Baseline single-threaded POD5 file reader for performance comparison.

This module implements BaselinePOD5Reader, a straightforward single-threaded
implementation that serves as a performance baseline for measuring the improvements
achieved by the AcceleratedPOD5Reader.

The baseline reader intentionally avoids optimizations to provide a fair comparison
point for evaluating multi-threading and zero-copy techniques.
"""

import pod5  # type: ignore
import numpy as np
from pathlib import Path
from typing import Iterator, List, Dict, Any
import time

class BaselinePOD5Reader:
    """
    Single-threaded baseline POD5 file reader for performance comparison.
    
    This reader provides a straightforward, unoptimized implementation that serves
    as a baseline for measuring the performance improvements of AcceleratedPOD5Reader.
    
    Key Characteristics:
    -------------------
    - Single-threaded: Processes files sequentially without parallelism
    - Simple implementation: Minimal complexity for clear baseline measurement
    - Same interface: Compatible API for fair performance comparison
    - Metrics tracking: Provides same statistics as AcceleratedPOD5Reader
    
    This implementation represents a typical "naive" approach to reading POD5 files
    without applying optimization techniques. It processes one file at a time in a
    single thread, which is sufficient for many use cases but becomes a bottleneck
    when processing large sequencing runs with multiple files.
    
    Expected Performance:
    --------------------
    - Throughput: ~5,000-10,000 reads/sec (single-threaded, storage-dependent)
    - Scalability: Does not utilize multiple CPU cores
    - Memory: Efficient due to generator pattern
    
    Usage Example:
    --------------
    >>> baseline = BaselinePOD5Reader()
    >>> for batch in baseline.read_file_batch("data/run.pod5", batch_size=1000):
    ...     for read_data in batch:
    ...         signal = read_data['signal']
    ...
    >>> stats = baseline.get_stats()
    >>> print(f"Baseline throughput: {stats['throughput']:.2f} reads/sec")
    """
    
    def __init__(self):
        """
        Initialize the baseline POD5 reader.
        
        No thread pool or complex initialization needed for baseline implementation.
        """
        # Performance tracking
        self._reads_processed = 0
        self._total_time = 0.0
        self._start_time = None
    
    def read_file_batch(
        self, 
        file_path: str, 
        batch_size: int = 1000
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Read a single POD5 file in batches using single-threaded approach.
        
        This method implements the same interface as AcceleratedPOD5Reader but
        without multi-threading or other advanced optimizations. It provides a
        fair baseline for performance comparison.
        
        Args:
            file_path (str): Path to the POD5 file to read.
            batch_size (int): Number of reads to yield per batch.
                            Default: 1000 reads.
        
        Yields:
            List[Dict[str, Any]]: Batches of read data dictionaries containing:
                - read_id (str): Unique read identifier
                - signal (np.ndarray): Raw signal data
                - num_samples (int): Number of samples in signal
                - sample_rate (int): Sampling frequency in Hz
                - channel (int): Nanopore channel number
                - start_time (int): Acquisition start time

        """
        if self._start_time is None:
            self._start_time = time.time()
        
        batch = []
        with pod5.Reader(Path(file_path)) as reader:
            # Iterate through all reads in the file sequentially
            for read_record in reader.reads():
                read_data = {
                    'read_id': str(read_record.read_id),
                    'signal': read_record.signal,
                    'num_samples': read_record.num_samples,
                    'sample_rate': read_record.sample_rate,
                    'channel': read_record.channel,
                    'start_time': read_record.start_time,
                }
                
                batch.append(read_data)
                self._reads_processed += 1
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
    
    def read_multiple_files(
        self,
        file_paths: List[str],
        batch_size: int = 1000
    ) -> Iterator[Dict[str, Any]]:
        """
        Read multiple POD5 files sequentially (single-threaded baseline).
        
        Unlike AcceleratedPOD5Reader, this method processes files one at a time
        without parallelism. This represents the baseline performance without
        multi-threading optimization.
        
        Args:
            file_paths (List[str]): List of POD5 file paths to process sequentially.
            batch_size (int): Batch size for each individual file read operation.
                            Default: 1000 reads.
        
        Yields:
            Dict[str, Any]: Individual read data dictionaries (flattened from batches).

        """
        if self._start_time is None:
            self._start_time = time.time()
        for file_path in file_paths:
            try:
                for batch in self.read_file_batch(file_path, batch_size):
                    # Flatten batches and yield individual reads
                    for read_data in batch:
                        yield read_data
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                raise
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get performance statistics for all read operations.
        
        Provides the same performance metrics as AcceleratedPOD5Reader for
        fair comparison of baseline vs optimized implementations.
        
        Returns:
            Dict[str, float]: Performance statistics containing:
                - reads_processed (int): Total number of reads processed
                - total_time (float): Elapsed time in seconds
                - throughput (float): Reads per second (reads_processed / total_time)
        >>> reader = BaselinePOD5Reader()
        >>> # ... process files ...
        >>> stats = reader.get_stats()
        >>> print(f"Baseline throughput: {stats['throughput']:.2f} reads/sec")
        
        Compare with AcceleratedPOD5Reader:
        -----------------------------------
        >>> baseline = BaselinePOD5Reader()
        >>> accelerated = AcceleratedPOD5Reader(num_threads=8)
        >>> # ... process same files with both readers ...
        >>> baseline_stats = baseline.get_stats()
        >>> accelerated_stats = accelerated.get_stats()
        >>> improvement = (accelerated_stats['throughput'] / 
        ...                baseline_stats['throughput'] - 1) * 100
        >>> print(f"Throughput improvement: {improvement:.1f}%")
        """
        if self._start_time is not None:
            self._total_time = time.time() - self._start_time
        
        throughput = 0.0
        if self._total_time > 0:
            throughput = self._reads_processed / self._total_time
        
        return {
            'reads_processed': self._reads_processed,
            'total_time': self._total_time,
            'throughput': throughput,
        }
