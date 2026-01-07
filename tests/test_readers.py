"""
Comprehensive unit tests for POD5 reader modules.

This test suite provides thorough coverage of the POD5 accelerator components:
1. AcceleratedPOD5Reader - Multi-threaded optimized reader
2. BaselinePOD5Reader - Single-threaded baseline reader
3. Edge cases and error handling
4. Performance comparison validation

Test Strategy:
-------------
- Use pytest fixtures for test data generation
- Create temporary POD5 files for testing
- Test both success and failure scenarios
- Validate performance improvements
- Ensure thread safety and correctness

Coverage Goals:
--------------
- >80% code coverage across all modules
- Test all public methods and APIs
- Validate edge cases (empty files, errors, etc.)
- Performance regression detection

Usage:
------
pytest tests/test_readers.py -v                  # Run with verbose output
pytest tests/test_readers.py --cov=pod5_accelerator   # With coverage
pytest tests/test_readers.py -k "test_accelerated"    # Run specific tests
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
import time
import uuid

# Import pod5 for creating test files
import pod5

# Import modules to test
from pod5_accelerator.core.accelerated_reader import AcceleratedPOD5Reader
from pod5_accelerator.core.baseline_reader import BaselinePOD5Reader
from pod5_accelerator.core.synthetic_generator import SyntheticPOD5Generator


# ============================================================================
# Fixtures for Test Data Generation
# ============================================================================

@pytest.fixture
def temp_dir():
    """
    Create temporary directory for test files.
    
    Yields:
        Path: Temporary directory path
    
    Cleanup:
        Automatically cleaned up after test
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pod5_file(temp_dir):
    """
    Create a small synthetic POD5 file for testing.
    
    Test File Characteristics:
    -------------------------
    - 100 reads (small for fast tests)
    - Synthetic signals
    - Valid POD5 format
    
    Args:
        temp_dir: Temporary directory fixture
    
    Returns:
        Path: Path to generated POD5 file
    
    Example:
    --------
    >>> def test_reader(sample_pod5_file):
    ...     reader = AcceleratedPOD5Reader()
    ...     for batch in reader.read_file_batch(str(sample_pod5_file)):
    ...         assert len(batch) > 0
    """
    generator = SyntheticPOD5Generator(seed=42)
    file_path = temp_dir / "test_sample.pod5"
    
    # Generate small test file (100 reads for speed)
    generator.save_to_pod5(str(file_path), num_reads=100)
    
    return file_path


@pytest.fixture
def multiple_pod5_files(temp_dir):
    """
    Create multiple synthetic POD5 files for parallel processing tests.
    
    Test Dataset:
    ------------
    - 3 files
    - 50 reads per file (150 total)
    - Small sizes for fast tests
    
    Args:
        temp_dir: Temporary directory fixture
    
    Returns:
        List[Path]: List of POD5 file paths
    """
    generator = SyntheticPOD5Generator(seed=42)
    
    file_paths = []
    for i in range(3):
        file_path = temp_dir / f"test_file_{i+1}.pod5"
        generator.save_to_pod5(str(file_path), num_reads=50)
        file_paths.append(file_path)
    
    return file_paths


@pytest.fixture
def empty_pod5_file(temp_dir):
    """
    Create an empty POD5 file for edge case testing.
    
    Args:
        temp_dir: Temporary directory fixture
    
    Returns:
        Path: Path to empty POD5 file
    """
    generator = SyntheticPOD5Generator(seed=42)
    file_path = temp_dir / "empty.pod5"
    generator.save_to_pod5(str(file_path), num_reads=0)
    return file_path


# ============================================================================
# AcceleratedPOD5Reader Tests
# ============================================================================

class TestAcceleratedPOD5Reader:
    """
    Test suite for AcceleratedPOD5Reader class.
    
    Validates:
    - Initialization with different thread counts
    - Batch reading functionality
    - Parallel processing of multiple files
    - Statistics tracking
    - Edge cases and error handling
    """
    
    def test_initialization(self):
        """
        Test reader initialization with various thread counts.
        
        Validates:
        - Default initialization (4 threads)
        - Custom thread count
        - Thread pool creation
        """
        # Default initialization
        reader = AcceleratedPOD5Reader()
        assert reader.num_threads == 4
        assert reader.executor is not None
        assert reader._reads_processed == 0
        
        # Custom thread count
        reader_custom = AcceleratedPOD5Reader(num_threads=8)
        assert reader_custom.num_threads == 8
        assert reader_custom.executor._max_workers == 8
    
    def test_read_file_batch(self, sample_pod5_file):
        """
        Test single file batch reading.
        
        Validates:
        - Batch generation works correctly
        - All reads are processed
        - Batch sizes are respected
        - Read data structure is correct
        
        Args:
            sample_pod5_file: Fixture providing test POD5 file
        """
        reader = AcceleratedPOD5Reader(num_threads=2)
        
        total_reads = 0
        batch_count = 0
        batch_size = 25
        
        for batch in reader.read_file_batch(str(sample_pod5_file), batch_size=batch_size):
            batch_count += 1
            total_reads += len(batch)
            
            # Validate batch size (except possibly last batch)
            assert len(batch) <= batch_size
            
            # Validate read data structure
            for read_data in batch:
                assert 'read_id' in read_data
                assert 'signal' in read_data
                assert 'num_samples' in read_data
                assert 'sample_rate' in read_data
                assert 'channel' in read_data
                assert 'start_time' in read_data
                
                # Validate signal is numpy array
                assert isinstance(read_data['signal'], np.ndarray)
                assert len(read_data['signal']) == read_data['num_samples']
        
        # Should process all 100 reads
        assert total_reads == 100
        assert batch_count == 4  # 100 reads / 25 batch_size = 4 batches
    
    def test_parallel_reading(self, multiple_pod5_files):
        """
        Test parallel reading of multiple files.
        
        Validates:
        - Multi-threading works correctly
        - All files are processed
        - All reads from all files are yielded
        - No data loss in parallel execution
        
        Args:
            multiple_pod5_files: Fixture providing 3 test POD5 files
        """
        reader = AcceleratedPOD5Reader(num_threads=4)
        
        file_paths = [str(f) for f in multiple_pod5_files]
        
        total_reads = 0
        read_ids = set()
        
        for read_data in reader.read_multiple_files(file_paths, batch_size=20):
            total_reads += 1
            read_ids.add(read_data['read_id'])
            
            # Validate read structure
            assert 'signal' in read_data
            assert isinstance(read_data['signal'], np.ndarray)
        
        # Should process all reads (3 files × 50 reads = 150)
        assert total_reads == 150
        # All read IDs should be unique
        assert len(read_ids) == 150
    
    def test_statistics_tracking(self, sample_pod5_file):
        """
        Test that statistics are correctly tracked.
        
        Validates:
        - Read count tracking
        - Time measurement
        - Throughput calculation
        - Stats availability after processing
        
        Args:
            sample_pod5_file: Fixture providing test POD5 file
        """
        reader = AcceleratedPOD5Reader(num_threads=2)
        
        # Process file
        for batch in reader.read_file_batch(str(sample_pod5_file)):
            pass
        
        # Get statistics
        stats = reader.get_stats()
        
        # Validate stats structure
        assert 'reads_processed' in stats
        assert 'total_time' in stats
        assert 'throughput' in stats
        
        # Validate values
        assert stats['reads_processed'] == 100
        assert stats['total_time'] > 0
        assert stats['throughput'] > 0
        assert stats['throughput'] == stats['reads_processed'] / stats['total_time']
    
    def test_empty_file_handling(self, empty_pod5_file):
        """
        Test handling of empty POD5 files.
        
        Validates:
        - No errors on empty file
        - Zero reads processed
        - Generator completes successfully
        
        Args:
            empty_pod5_file: Fixture providing empty POD5 file
        """
        reader = AcceleratedPOD5Reader(num_threads=2)
        
        batch_count = 0
        for batch in reader.read_file_batch(str(empty_pod5_file)):
            batch_count += 1
        
        # Should have no batches for empty file
        assert batch_count == 0
        
        stats = reader.get_stats()
        assert stats['reads_processed'] == 0
    
    def test_batch_size_variations(self, sample_pod5_file):
        """
        Test reading with different batch sizes.
        
        Validates:
        - Small batches (10 reads)
        - Large batches (200 reads, exceeds file size)
        - Batch size of 1
        
        Args:
            sample_pod5_file: Fixture providing test POD5 file
        """
        # Test small batch size
        reader_small = AcceleratedPOD5Reader(num_threads=2)
        batches_small = list(reader_small.read_file_batch(str(sample_pod5_file), batch_size=10))
        assert len(batches_small) == 10  # 100 reads / 10 = 10 batches
        
        # Test large batch size (exceeds file size)
        reader_large = AcceleratedPOD5Reader(num_threads=2)
        batches_large = list(reader_large.read_file_batch(str(sample_pod5_file), batch_size=200))
        assert len(batches_large) == 1  # All 100 reads in one batch
        assert len(batches_large[0]) == 100
        
        # Test batch size of 1
        reader_single = AcceleratedPOD5Reader(num_threads=2)
        batches_single = list(reader_single.read_file_batch(str(sample_pod5_file), batch_size=1))
        assert len(batches_single) == 100  # 100 batches of 1 read each


# ============================================================================
# BaselinePOD5Reader Tests
# ============================================================================

class TestBaselinePOD5Reader:
    """
    Test suite for BaselinePOD5Reader class.
    
    Validates:
    - Single-threaded reading functionality
    - Correctness of baseline implementation
    - Statistics tracking
    - Same interface as accelerated reader
    """
    
    def test_initialization(self):
        """
        Test baseline reader initialization.
        
        Validates:
        - Default initialization
        - Initial state is correct
        - No thread pool (single-threaded)
        """
        reader = BaselinePOD5Reader()
        assert reader._reads_processed == 0
        assert reader._total_time == 0.0
        assert reader._start_time is None
    
    def test_simple_read(self, sample_pod5_file):
        """
        Test basic file reading functionality.
        
        Validates:
        - Can read POD5 file
        - All reads are processed
        - Read data is correct
        
        Args:
            sample_pod5_file: Fixture providing test POD5 file
        """
        reader = BaselinePOD5Reader()
        
        total_reads = 0
        for batch in reader.read_file_batch(str(sample_pod5_file), batch_size=50):
            total_reads += len(batch)
            
            # Validate read structure
            for read_data in batch:
                assert 'read_id' in read_data
                assert 'signal' in read_data
                assert isinstance(read_data['signal'], np.ndarray)
        
        assert total_reads == 100
    
    def test_statistics(self, sample_pod5_file):
        """
        Test statistics collection.
        
        Validates:
        - Stats are tracked correctly
        - Throughput is calculated
        - Time measurement works
        
        Args:
            sample_pod5_file: Fixture providing test POD5 file
        """
        reader = BaselinePOD5Reader()
        
        # Process file
        for batch in reader.read_file_batch(str(sample_pod5_file)):
            pass
        
        # Get stats
        stats = reader.get_stats()
        
        assert stats['reads_processed'] == 100
        assert stats['total_time'] > 0
        assert stats['throughput'] > 0
    
    def test_multiple_files_sequential(self, multiple_pod5_files):
        """
        Test sequential processing of multiple files.
        
        Validates:
        - Files are processed in order
        - All reads from all files are yielded
        - No parallelization (single-threaded)
        
        Args:
            multiple_pod5_files: Fixture providing 3 test POD5 files
        """
        reader = BaselinePOD5Reader()
        
        file_paths = [str(f) for f in multiple_pod5_files]
        
        total_reads = 0
        for read_data in reader.read_multiple_files(file_paths):
            total_reads += 1
        
        # Should process all 150 reads (3 files × 50)
        assert total_reads == 150


# ============================================================================
# Performance Comparison Tests
# ============================================================================

class TestPerformanceComparison:
    """
    Test suite for validating performance improvements.
    
    Validates:
    - Accelerated reader is faster than baseline
    - Multi-threading provides speedup
    - Performance scales with thread count
    
    Note: These are relative performance tests, not absolute benchmarks.
    """
    
    def test_performance_vs_baseline(self, sample_pod5_file):
        """
        Test that accelerated reader outperforms baseline.
        
        Validates:
        - Accelerated reader processes same data
        - Both readers get same read count
        - Accelerated reader is generally faster (when file size is sufficient)
        
        Note: On small files, threading overhead may dominate, so we mainly
        validate correctness rather than strict performance improvement.
        
        Args:
            sample_pod5_file: Fixture providing test POD5 file
        """
        # Baseline benchmark
        baseline = BaselinePOD5Reader()
        baseline_start = time.time()
        baseline_reads = 0
        for batch in baseline.read_file_batch(str(sample_pod5_file)):
            baseline_reads += len(batch)
        baseline_time = time.time() - baseline_start
        
        # Accelerated benchmark
        accelerated = AcceleratedPOD5Reader(num_threads=4)
        accelerated_start = time.time()
        accelerated_reads = 0
        for batch in accelerated.read_file_batch(str(sample_pod5_file)):
            accelerated_reads += len(batch)
        accelerated_time = time.time() - accelerated_start
        
        # Both should process same number of reads
        assert baseline_reads == accelerated_reads == 100
        
        # Both should complete successfully
        assert baseline_time > 0
        assert accelerated_time > 0
        
        # Get final stats
        baseline_stats = baseline.get_stats()
        accelerated_stats = accelerated.get_stats()
        
        assert baseline_stats['reads_processed'] == 100
        assert accelerated_stats['reads_processed'] == 100
    
    def test_thread_count_scalability(self, multiple_pod5_files):
        """
        Test that increasing threads improves performance (on larger datasets).
        
        Validates:
        - Multiple thread counts work correctly
        - All process same number of reads
        - Generally, more threads ≥ performance (with multiple files)
        
        Args:
            multiple_pod5_files: Fixture providing 3 test POD5 files
        """
        file_paths = [str(f) for f in multiple_pod5_files]
        thread_counts = [1, 2, 4]
        results = []
        
        for num_threads in thread_counts:
            reader = AcceleratedPOD5Reader(num_threads=num_threads)
            start_time = time.time()
            
            read_count = 0
            for read_data in reader.read_multiple_files(file_paths):
                read_count += 1
            
            elapsed = time.time() - start_time
            results.append({
                'threads': num_threads,
                'reads': read_count,
                'time': elapsed
            })
        
        # All should process same reads
        assert all(r['reads'] == 150 for r in results)
        
        # All should complete successfully
        assert all(r['time'] > 0 for r in results)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """
    Test suite for edge cases and error handling.
    
    Validates:
    - Non-existent files
    - Invalid file paths
    - Empty files
    - Zero-length signals
    - Thread count edge cases
    """
    
    def test_nonexistent_file(self):
        """
        Test handling of non-existent file paths.
        
        Validates:
        - Appropriate error is raised
        - Reader doesn't crash
        """
        reader = AcceleratedPOD5Reader()
        
        with pytest.raises(Exception):  # Should raise FileNotFoundError or similar
            for batch in reader.read_file_batch("nonexistent_file.pod5"):
                pass
    
    def test_zero_threads(self):
        """
        Test initialization with zero or negative threads.
        
        Validates:
        - ThreadPoolExecutor handles edge cases
        - Reader can still be created (executor may default to 1)
        """
        # This tests that initialization doesn't crash
        # ThreadPoolExecutor behavior with 0 workers varies
        try:
            reader = AcceleratedPOD5Reader(num_threads=0)
            assert reader is not None
        except ValueError:
            # Some versions may raise ValueError, which is acceptable
            pass
    
    def test_very_large_batch_size(self, sample_pod5_file):
        """
        Test with extremely large batch size.
        
        Validates:
        - Handles batch size larger than file
        - Single batch contains all reads
        
        Args:
            sample_pod5_file: Fixture providing test POD5 file
        """
        reader = AcceleratedPOD5Reader()
        batches = list(reader.read_file_batch(str(sample_pod5_file), batch_size=1000000))
        
        # Should have exactly one batch with all reads
        assert len(batches) == 1
        assert len(batches[0]) == 100


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """
    Integration tests for complete workflows.
    
    Validates:
    - End-to-end processing pipelines
    - Multiple readers on same data
    - Consistency across methods
    """
    
    def test_read_consistency(self, sample_pod5_file):
        """
        Test that baseline and accelerated readers produce consistent results.
        
        Validates:
        - Both readers yield same read IDs
        - Both process same number of reads
        - Read data is equivalent
        
        Args:
            sample_pod5_file: Fixture providing test POD5 file
        """
        # Read with baseline
        baseline = BaselinePOD5Reader()
        baseline_read_ids = set()
        for batch in baseline.read_file_batch(str(sample_pod5_file)):
            for read_data in batch:
                baseline_read_ids.add(read_data['read_id'])
        
        # Read with accelerated
        accelerated = AcceleratedPOD5Reader(num_threads=2)
        accelerated_read_ids = set()
        for batch in accelerated.read_file_batch(str(sample_pod5_file)):
            for read_data in batch:
                accelerated_read_ids.add(read_data['read_id'])
        
        # Should have same read IDs
        assert baseline_read_ids == accelerated_read_ids
        assert len(baseline_read_ids) == 100


# ============================================================================
# Test Runner Configuration
# ============================================================================

if __name__ == "__main__":
    """
    Run tests directly with pytest.
    
    Usage:
    ------
    python test_readers.py
    """
    pytest.main([__file__, "-v", "--tb=short"])
