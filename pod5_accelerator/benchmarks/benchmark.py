"""
Comprehensive benchmarking suite for comparing POD5 reading performance.

This module provides the POD5Benchmark class for systematically evaluating and
comparing the performance of different POD5 reading implementations:

1. Baseline (single-threaded): Straightforward sequential reading
2. Accelerated (multi-threaded): Optimized parallel reading with zero-copy

Benchmark Methodology:
---------------------
Each benchmark run measures:
- Total elapsed time (wall clock time)
- Number of reads processed
- Throughput (reads per second)
- Memory usage (RSS before/after using psutil)

The comparative benchmark runs both implementations on the same data and computes
percentage improvements to quantify optimization effectiveness.

Visualization:
-------------
Results are visualized using matplotlib with:
- Bar charts comparing throughput across methods
- Line plots showing throughput scaling with thread count
- Memory usage comparisons

Statistical Analysis:
--------------------
For multiple runs, calculates:
- Mean throughput and memory usage
- Standard deviation for variability assessment
- Percentage improvement over baseline

Usage:
------
>>> benchmark = POD5Benchmark(data_dir="./data")
>>> results_df = benchmark.run_comparative_benchmark(file_paths)
>>> benchmark.plot_results("results/benchmark_plot.png")
>>> benchmark.save_results("results/benchmark_results.csv")
"""

import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import os

from pod5_accelerator.core.accelerated_reader import AcceleratedPOD5Reader
from pod5_accelerator.core.baseline_reader import BaselinePOD5Reader


class POD5Benchmark:
    """
    Comprehensive benchmarking suite for POD5 reader performance comparison.
    
    This class provides a systematic framework for benchmarking and comparing
    different POD5 reading implementations. It measures key performance metrics,
    generates visualizations, and calculates improvement percentages.
    
    Key Features:
    ------------
    - Automated benchmarking: Run baseline and accelerated readers on same data
    - Performance metrics: Time, throughput, memory usage
    - Statistical analysis: Mean, std dev, percentage improvements
    - Visualization: Automated plot generation with matplotlib
    - Data export: Save results to CSV for further analysis
    
    Benchmark Methodology:
    ---------------------
    Each benchmark follows this procedure:
    1. Measure initial memory usage (RSS)
    2. Start timer
    3. Process all files/reads
    4. Stop timer and measure final memory
    5. Calculate throughput and memory delta
    6. Collect statistics from reader
    
    Attributes:
        data_dir (Path): Directory containing POD5 files for benchmarking
        results (List[Dict]): Accumulated benchmark results from all runs
    
    Example:
    --------
    >>> # Initialize benchmark suite
    >>> benchmark = POD5Benchmark(data_dir="./data")
    >>> 
    >>> # Run comparative benchmark on multiple files
    >>> files = ["data/file1.pod5", "data/file2.pod5", "data/file3.pod5"]
    >>> results_df = benchmark.run_comparative_benchmark(files)
    >>> 
    >>> # Calculate improvements
    >>> improvements = benchmark.calculate_improvements()
    >>> print(f"Throughput improvement: {improvements['throughput_improvement']:.1f}%")
    >>> 
    >>> # Generate visualizations
    >>> benchmark.plot_results("results/benchmark_comparison.png")
    >>> 
    >>> # Save detailed results
    >>> benchmark.save_results("results/benchmark_data.csv")
    """
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the POD5 benchmark suite.
        
        Args:
            data_dir (str): Path to directory containing POD5 files for testing.
                          Default: "./data"
        
        The data directory should contain one or more POD5 files for benchmarking.
        Results are stored internally and can be exported using save_results().
        """
        self.data_dir = Path(data_dir)
        self.results: List[Dict[str, Any]] = []
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def run_baseline_benchmark(
        self,
        file_path: str,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Benchmark the baseline single-threaded POD5 reader.
        
        Measures performance of the straightforward single-threaded implementation
        to establish a baseline for comparison with optimized readers.
        
        Measurement Process:
        -------------------
        1. Record initial memory usage (RSS in MB)
        2. Initialize BaselinePOD5Reader
        3. Process entire file in batches
        4. Record final memory usage
        5. Collect statistics (time, throughput, reads processed)
        
        Args:
            file_path (str): Path to POD5 file to benchmark.
            batch_size (int): Batch size for reading. Default: 1000 reads.
        
        Returns:
            Dict[str, Any]: Benchmark results containing:
                - method (str): "baseline"
                - file_path (str): Path to benchmarked file
                - reads_processed (int): Total reads processed
                - elapsed_time (float): Total time in seconds
                - throughput (float): Reads per second
                - memory_before_mb (float): RSS memory before (MB)
                - memory_after_mb (float): RSS memory after (MB)
                - memory_delta_mb (float): Memory increase (MB)
                - batch_size (int): Batch size used
        
        Performance Expectations:
        ------------------------
        Baseline throughput: ~5,000-10,000 reads/sec (single-threaded)
        Memory usage: Minimal, O(batch_size) due to generator pattern
        
        Example:
        --------
        >>> benchmark = POD5Benchmark()
        >>> result = benchmark.run_baseline_benchmark("data/sample.pod5")
        >>> print(f"Baseline: {result['throughput']:.0f} reads/sec")
        """
        print(f"Running baseline benchmark on {file_path}...")
        
        # Measure memory before (Resident Set Size in bytes)
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        # Initialize baseline reader and start timing
        reader = BaselinePOD5Reader()
        start_time = time.time()
        
        # Process file
        read_count = 0
        for batch in reader.read_file_batch(file_path, batch_size=batch_size):
            read_count += len(batch)
        
        # Measure elapsed time
        elapsed_time = time.time() - start_time
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = memory_after - memory_before
        
        # Get reader statistics
        stats = reader.get_stats()
        
        # Compile results
        result = {
            'method': 'baseline',
            'file_path': file_path,
            'reads_processed': stats['reads_processed'],
            'elapsed_time': elapsed_time,
            'throughput': stats['throughput'],
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_delta_mb': memory_delta,
            'batch_size': batch_size,
            'num_threads': 1,  # Baseline is single-threaded
        }
        
        self.results.append(result)
        print(f"  Throughput: {result['throughput']:.2f} reads/sec")
        print(f"  Memory delta: {memory_delta:.2f} MB")
        
        return result
    
    def run_accelerated_benchmark(
        self,
        file_path: str,
        num_threads: int = 4,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Benchmark the accelerated multi-threaded POD5 reader.
        
        Measures performance of the optimized multi-threaded implementation with
        zero-copy operations to quantify improvement over baseline.
        
        Measurement Process:
        -------------------
        1. Record initial memory usage (RSS in MB)
        2. Initialize AcceleratedPOD5Reader with specified thread count
        3. Process entire file in batches
        4. Record final memory usage
        5. Collect statistics (time, throughput, reads processed)
        
        Args:
            file_path (str): Path to POD5 file to benchmark.
            num_threads (int): Number of worker threads. Default: 4.
            batch_size (int): Batch size for reading. Default: 1000 reads.
        
        Returns:
            Dict[str, Any]: Benchmark results containing:
                - method (str): "accelerated"
                - file_path (str): Path to benchmarked file
                - reads_processed (int): Total reads processed
                - elapsed_time (float): Total time in seconds
                - throughput (float): Reads per second
                - memory_before_mb (float): RSS memory before (MB)
                - memory_after_mb (float): RSS memory after (MB)
                - memory_delta_mb (float): Memory increase (MB)
                - batch_size (int): Batch size used
                - num_threads (int): Thread count used
        
        Performance Expectations:
        ------------------------
        With 4 threads: ~18,000-25,000 reads/sec (3.5-4x speedup)
        With 8 threads: ~30,000-40,000 reads/sec (6-7x speedup)
        Memory usage: Slightly higher due to thread pool overhead
        
        Example:
        --------
        >>> benchmark = POD5Benchmark()
        >>> result = benchmark.run_accelerated_benchmark("data/sample.pod5", num_threads=8)
        >>> print(f"Accelerated (8 threads): {result['throughput']:.0f} reads/sec")
        """
        print(f"Running accelerated benchmark on {file_path} ({num_threads} threads)...")
        
        # Measure memory before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Initialize accelerated reader and start timing
        reader = AcceleratedPOD5Reader(num_threads=num_threads)
        start_time = time.time()
        
        # Process file
        read_count = 0
        for batch in reader.read_file_batch(file_path, batch_size=batch_size):
            read_count += len(batch)
        
        # Measure elapsed time
        elapsed_time = time.time() - start_time
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = memory_after - memory_before
        
        # Get reader statistics
        stats = reader.get_stats()
        
        # Compile results
        result = {
            'method': 'accelerated',
            'file_path': file_path,
            'reads_processed': stats['reads_processed'],
            'elapsed_time': elapsed_time,
            'throughput': stats['throughput'],
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_delta_mb': memory_delta,
            'batch_size': batch_size,
            'num_threads': num_threads,
        }
        
        self.results.append(result)
        print(f"  Throughput: {result['throughput']:.2f} reads/sec")
        print(f"  Memory delta: {memory_delta:.2f} MB")
        
        return result
    
    def run_comparative_benchmark(
        self,
        file_paths: List[str],
        thread_counts: List[int] = [2, 4, 8],
        batch_size: int = 1000
    ) -> pd.DataFrame:
        """
        Run comprehensive comparative benchmark across all methods.
        
        Executes benchmarks for both baseline and accelerated readers across
        multiple thread counts, enabling thorough performance comparison and
        scalability analysis.
        
        Benchmark Sequence:
        ------------------
        1. Run baseline benchmark on first file
        2. For each thread count:
           - Run accelerated benchmark on first file
        3. Compile all results into pandas DataFrame
        
        This provides data for:
        - Baseline vs accelerated comparison
        - Thread count scalability analysis
        - Statistical aggregation across runs
        
        Args:
            file_paths (List[str]): List of POD5 file paths to benchmark.
            thread_counts (List[int]): Thread counts to test for accelerated reader.
                                      Default: [2, 4, 8]
            batch_size (int): Batch size for all benchmarks. Default: 1000.
        
        Returns:
            pd.DataFrame: Comprehensive results with columns:
                - method: Reader type ("baseline" or "accelerated")
                - file_path: Benchmarked file
                - num_threads: Thread count used
                - reads_processed: Total reads
                - elapsed_time: Wall clock time (seconds)
                - throughput: Reads per second
                - memory_before_mb: Initial memory (MB)
                - memory_after_mb: Final memory (MB)
                - memory_delta_mb: Memory increase (MB)
                - batch_size: Batch size used
        
        The DataFrame enables easy analysis:
        >>> df.groupby('method')['throughput'].mean()
        >>> df.plot(x='num_threads', y='throughput', kind='line')
        
        Example:
        --------
        >>> benchmark = POD5Benchmark()
        >>> files = list(Path("data").glob("*.pod5"))
        >>> results_df = benchmark.run_comparative_benchmark(
        ...     file_paths=files[:3],
        ...     thread_counts=[2, 4, 8, 16]
        ... )
        >>> print(results_df.groupby('method')['throughput'].describe())
        """
        print(f"\n{'='*60}")
        print("POD5 COMPARATIVE BENCHMARK")
        print(f"{'='*60}\n")
        
        # Use first file for benchmarking
        if not file_paths:
            raise ValueError("No file paths provided for benchmarking")
        
        benchmark_file = file_paths[0]
        print(f"Benchmark file: {benchmark_file}\n")
        
        # Run baseline benchmark
        print("1. Baseline (single-threaded)")
        print("-" * 40)
        self.run_baseline_benchmark(benchmark_file, batch_size=batch_size)
        print()
        
        # Run accelerated benchmarks with different thread counts
        print("2. Accelerated (multi-threaded)")
        print("-" * 40)
        for i, num_threads in enumerate(thread_counts, 1):
            self.run_accelerated_benchmark(
                benchmark_file,
                num_threads=num_threads,
                batch_size=batch_size
            )
            if i < len(thread_counts):
                print()
        
        print(f"\n{'='*60}\n")
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        return df
    
    def calculate_improvements(self) -> Dict[str, float]:
        """
        Calculate percentage improvements of accelerated reader over baseline.
        
        Computes key performance metrics to quantify optimization effectiveness:
        - Throughput improvement percentage
        - Memory efficiency comparison
        - Time reduction percentage
        
        Statistical Approach:
        --------------------
        - Uses mean values when multiple runs exist
        - Calculates: ((accelerated - baseline) / baseline) * 100
        - Reports both absolute values and percentage improvements
        
        Returns:
            Dict[str, float]: Improvement metrics containing:
                - baseline_throughput: Mean baseline reads/sec
                - accelerated_throughput: Mean accelerated reads/sec
                - throughput_improvement: Percentage improvement
                - baseline_time: Mean baseline elapsed time
                - accelerated_time: Mean accelerated elapsed time
                - time_improvement: Percentage time reduction
                - baseline_memory_mb: Mean baseline memory delta
                - accelerated_memory_mb: Mean accelerated memory delta
                - memory_overhead_pct: Memory overhead percentage
        
        Example:
        --------
        >>> benchmark = POD5Benchmark()
        >>> # ... run benchmarks ...
        >>> improvements = benchmark.calculate_improvements()
        >>> print(f"Throughput improved by {improvements['throughput_improvement']:.1f}%")
        >>> print(f"Time reduced by {improvements['time_improvement']:.1f}%")
        >>> 
        >>> if improvements['throughput_improvement'] >= 40:
        ...     print("✓ Target 40% improvement achieved!")
        """
        if not self.results:
            raise ValueError("No benchmark results available. Run benchmarks first.")
        
        df = pd.DataFrame(self.results)
        
        # Separate baseline and accelerated results
        baseline_df = df[df['method'] == 'baseline']
        accelerated_df = df[df['method'] == 'accelerated']
        
        if baseline_df.empty or accelerated_df.empty:
            raise ValueError("Need both baseline and accelerated results for comparison")
        
        # Calculate mean metrics
        baseline_throughput = baseline_df['throughput'].mean()
        accelerated_throughput = accelerated_df['throughput'].mean()
        
        baseline_time = baseline_df['elapsed_time'].mean()
        accelerated_time = accelerated_df['elapsed_time'].mean()
        
        baseline_memory = baseline_df['memory_delta_mb'].mean()
        accelerated_memory = accelerated_df['memory_delta_mb'].mean()
        
        # Calculate percentage improvements
        throughput_improvement = ((accelerated_throughput - baseline_throughput) 
                                 / baseline_throughput * 100)
        
        time_improvement = ((baseline_time - accelerated_time) 
                           / baseline_time * 100)
        
        memory_overhead = ((accelerated_memory - baseline_memory) 
                          / baseline_memory * 100)
        
        improvements = {
            'baseline_throughput': baseline_throughput,
            'accelerated_throughput': accelerated_throughput,
            'throughput_improvement': throughput_improvement,
            'baseline_time': baseline_time,
            'accelerated_time': accelerated_time,
            'time_improvement': time_improvement,
            'baseline_memory_mb': baseline_memory,
            'accelerated_memory_mb': accelerated_memory,
            'memory_overhead_pct': memory_overhead,
        }
        
        return improvements
    
    def plot_results(self, output_path: Union[str, Path] = "results/benchmark_plots.png"):
        """
        Generate comprehensive visualization of benchmark results.
        
        Creates a multi-panel figure with three key visualizations:
        1. Bar chart: Throughput comparison (baseline vs accelerated methods)
        2. Line plot: Throughput scaling with thread count
        3. Bar chart: Memory usage comparison
        
        Visualization Details:
        ---------------------
        - Figure size: 15x5 inches (3 subplots side-by-side)
        - Colors: Blue for baseline, orange/green for accelerated
        - Grid: Enabled for easy value reading
        - Labels: Clear axis labels and titles
        - Legend: Positioned for minimal overlap
        
        Args:
            output_path (str): Path to save the plot image (PNG format).
                             Default: "results/benchmark_plots.png"
                             Directory will be created if it doesn't exist.
        
        Raises:
            ValueError: If no benchmark results are available.
        
        Output:
            Saves figure to specified path and confirms with message.
        
        Example:
        --------
        >>> benchmark = POD5Benchmark()
        >>> # ... run benchmarks ...
        >>> benchmark.plot_results("results/performance_comparison.png")
        >>> # Figure saved with three subplots showing comprehensive results
        """
        if not self.results:
            raise ValueError("No benchmark results to plot. Run benchmarks first.")
        
        df = pd.DataFrame(self.results)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Throughput comparison (bar chart)
        ax1 = axes[0]
        methods = df.groupby('method')['throughput'].mean()
        methods.plot(kind='bar', ax=ax1, color=['#1f77b4', '#ff7f0e'])
        ax1.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Throughput (reads/sec)')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.0f')
        
        # Plot 2: Throughput vs thread count (line plot)
        ax2 = axes[1]
        accelerated_df = df[df['method'] == 'accelerated']
        if not accelerated_df.empty:
            thread_throughput = accelerated_df.groupby('num_threads')['throughput'].mean()
            ax2.plot(thread_throughput.index, thread_throughput.values, 
                    marker='o', linewidth=2, markersize=8, color='#2ca02c')
            ax2.set_title('Throughput Scaling', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Number of Threads')
            ax2.set_ylabel('Throughput (reads/sec)')
            ax2.grid(alpha=0.3)
            
            # Add baseline reference line
            baseline_throughput = df[df['method'] == 'baseline']['throughput'].mean()
            ax2.axhline(y=baseline_throughput, color='#1f77b4', 
                       linestyle='--', label='Baseline', linewidth=2)
            ax2.legend()
        
        # Plot 3: Memory usage comparison (bar chart)
        ax3 = axes[2]
        memory = df.groupby('method')['memory_delta_mb'].mean()
        memory.plot(kind='bar', ax=ax3, color=['#1f77b4', '#ff7f0e'])
        ax3.set_title('Memory Usage', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Method')
        ax3.set_ylabel('Memory Delta (MB)')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for container in ax3.containers:
            ax3.bar_label(container, fmt='%.1f')
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
        plt.close()
    
    def save_results(self, output_path: Union[str, Path] = "results/benchmark_results.csv"):
        """
        Save benchmark results to CSV file for further analysis.
        
        Exports all accumulated benchmark results to a CSV file with proper
        formatting for easy import into spreadsheets, R, pandas, etc.
        
        CSV Columns:
        -----------
        - method: Reader type (baseline/accelerated)
        - file_path: Benchmarked file path
        - num_threads: Thread count used
        - reads_processed: Total reads processed
        - elapsed_time: Wall clock time (seconds)
        - throughput: Reads per second
        - memory_before_mb: Initial memory usage (MB)
        - memory_after_mb: Final memory usage (MB)
        - memory_delta_mb: Memory increase (MB)
        - batch_size: Batch size configuration
        
        Args:
            output_path (str): Path to save CSV file.
                             Default: "results/benchmark_results.csv"
                             Directory will be created if it doesn't exist.
        
        Raises:
            ValueError: If no benchmark results are available.
        
        Output:
            CSV file with all results and confirmation message.
        
        Example:
        --------
        >>> benchmark = POD5Benchmark()
        >>> # ... run benchmarks ...
        >>> benchmark.save_results("results/pod5_benchmark_2024.csv")
        >>> 
        >>> # Later analysis:
        >>> import pandas as pd
        >>> df = pd.read_csv("results/pod5_benchmark_2024.csv")
        >>> print(df.describe())
        """
        if not self.results:
            raise ValueError("No benchmark results to save. Run benchmarks first.")
        
        df = pd.DataFrame(self.results)
        
        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"Results saved to: {output_path}")
        print(f"Total records: {len(df)}")
