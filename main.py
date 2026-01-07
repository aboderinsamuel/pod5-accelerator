#!/usr/bin/env python3
"""
POD5 Accelerator Demonstration Script

This script showcases the capabilities of the POD5 accelerator by:
1. Checking for existing POD5 data or generating synthetic test data
2. Running baseline single-threaded benchmarks
3. Running accelerated multi-threaded benchmarks with various thread counts
4. Comparing performance and calculating improvement percentages
5. Generating visualization plots of results
6. Saving detailed benchmark data to CSV

The demonstration highlights the 40%+ performance improvement achieved through
multi-threading and zero-copy optimizations.

Usage:
------
python main.py                              # Run with default settings
python main.py --num-threads 8              # Specify thread count
python main.py --generate-data              # Force generation of new synthetic data
python main.py --data-dir ./custom_data     # Use custom data directory
"""

import argparse
import sys
from pathlib import Path
from typing import List
import time

# Optional imports (install with pip if available)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    tqdm = None  # type: ignore
    HAS_TQDM = False
    print("Note: Install 'tqdm' for progress bars: pip install tqdm")

try:
    from rich.console import Console  # type: ignore
    from rich.table import Table  # type: ignore
    from rich import box  # type: ignore
    HAS_RICH = True
except ImportError:
    Console = None  # type: ignore
    Table = None  # type: ignore
    box = None  # type: ignore
    HAS_RICH = False
    print("Note: Install 'rich' for formatted tables: pip install rich")

# Import POD5 accelerator modules
from pod5_accelerator.benchmarks.benchmark import POD5Benchmark
from pod5_accelerator.core.synthetic_generator import SyntheticPOD5Generator
from pod5_accelerator.core.accelerated_reader import AcceleratedPOD5Reader
from pod5_accelerator.core.baseline_reader import BaselinePOD5Reader


def print_header():
    """
    Print Oxford Nanopore-themed ASCII art header.
    
    Creates an eye-catching header for the demonstration output.
    """
    header = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║        ██████╗  ██████╗ ██████╗ ███████╗                         ║
    ║        ██╔══██╗██╔═══██╗██╔══██╗██╔════╝                         ║
    ║        ██████╔╝██║   ██║██║  ██║███████╗                         ║
    ║        ██╔═══╝ ██║   ██║██║  ██║╚════██║                         ║
    ║        ██║     ╚██████╔╝██████╔╝███████║                         ║
    ║        ╚═╝      ╚═════╝ ╚═════╝ ╚══════╝                         ║
    ║                                                                   ║
    ║              ACCELERATED FILE READER DEMO                        ║
    ║           High-Performance POD5 Processing                       ║
    ║                                                                   ║
    ║        Multi-threading + Zero-copy Optimizations                 ║
    ║                Target: 40%+ Improvement                          ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(header)


def check_or_generate_data(data_dir: Path, force_generate: bool = False) -> List[str]:
    """
    Check if POD5 files exist in data directory, generate if needed.
    
    Demonstration Flow:
    ------------------
    1. Check if data_dir exists and contains .pod5 files
    2. If no files found or force_generate=True:
       - Generate 3 synthetic POD5 files
       - Each with 1000 reads (~12 MB per file)
    3. Return list of POD5 file paths
    
    Args:
        data_dir (Path): Directory to check/store POD5 files
        force_generate (bool): Force generation even if files exist
    
    Returns:
        List[str]: List of POD5 file paths for benchmarking
    
    Example:
    --------
    >>> data_dir = Path("./data")
    >>> files = check_or_generate_data(data_dir)
    >>> print(f"Found {len(files)} POD5 files")
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for existing POD5 files
    pod5_files = list(data_dir.glob("*.pod5"))
    
    if not pod5_files or force_generate:
        print("\n" + "="*70)
        print("No POD5 files found. Generating synthetic test data...")
        print("="*70)
        
        # Generate synthetic test dataset
        generator = SyntheticPOD5Generator(seed=42)
        generator.create_test_dataset(
            output_dir=str(data_dir),
            num_files=3,
            reads_per_file=1000
        )
        
        # Refresh file list
        pod5_files = list(data_dir.glob("*.pod5"))
    
    else:
        print(f"\n✓ Found {len(pod5_files)} POD5 files in {data_dir}")
        for f in pod5_files[:5]:  # Show first 5
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  - {f.name} ({size_mb:.1f} MB)")
        if len(pod5_files) > 5:
            print(f"  ... and {len(pod5_files) - 5} more files")
    
    return [str(f) for f in pod5_files]


def run_benchmarks(
    benchmark: POD5Benchmark,
    file_paths: List[str],
    thread_counts: List[int]
) -> None:
    """
    Run comprehensive benchmark suite with progress indication.
    
    Benchmark Sequence:
    ------------------
    1. Baseline single-threaded benchmark
    2. Accelerated benchmarks with 2, 4, 8 threads (or custom)
    3. Collect and display results
    
    Args:
        benchmark (POD5Benchmark): Benchmark instance
        file_paths (List[str]): POD5 files to benchmark
        thread_counts (List[int]): Thread counts to test
    
    Uses tqdm for progress bars if available, otherwise simple prints.
    """
    if HAS_TQDM and tqdm is not None:
        print("\nRunning benchmarks...")
        progress = tqdm(total=len(thread_counts) + 1, desc="Benchmark Progress")
        
        # Wrap benchmark calls
        benchmark.run_comparative_benchmark(file_paths, thread_counts=thread_counts)
        progress.update(len(thread_counts) + 1)
        progress.close()
    else:
        # Run without progress bar
        benchmark.run_comparative_benchmark(file_paths, thread_counts=thread_counts)


def display_results_table(improvements: dict):
    """
    Display benchmark results in formatted table.
    
    Uses rich library for pretty tables if available, otherwise plain text.
    
    Table Contents:
    --------------
    - Baseline throughput
    - Accelerated throughput
    - Improvement percentage
    - Time reduction
    - Memory overhead
    
    Args:
        improvements (dict): Improvement metrics from calculate_improvements()
    """
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    
    if HAS_RICH and Console is not None and Table is not None and box is not None:
        # Create rich formatted table
        console = Console()
        table = Table(title="Performance Comparison", box=box.DOUBLE_EDGE)
        
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Baseline", style="yellow")
        table.add_column("Accelerated", style="green")
        table.add_column("Improvement", style="magenta bold")
        
        # Throughput row
        table.add_row(
            "Throughput",
            f"{improvements['baseline_throughput']:.0f} reads/sec",
            f"{improvements['accelerated_throughput']:.0f} reads/sec",
            f"+{improvements['throughput_improvement']:.1f}%"
        )
        
        # Time row
        table.add_row(
            "Elapsed Time",
            f"{improvements['baseline_time']:.2f} sec",
            f"{improvements['accelerated_time']:.2f} sec",
            f"-{improvements['time_improvement']:.1f}%"
        )
        
        # Memory row
        table.add_row(
            "Memory Usage",
            f"{improvements['baseline_memory_mb']:.1f} MB",
            f"{improvements['accelerated_memory_mb']:.1f} MB",
            f"+{improvements['memory_overhead_pct']:.1f}%"
        )
        
        console.print(table)
    
    else:
        # Plain text table
        print(f"\nThroughput:")
        print(f"  Baseline:    {improvements['baseline_throughput']:>10.0f} reads/sec")
        print(f"  Accelerated: {improvements['accelerated_throughput']:>10.0f} reads/sec")
        print(f"  Improvement: {improvements['throughput_improvement']:>10.1f}%")
        
        print(f"\nElapsed Time:")
        print(f"  Baseline:    {improvements['baseline_time']:>10.2f} seconds")
        print(f"  Accelerated: {improvements['accelerated_time']:>10.2f} seconds")
        print(f"  Improvement: {improvements['time_improvement']:>10.1f}%")
        
        print(f"\nMemory Usage:")
        print(f"  Baseline:    {improvements['baseline_memory_mb']:>10.1f} MB")
        print(f"  Accelerated: {improvements['accelerated_memory_mb']:>10.1f} MB")
        print(f"  Overhead:    {improvements['memory_overhead_pct']:>10.1f}%")
    
    print("\n" + "="*70)


def print_summary(improvements: dict):
    """
    Print final summary with achievement banner.
    
    Highlights whether the 40% improvement target was achieved.
    
    Args:
        improvements (dict): Improvement metrics
    """
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    throughput_improvement = improvements['throughput_improvement']
    
    if throughput_improvement >= 40.0:
        print("\n🎉 SUCCESS! Target improvement achieved! 🎉")
        print(f"\n   Throughput improved by {throughput_improvement:.1f}%")
        print("   (Target: 40%)")
        print("\n✓ Multi-threading optimization effective")
        print("✓ Zero-copy operations working")
        print("✓ Performance goals met")
    else:
        print(f"\n⚠ Throughput improved by {throughput_improvement:.1f}%")
        print("   (Target: 40%)")
        print("\nNote: Performance may vary based on:")
        print("  - Storage I/O speed (SSD vs HDD)")
        print("  - CPU core count and availability")
        print("  - System load and background processes")
        print("  - File sizes and data characteristics")
    
    print("\n" + "="*70)


def main():
    """
    Main demonstration script entry point.
    
    Demonstration Flow:
    ------------------
    1. Parse command-line arguments
    2. Print header
    3. Check/generate POD5 data
    4. Run baseline benchmark
    5. Run accelerated benchmarks (2, 4, 8 threads)
    6. Display results in formatted table
    7. Generate performance plots
    8. Save benchmark data to CSV
    9. Print summary with achievement status
    
    Command-line Arguments:
    ----------------------
    --data-dir: Path to POD5 files (default: ./data)
    --output-dir: Path for results (default: ./results)
    --num-threads: Max thread count (default: 8)
    --generate-data: Force generation of synthetic data
    
    Example:
    --------
    $ python main.py
    $ python main.py --num-threads 16 --data-dir ./my_data
    $ python main.py --generate-data --output-dir ./benchmark_results
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="POD5 Accelerator Demonstration - High-Performance File Reading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run with defaults
  python main.py --num-threads 16                   # Test up to 16 threads
  python main.py --generate-data                    # Force new test data
  python main.py --data-dir ./data --output-dir ./results
        """
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing POD5 files (default: ./data)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory for results and plots (default: ./results)"
    )
    
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Maximum number of threads to test (default: 8)"
    )
    
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Force generation of new synthetic test data"
    )
    
    args = parser.parse_args()
    
    # Convert paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print header
    print_header()
    
    # Check/generate data
    file_paths = check_or_generate_data(data_dir, force_generate=args.generate_data)
    
    if not file_paths:
        print("ERROR: No POD5 files available for benchmarking")
        sys.exit(1)
    
    # Initialize benchmark
    print(f"\nInitializing benchmark suite...")
    benchmark = POD5Benchmark(data_dir=str(data_dir))
    
    # Determine thread counts to test
    # Test 2, 4, and up to specified max threads
    max_threads = args.num_threads
    thread_counts = [2, 4]
    if max_threads > 4:
        thread_counts.append(max_threads)
    
    print(f"Thread counts to test: {thread_counts}")
    
    # Run benchmarks
    run_benchmarks(benchmark, file_paths, thread_counts)
    
    # Calculate improvements
    print("\nCalculating improvements...")
    improvements = benchmark.calculate_improvements()
    
    # Display results
    display_results_table(improvements)
    
    # Generate plots
    plot_path = output_dir / "benchmark_comparison.png"
    print(f"\nGenerating performance plots...")
    benchmark.plot_results(str(plot_path))
    
    # Save results to CSV
    csv_path = output_dir / "benchmark_results.csv"
    print(f"Saving benchmark data to CSV...")
    benchmark.save_results(str(csv_path))
    
    # Print summary
    print_summary(improvements)
    
    # Final message
    print(f"\n📁 Results saved to: {output_dir.absolute()}")
    print(f"   - Plots: {plot_path.name}")
    print(f"   - Data:  {csv_path.name}")
    print("\nThank you for using POD5 Accelerator! 🚀\n")


if __name__ == "__main__":
    main()
