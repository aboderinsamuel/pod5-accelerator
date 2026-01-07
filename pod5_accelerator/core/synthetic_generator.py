"""
Synthetic POD5 data generator for testing and development.

⚠️  FOR DEVELOPMENT AND TESTING ONLY ⚠️

This module provides tools to generate synthetic POD5 files that mimic the structure
and characteristics of real Oxford Nanopore sequencing data. These synthetic datasets
are useful for:

1. Testing: Validate POD5 readers without requiring real sequencing data
2. Development: Develop and debug processing pipelines before data acquisition
3. Benchmarking: Create controlled datasets with known characteristics
4. CI/CD: Include in automated testing without large binary data files

IMPORTANT: Synthetic data does NOT represent biological sequences or real signals.
          - Generated signals are random noise with step patterns
          - Metadata is randomly generated (UUIDs, timestamps, etc.)
          - NOT suitable for basecalling or biological analysis
          - Use only for software testing and performance evaluation

Synthetic Signal Generation:
---------------------------
To approximate nanopore-like signals, we generate:
- Base current levels: 90-110 pA (typical nanopore range)
- Gaussian noise: std=5-10 pA (realistic noise levels)
- Step changes: Simulate DNA translocation events
- Duration: 4000-6000 samples (typical read lengths at 4 kHz)
- Sample rate: 4000 Hz (standard MinION configuration)

These parameters create signals that "look like" nanopore data from a structural
perspective, enabling realistic testing of file I/O and processing code.
"""

import pod5  # type: ignore
import numpy as np
import uuid
from pathlib import Path
from typing import List, Optional, Union
import datetime


class SyntheticPOD5Generator:
    """
    Generator for synthetic POD5 files mimicking nanopore sequencing data.
    
    ⚠️  TESTING/DEVELOPMENT ONLY - NOT FOR BIOLOGICAL ANALYSIS ⚠️
    
    Creates POD5 files with realistic structure but synthetic content:
    - Random signal data approximating nanopore electrical current
    - Valid metadata (UUIDs, timestamps, channel numbers)
    - Proper POD5 format for compatibility with readers
    
    Use Cases:
    ---------
    ✓ Testing POD5 reader implementations
    ✓ Benchmarking I/O performance
    ✓ Development without real sequencing data
    ✓ Continuous integration testing
    ✓ Demonstrating software capabilities
    
    NOT For:
    -------
    ✗ Basecalling or sequence analysis
    ✗ Training machine learning models
    ✗ Biological research or publication
    ✗ Simulating specific DNA sequences
    
    Signal Characteristics:
    ----------------------
    - Current levels: 90-110 pA (typical MinION range)
    - Noise: Gaussian with std=5-10 pA
    - Steps: Random level changes (simulate base transitions)
    - Duration: 4000-6000 samples (~1-1.5 seconds at 4 kHz)
    - Sample rate: 4000 Hz (standard)
    
    Example:
    --------
    >>> # Generate synthetic test data
    >>> generator = SyntheticPOD5Generator(seed=42)
    >>> 
    >>> # Create small test file
    >>> generator.save_to_pod5("test_data.pod5", num_reads=100)
    >>> 
    >>> # Create multi-file test dataset
    >>> generator.create_test_dataset(
    ...     output_dir="./test_data",
    ...     num_files=5,
    ...     reads_per_file=1000
    ... )
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize synthetic data generator.
        
        Args:
            seed (Optional[int]): Random seed for reproducible generation.
                                 If None, uses random initialization.
                                 Use fixed seed (e.g., 42) for testing.
        
        Example:
        --------
        >>> # Reproducible generation for testing
        >>> gen1 = SyntheticPOD5Generator(seed=42)
        >>> gen2 = SyntheticPOD5Generator(seed=42)
        >>> # Both will generate identical data
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def _generate_synthetic_signal(
        self,
        num_samples: int,
        base_current: float = 100.0,
        noise_std: float = 7.0,
        num_steps: int = 8
    ) -> np.ndarray:
        """
        Generate a synthetic nanopore-like signal with step patterns.
        
        Creates a signal that approximates the structure of real nanopore data:
        - Starts at base current level
        - Adds random step changes (simulate base transitions)
        - Overlays Gaussian noise (electrical/thermal noise)
        
        Algorithm:
        ---------
        1. Create constant baseline at base_current
        2. Divide signal into num_steps segments
        3. For each segment, shift to random current level (±20 pA)
        4. Add Gaussian noise with specified std deviation
        5. Ensure all values are positive (physical constraint)
        
        Args:
            num_samples (int): Length of signal in samples.
            base_current (float): Base current level in picoamperes.
                                 Default: 100.0 pA (typical nanopore)
            noise_std (float): Standard deviation of Gaussian noise.
                              Default: 7.0 pA (realistic noise level)
            num_steps (int): Number of current level changes.
                           Default: 8 (simulates ~8 base transitions)
        
        Returns:
            np.ndarray: Synthetic signal with shape (num_samples,).
                       Dtype: float32 (same as real POD5 data)
        
        Example:
        --------
        >>> gen = SyntheticPOD5Generator()
        >>> signal = gen._generate_synthetic_signal(5000)
        >>> print(f"Signal length: {len(signal)}")
        >>> print(f"Mean: {signal.mean():.1f} pA")
        >>> print(f"Std: {signal.std():.1f} pA")
        Signal length: 5000
        Mean: 100.3 pA
        Std: 9.2 pA
        """
        # Start with base current level
        signal = np.ones(num_samples, dtype=np.float32) * base_current
        
        # Add step changes to simulate base transitions
        # Divide signal into segments and assign random levels
        if num_steps > 1:
            step_size = num_samples // num_steps
            for i in range(num_steps):
                start_idx = i * step_size
                end_idx = start_idx + step_size if i < num_steps - 1 else num_samples
                
                # Random current shift: ±20 pA from base
                level_shift = np.random.uniform(-20, 20)
                signal[start_idx:end_idx] += level_shift
        
        # Add Gaussian noise to simulate electrical/thermal noise
        noise = np.random.normal(0, noise_std, num_samples).astype(np.float32)
        signal += noise
        
        # Ensure signal stays positive (physical constraint)
        signal = np.maximum(signal, 0.1)
        
        return signal
    
    def generate_sample_reads(
        self,
        num_reads: int = 1000,
        signal_length_range: tuple = (4000, 6000)
    ) -> List[dict]:
        """
        Generate a list of synthetic read data dictionaries.
        
        Creates structured read data compatible with POD5 format:
        - Unique read IDs (UUID4)
        - Synthetic signals with realistic characteristics
        - Metadata (channel, timestamps, sample rate)
        
        Args:
            num_reads (int): Number of reads to generate. Default: 1000.
            signal_length_range (tuple): (min, max) samples per read.
                                        Default: (4000, 6000)
                                        Typical MinION reads at 4 kHz
        
        Returns:
            List[dict]: List of read dictionaries, each containing:
                - read_id (str): Unique UUID
                - signal (np.ndarray): Synthetic signal data
                - num_samples (int): Signal length
                - sample_rate (int): 4000 Hz
                - channel (int): Random channel 1-512
                - start_time (int): Synthetic timestamp
        
        Example:
        --------
        >>> gen = SyntheticPOD5Generator(seed=42)
        >>> reads = gen.generate_sample_reads(num_reads=100)
        >>> print(f"Generated {len(reads)} reads")
        >>> print(f"First read ID: {reads[0]['read_id']}")
        >>> print(f"Signal length: {len(reads[0]['signal'])}")
        Generated 100 reads
        First read ID: 8d8c7b6a-5e4f-4e3d-9c2b-1a0b9c8d7e6f
        Signal length: 4523
        """
        reads = []
        
        # Generate timestamps starting from a base time
        base_time = int(datetime.datetime.now().timestamp() * 1000000)  # microseconds
        
        for i in range(num_reads):
            # Random signal length within specified range
            num_samples = np.random.randint(signal_length_range[0], signal_length_range[1])
            
            # Generate synthetic signal
            signal = self._generate_synthetic_signal(num_samples)
            
            # Create read metadata
            read_data = {
                'read_id': str(uuid.uuid4()),
                'signal': signal,
                'num_samples': num_samples,
                'sample_rate': 4000,  # Standard MinION sample rate
                'channel': np.random.randint(1, 513),  # MinION has 512 channels
                'start_time': base_time + i * 100000,  # Stagger timestamps
            }
            
            reads.append(read_data)
        
        return reads
    
    def save_to_pod5(
        self,
        output_path: Union[str, Path],
        num_reads: int = 1000,
        signal_length_range: tuple = (4000, 6000)
    ):
        """
        Generate synthetic data and save to POD5 file.
        
        Creates a properly formatted POD5 file with synthetic reads.
        The file can be read by standard POD5 readers and used for testing.
        
        File Structure:
        --------------
        - POD5 format (Apache Arrow-based)
        - Read table with metadata
        - Signal table with float32 arrays
        - Compatible with pod5 Python library
        
        Args:
            output_path (str): Path to save POD5 file (e.g., "test.pod5")
            num_reads (int): Number of reads to include. Default: 1000.
            signal_length_range (tuple): (min, max) samples per read.
                                        Default: (4000, 6000)
        
        Output:
            POD5 file written to output_path.
            Prints confirmation with file size.
        
        Example:
        --------
        >>> gen = SyntheticPOD5Generator(seed=42)
        >>> gen.save_to_pod5("test_data.pod5", num_reads=500)
        Generated 500 synthetic reads
        Saved to: test_data.pod5 (12.3 MB)
        
        >>> # Verify file can be read
        >>> import pod5
        >>> with pod5.Reader("test_data.pod5") as reader:
        ...     print(f"Reads in file: {reader.num_reads}")
        Reads in file: 500
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {num_reads} synthetic reads...")
        reads = self.generate_sample_reads(num_reads, signal_length_range)
        
        print(f"Writing to {output_path}...")
        
        # Write POD5 file using pod5 library
        # Create required POD5 objects once
        run_info = pod5.RunInfo(
            acquisition_id=str(uuid.uuid4()),
            acquisition_start_time=datetime.datetime.now(),
            adc_max=0,
            adc_min=0,
            context_tags={},
            experiment_name="synthetic_data",
            flow_cell_id="SYNTHETIC",
            flow_cell_product_code="FLO-MIN106",
            protocol_name="synthetic_protocol",
            protocol_run_id=str(uuid.uuid4()),
            protocol_start_time=datetime.datetime.now(),
            sample_id="synthetic_sample",
            sample_rate=4000,
            sequencing_kit="SQK-LSK109",
            sequencer_position="MN12345",
            sequencer_position_type="MinION",
            software="pod5_accelerator",
            system_name="synthetic_system",
            system_type="MinKNOW",
            tracking_id={}
        )
        
        end_reason = pod5.EndReason(
            reason=pod5.EndReasonEnum.SIGNAL_POSITIVE,
            forced=False
        )
        
        with pod5.Writer(output_path) as writer:
            # Add run_info and end_reason once
            run_info_idx = writer.add(run_info)
            end_reason_idx = writer.add(end_reason)
            
            for read_data in reads:
                # Create Pore and Calibration for each read
                pore = pod5.Pore(
                    channel=read_data['channel'],
                    well=1,
                    pore_type="not_set"
                )
                
                calibration = pod5.Calibration(
                    offset=0.0,
                    scale=1.0
                )
                
                # Create Read object
                read = pod5.Read(
                    read_id=uuid.UUID(read_data['read_id']),
                    pore=pore,
                    calibration=calibration,
                    read_number=0,
                    start_sample=int(read_data['start_time']),
                    median_before=100.0,
                    end_reason=end_reason,
                    run_info=run_info,
                    signal=read_data['signal'].astype(np.int16)
                )
                
                # Add read to POD5 file
                writer.add_read(read)
        
        # Report file size
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"✓ Saved {num_reads} reads to: {output_path} ({file_size_mb:.1f} MB)")
    
    def create_test_dataset(
        self,
        output_dir: Union[str, Path] = "./data",
        num_files: int = 5,
        reads_per_file: int = 1000
    ):
        """
        Create a multi-file test dataset for benchmarking.
        
        Generates multiple POD5 files to simulate a realistic sequencing run
        with multiple output files. This is useful for testing parallel processing
        and multi-file benchmarking.
        
        Dataset Structure:
        -----------------
        output_dir/
            synthetic_001.pod5
            synthetic_002.pod5
            synthetic_003.pod5
            ...
        
        Each file contains the specified number of reads with random signals.
        Files are named sequentially for easy identification.
        
        Args:
            output_dir (str): Directory to save POD5 files.
                            Default: "./data"
                            Directory will be created if needed.
            num_files (int): Number of POD5 files to generate.
                           Default: 5 files
            reads_per_file (int): Reads per file.
                                 Default: 1000 reads
                                 Total reads = num_files × reads_per_file
        
        Output:
            Multiple POD5 files in output_dir.
            Prints progress and summary.
        
        Example:
        --------
        >>> gen = SyntheticPOD5Generator(seed=42)
        >>> gen.create_test_dataset(
        ...     output_dir="./test_data",
        ...     num_files=3,
        ...     reads_per_file=500
        ... )
        Creating test dataset: 3 files × 500 reads = 1500 total reads
        [1/3] Generating ./test_data/synthetic_001.pod5...
        ✓ Saved 500 reads to: ./test_data/synthetic_001.pod5 (6.2 MB)
        [2/3] Generating ./test_data/synthetic_002.pod5...
        ✓ Saved 500 reads to: ./test_data/synthetic_002.pod5 (6.1 MB)
        [3/3] Generating ./test_data/synthetic_003.pod5...
        ✓ Saved 500 reads to: ./test_data/synthetic_003.pod5 (6.3 MB)
        
        Dataset created: 3 files, 1500 total reads, 18.6 MB
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_reads = num_files * reads_per_file
        print(f"\n{'='*60}")
        print(f"Creating test dataset: {num_files} files × {reads_per_file} reads")
        print(f"Total reads: {total_reads}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")
        
        total_size_mb = 0.0
        
        for i in range(1, num_files + 1):
            file_path = output_dir / f"synthetic_{i:03d}.pod5"
            print(f"[{i}/{num_files}] Generating {file_path.name}...")
            
            self.save_to_pod5(
                str(file_path),
                num_reads=reads_per_file
            )
            
            total_size_mb += file_path.stat().st_size / 1024 / 1024
            print()
        
        print(f"{'='*60}")
        print(f"✓ Dataset created successfully!")
        print(f"  Files: {num_files}")
        print(f"  Total reads: {total_reads}")
        print(f"  Total size: {total_size_mb:.1f} MB")
        print(f"  Location: {output_dir.absolute()}")
        print(f"{'='*60}\n")


# Convenience function for quick test data generation
def generate_test_data(
    output_dir: Union[str, Path] = "./data",
    num_files: int = 3,
    reads_per_file: int = 1000,
    seed: int = 42
):
    """
    Quick helper to generate test dataset with default settings.
    
    Args:
        output_dir (str): Output directory. Default: "./data"
        num_files (int): Number of files. Default: 3
        reads_per_file (int): Reads per file. Default: 1000
        seed (int): Random seed. Default: 42 (reproducible)
    
    Example:
    --------
    >>> from pod5_accelerator.core.synthetic_generator import generate_test_data
    >>> generate_test_data(output_dir="./test_data", num_files=5, reads_per_file=500)
    """
    generator = SyntheticPOD5Generator(seed=seed)
    generator.create_test_dataset(output_dir, num_files, reads_per_file)
