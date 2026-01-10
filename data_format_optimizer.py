"""
DATA FORMAT OPTIMIZATION MODULE
Based on Code Analysis Recommendations (Section 2.3.1)

Features:
- HDF5 format for better I/O performance
- Zarr format for cloud-native storage
- Memory mapping for large datasets
- Chunking and compression optimization
- Automatic format conversion
"""

import os
import h5py
import zarr
import numpy as np
import torch
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFormatOptimizer:
    """
    OPTIMIZED DATA FORMAT HANDLER
    Implements HDF5/Zarr recommendations from code analysis
    """
    
    def __init__(self, 
                 preferred_format: str = 'hdf5',
                 compression: str = 'lzf',
                 chunk_size: Optional[Tuple[int, ...]] = None):
        """
        Initialize data format optimizer
        
        Args:
            preferred_format: 'hdf5', 'zarr', or 'tensor'
            compression: Compression algorithm ('lzf', 'gzip', 'szip')
            chunk_size: Chunk size for HDF5/Zarr (None for auto)
        """
        self.preferred_format = preferred_format
        self.compression = compression
        self.chunk_size = chunk_size
        
        if preferred_format not in ['hdf5', 'zarr', 'tensor']:
            raise ValueError(f"Unsupported format: {preferred_format}")
        
        logger.info(f"Data format optimizer initialized: {preferred_format}")
    
    def optimize_tensor_for_storage(self, 
                                  tensor: torch.Tensor,
                                  output_path: str,
                                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Convert and optimize tensor for efficient storage
        
        Args:
            tensor: Input tensor to optimize
            output_path: Output file path (extension will be adjusted)
            metadata: Optional metadata to store
            
        Returns:
            Path to optimized file
        """
        output_path = Path(output_path)
        if self.preferred_format == 'hdf5':
            output_path = output_path.with_suffix('.h5')
        elif self.preferred_format == 'zarr':
            output_path = output_path.with_suffix('.zarr')
        else:
            output_path = output_path.with_suffix('.pt')
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        if self.preferred_format == 'hdf5':
            self._save_as_hdf5(tensor, output_path, metadata)
        elif self.preferred_format == 'zarr':
            self._save_as_zarr(tensor, output_path, metadata)
        else:
            self._save_as_tensor(tensor, output_path, metadata)
        
        elapsed = time.time() - start_time
        file_size = output_path.stat().st_size / (1024**2)
        
        logger.info(f"Saved {tensor.shape} tensor as {self.preferred_format}")
        logger.info(f"   File: {output_path}")
        logger.info(f"   Size: {file_size:.1f} MB")
        logger.info(f"   Time: {elapsed:.2f}s")
        
        return str(output_path)
    
    def _save_as_hdf5(self, tensor: torch.Tensor, path: Path, metadata: Optional[Dict] = None):
        """Save tensor as HDF5 with optimization"""
        data = tensor.cpu().numpy()
        
        chunk_size = self.chunk_size
        if chunk_size is None:
            if len(data.shape) == 5:
                chunk_size = (1, data.shape[1], data.shape[2], 
                            min(256, data.shape[3]), min(256, data.shape[4]))
            else:
                chunk_size = True
        
        with h5py.File(path, 'w') as f:
            dataset = f.create_dataset(
                'data',
                data=data,
                compression=self.compression,
                chunks=chunk_size,
                shuffle=True,
                fletcher32=True
            )
            
            if metadata:
                for key, value in metadata.items():
                    dataset.attrs[key] = value
            
            dataset.attrs['dtype'] = str(tensor.dtype)
            dataset.attrs['device'] = str(tensor.device)
            dataset.attrs['shape'] = data.shape
    
    def _save_as_zarr(self, tensor: torch.Tensor, path: Path, metadata: Optional[Dict] = None):
        """Save tensor as Zarr with optimization"""
        data = tensor.cpu().numpy()
        
        chunk_size = self.chunk_size
        if chunk_size is None:
            if len(data.shape) == 5:
                chunk_size = (1, data.shape[1], data.shape[2], 
                            min(256, data.shape[3]), min(256, data.shape[4]))
            else:
                chunk_size = None
        
        z = zarr.open(str(path), mode='w', shape=data.shape, dtype=data.dtype,
                     chunks=chunk_size, compressor=zarr.Blosc(cname='lz4', clevel=5))
        
        z[:] = data
        
        if metadata:
            z.attrs.update(metadata)
        
        z.attrs['dtype'] = str(tensor.dtype)
        z.attrs['device'] = str(tensor.device)
        z.attrs['shape'] = data.shape
    
    def _save_as_tensor(self, tensor: torch.Tensor, path: Path, metadata: Optional[Dict] = None):
        """Save as PyTorch tensor with metadata - supports both old and new formats"""
        if tensor.dim() == 3:
            save_dict = {
                'frame_data': tensor,
                'metadata': metadata or {},
                'config': {
                    'channels': tensor.shape[0],
                    'resolution': (tensor.shape[1], tensor.shape[2]),
                    'format_version': '3.0_single_frame'
                },
                'version': '3.0_single_frame'
            }
        else:
            save_dict = {
                'data': tensor,
                'metadata': metadata or {}
            }

        torch.save(save_dict, path)
    
    def load_optimized_tensor(self, file_path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Load tensor from optimized format
        
        Args:
            file_path: Path to optimized file
            
        Returns:
            Tuple of (tensor, metadata)
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.h5':
            return self._load_from_hdf5(file_path)
        elif file_path.suffix == '.zarr':
            return self._load_from_zarr(file_path)
        elif file_path.suffix == '.pt':
            return self._load_from_tensor(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _load_from_hdf5(self, path: Path) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Load tensor from HDF5"""
        with h5py.File(path, 'r') as f:
            dataset = f['data']
            
            data = dataset[:]
            
            metadata = dict(dataset.attrs)
            
            tensor = torch.from_numpy(data)
            
            if 'dtype' in metadata:
                original_dtype = metadata['dtype']
                if 'float32' in original_dtype:
                    tensor = tensor.float()
                elif 'float16' in original_dtype:
                    tensor = tensor.half()
        
        return tensor, metadata
    
    def _load_from_zarr(self, path: Path) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Load tensor from Zarr"""
        z = zarr.open(str(path), mode='r')
        
        data = z[:]
        
        metadata = dict(z.attrs)
        
        tensor = torch.from_numpy(data)
        
        if 'dtype' in metadata:
            original_dtype = metadata['dtype']
            if 'float32' in original_dtype:
                tensor = tensor.float()
            elif 'float16' in original_dtype:
                tensor = tensor.half()
        
        return tensor, metadata
    
    def _load_from_tensor(self, path: Path) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Load tensor from PyTorch format - handles both old and new formats"""
        try:
            save_dict = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            if "weights_only" in str(e):
                save_dict = torch.load(path, map_location='cpu')
            else:
                raise e

        if isinstance(save_dict, dict) and 'frame_data' in save_dict:
            return save_dict['frame_data'], save_dict.get('metadata', {})
        elif isinstance(save_dict, dict) and 'data' in save_dict:
            return save_dict['data'], save_dict.get('metadata', {})
        elif isinstance(save_dict, torch.Tensor):
            return save_dict, {}
        else:
            raise ValueError(f"Unknown tensor format in {path}: {type(save_dict)}")
    
    def create_memory_mapped_dataset(self, 
                                   file_paths: list,
                                   output_path: str,
                                   max_workers: int = 4) -> str:
        """
        Create memory-mapped dataset from multiple files
        
        Args:
            file_paths: List of input file paths
            output_path: Output path for combined dataset
            max_workers: Number of parallel workers
            
        Returns:
            Path to memory-mapped dataset
        """
        logger.info(f"Creating memory-mapped dataset from {len(file_paths)} files...")
        
        total_samples = 0
        sample_shape = None
        
        for file_path in file_paths[:5]:
            try:
                tensor, _ = self.load_optimized_tensor(file_path)
                if sample_shape is None:
                    sample_shape = tensor.shape[1:]
                total_samples += tensor.shape[0]
            except Exception as e:
                logger.warning(f"Skipping {file_path}: {e}")
        
        if sample_shape is None:
            raise ValueError("Could not determine sample shape from input files")
        
        avg_samples_per_file = total_samples / min(5, len(file_paths))
        estimated_total = int(avg_samples_per_file * len(file_paths))
        
        logger.info(f"Estimated dataset size: {estimated_total} samples of shape {sample_shape}")
        
        output_path = Path(output_path)
        if self.preferred_format == 'hdf5':
            output_path = output_path.with_suffix('.h5')
            return self._create_hdf5_memmap(file_paths, output_path, sample_shape, estimated_total, max_workers)
        elif self.preferred_format == 'zarr':
            output_path = output_path.with_suffix('.zarr')
            return self._create_zarr_memmap(file_paths, output_path, sample_shape, estimated_total, max_workers)
        else:
            raise NotImplementedError("Memory mapping not supported for tensor format")
    
    def _create_hdf5_memmap(self, file_paths, output_path, sample_shape, estimated_total, max_workers):
        """Create HDF5 memory-mapped dataset"""
        full_shape = (estimated_total,) + sample_shape
        
        with h5py.File(output_path, 'w') as f:
            chunk_shape = (1,) + sample_shape
            dataset = f.create_dataset(
                'data',
                shape=full_shape,
                dtype=np.float32,
                chunks=chunk_shape,
                compression=self.compression,
                maxshape=(None,) + sample_shape
            )
            
            current_idx = 0
            
            def process_file(file_path):
                try:
                    tensor, metadata = self.load_optimized_tensor(file_path)
                    return tensor.cpu().numpy(), metadata
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
                    return None, None
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for result in executor.map(process_file, file_paths):
                    data, metadata = result
                    if data is not None:
                        end_idx = current_idx + data.shape[0]
                        
                        if end_idx > dataset.shape[0]:
                            dataset.resize((end_idx,) + sample_shape)
                        
                        dataset[current_idx:end_idx] = data
                        current_idx = end_idx
            
            dataset.resize((current_idx,) + sample_shape)
            
            dataset.attrs['num_samples'] = current_idx
            dataset.attrs['sample_shape'] = sample_shape
            dataset.attrs['format'] = 'memory_mapped_hdf5'
        
        logger.info(f"Created HDF5 memory-mapped dataset: {output_path}")
        logger.info(f"   Final size: {current_idx} samples")
        
        return str(output_path)
    
    def _create_zarr_memmap(self, file_paths, output_path, sample_shape, estimated_total, max_workers):
        """Create Zarr memory-mapped dataset"""
        raise NotImplementedError("Zarr memory mapping implementation pending")


def benchmark_formats(tensor: torch.Tensor, output_dir: str = "./benchmark") -> Dict[str, Dict[str, float]]:
    """
    Benchmark different data formats for I/O performance
    
    Args:
        tensor: Test tensor
        output_dir: Directory for benchmark files
        
    Returns:
        Dictionary with benchmark results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    formats = ['hdf5', 'zarr', 'tensor']
    results = {}
    
    logger.info(f"Benchmarking data formats with tensor shape {tensor.shape}")
    
    for fmt in formats:
        logger.info(f"Testing {fmt}...")
        
        optimizer = DataFormatOptimizer(preferred_format=fmt)
        
        start_time = time.time()
        file_path = optimizer.optimize_tensor_for_storage(
            tensor, 
            str(output_dir / f"benchmark_test.{fmt}")
        )
        write_time = time.time() - start_time
        
        start_time = time.time()
        loaded_tensor, metadata = optimizer.load_optimized_tensor(file_path)
        read_time = time.time() - start_time
        
        file_size = Path(file_path).stat().st_size / (1024**2)
        
        results[fmt] = {
            'write_time': write_time,
            'read_time': read_time,
            'file_size_mb': file_size,
            'total_time': write_time + read_time
        }
        
        if not torch.allclose(tensor, loaded_tensor, atol=1e-6):
            logger.warning(f"Data integrity check failed for {fmt}")
    
    # Print results
    print("\n" + "="*60)
    print("DATA FORMAT BENCHMARK RESULTS")
    print("="*60)
    print(f"{'Format':<10} {'Write(s)':<10} {'Read(s)':<10} {'Size(MB)':<12} {'Total(s)':<10}")
    print("-"*60)
    
    for fmt, metrics in results.items():
        print(f"{fmt:<10} {metrics['write_time']:<10.3f} {metrics['read_time']:<10.3f} "
              f"{metrics['file_size_mb']:<12.1f} {metrics['total_time']:<10.3f}")
    
    print("="*60)
    
    return results


if __name__ == "__main__":
    print("Testing data format optimization...")
    
    test_tensor = torch.randn(2, 5, 8, 256, 256, dtype=torch.float32)
    
    for fmt in ['hdf5', 'zarr', 'tensor']:
        print(f"\nTesting {fmt} format...")
        
        optimizer = DataFormatOptimizer(preferred_format=fmt)
        
        file_path = optimizer.optimize_tensor_for_storage(
            test_tensor, 
            f"./test_output.{fmt}",
            metadata={'test': True, 'channels': 5}
        )
        
        loaded_tensor, metadata = optimizer.load_optimized_tensor(file_path)
        
        if torch.allclose(test_tensor, loaded_tensor):
            print(f"   {fmt} format test passed")
        else:
            print(f"   {fmt} format test failed")
    
    print("\nRunning format benchmark...")
    benchmark_results = benchmark_formats(test_tensor)
    
    print("\nData format optimization test completed")
