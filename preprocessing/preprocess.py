import h5py
import numpy as np
import torch
import torch.nn.functional as F
import glob
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import warnings
from tqdm import tqdm
import gc
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings('ignore')

class UltraFastConfig:
    """Ultra-optimized configuration for Mac Mini M4 (10 cores, 16GB RAM)"""
    # FIXED Paths for your 35GB dataset
    HDF5_DIR: str = '/media/mayank/ExtHDD/ISRO_BAH/dataset/INSAT-3DS/JUN25/ASIA_MER'  # Your H5 files directory with 1,440 files
    OUTPUT_DIR: str = 'preprocessed'  # Your output directory
    
    # Channel selection (5 key channels only)
    SELECTED_CHANNELS: List[str] = ['VIS', 'WV', 'SWIR', 'TIR1', 'TIR2']
    
    # FIXED Channel configurations based on actual INSAT L1C H5 structure
    CHANNEL_CONFIGS: Dict[str, Dict] = {
        'VIS': {
            'type': 'visible',
            'image_path': 'IMG_VIS',
            'priority_paths': ['IMG_VIS_ALBEDO', 'IMG_VIS_RADIANCE', 'IMG_VIS'],  # Try calibrated first, fallback to raw
            'priority_units': ['albedo', 'radiance', 'counts'],
            'norm_ranges': {'albedo': [0, 100], 'radiance': [0, 100], 'counts': [0, 1023]},
            'raw_range': [0, 1023],  # Raw count range for direct processing
            'physical_range': [0, 100]  # Physical units range
        },
        'WV': {
            'type': 'infrared',
            'image_path': 'IMG_WV',
            'priority_paths': ['IMG_WV_TEMP', 'IMG_WV_RADIANCE', 'IMG_WV'],
            'priority_units': ['temperature', 'radiance', 'counts'],
            'norm_ranges': {'temperature': [180, 340], 'radiance': [0, 20], 'counts': [0, 1023]},
            'raw_range': [0, 1023],
            'physical_range': [180, 340]
        },
        'SWIR': {
            'type': 'visible',
            'image_path': 'IMG_SWIR',
            'priority_paths': ['IMG_SWIR_ALBEDO', 'IMG_SWIR_RADIANCE', 'IMG_SWIR'],  # Try calibrated first, fallback to raw
            'priority_units': ['albedo', 'radiance', 'counts'],
            'norm_ranges': {'albedo': [0, 100], 'radiance': [0, 50], 'counts': [0, 1023]},
            'raw_range': [0, 1023],
            'physical_range': [0, 100]
        },
        'TIR1': {
            'type': 'infrared',
            'image_path': 'IMG_TIR1',
            'priority_paths': ['IMG_TIR1_TEMP', 'IMG_TIR1_RADIANCE', 'IMG_TIR1'],
            'priority_units': ['temperature', 'radiance', 'counts'],
            'norm_ranges': {'temperature': [180, 340], 'radiance': [0, 10], 'counts': [0, 1023]},
            'raw_range': [0, 1023],
            'physical_range': [180, 340]
        },
        'TIR2': {
            'type': 'infrared',
            'image_path': 'IMG_TIR2',
            'priority_paths': ['IMG_TIR2_TEMP', 'IMG_TIR2_RADIANCE', 'IMG_TIR2'],
            'priority_units': ['temperature', 'radiance', 'counts'],
            'norm_ranges': {'temperature': [180, 340], 'radiance': [0, 8], 'counts': [0, 1023]},
            'raw_range': [0, 1023],
            'physical_range': [180, 340]
        }
    }
    
    # Sequence parameters
    INPUT_FRAMES: int = 8
    FORECAST_FRAMES: int = 4
    TOTAL_SEQUENCE_LENGTH: int = 12
    
    # Processing parameters optimized for Mac Mini M4
    TARGET_RESOLUTION: tuple = (720, 720)
    MAX_WORKERS: int = 8  # Optimal for M4 (10 cores - 2 for system)
    NATIVE_RESOLUTION: tuple = (1616, 1737)  # INSAT-3DS native resolution
    
    # Ultra-fast mode flags
    ULTRA_FAST_MODE: bool = True
    NO_LOGGING: bool = True
    NO_VALIDATION: bool = True
    FAST_RESIZE_ONLY: bool = True

config = UltraFastConfig()

# Minimal logging setup
if not config.NO_LOGGING:
    logging.basicConfig(level=logging.ERROR, format='%(message)s')
    logger = logging.getLogger()
else:
    logger = logging.getLogger()
    logger.disabled = True

def fast_extract_channel(hdf5_file: h5py.File, channel_name: str) -> Optional[np.ndarray]:
    """FIXED channel extraction - prioritize direct raw data for reliability"""
    try:
        channel_config = config.CHANNEL_CONFIGS[channel_name]
        image_path = channel_config['image_path']

        # FIXED: Use direct raw data processing first for reliability
        if image_path in hdf5_file:
            try:
                raw_data = hdf5_file[image_path][...]

                # Remove singleton dimensions if present
                if raw_data.ndim == 3 and raw_data.shape[0] == 1:
                    raw_data = raw_data[0]

                # Convert raw counts to physical units
                raw_range = channel_config['raw_range']
                physical_range = channel_config['physical_range']

                # Handle fill values (typically 0 for INSAT)
                valid_mask = raw_data != 0

                # Convert to physical units
                physical_data = np.zeros_like(raw_data, dtype=np.float32)
                physical_data[valid_mask] = (
                    (raw_data[valid_mask].astype(np.float32) - raw_range[0]) /
                    (raw_range[1] - raw_range[0]) *
                    (physical_range[1] - physical_range[0]) + physical_range[0]
                )
                physical_data[np.logical_not(valid_mask)] = physical_range[0]

                # Apply quantile-based clipping before min-max scaling
                # Use 5th and 95th percentiles to remove outliers
                valid_data = physical_data[valid_mask]
                if len(valid_data) > 0:
                    q1, q99 = np.percentile(valid_data, [1, 99])
                    physical_data = np.clip(physical_data, q1, q99)
                    # Use the quantile-clipped range for min-max scaling
                    scale_min, scale_max = q1, q99
                else:
                    # Fallback to original range if no valid data
                    scale_min, scale_max = physical_range[0], physical_range[1]

                # Normalize to [-1, 1] for neural network training using quantile-clipped range
                normalized = 2.0 * (physical_data - scale_min) / (scale_max - scale_min) - 1.0
                normalized = np.clip(normalized, -1.0, 1.0)

                return normalized.astype(np.float32)

            except Exception as e:
                if not config.NO_LOGGING:
                    print(f"Warning: Direct extraction failed for {channel_name}: {e}")

        # Fallback to lookup table approach if direct method fails
        priority_paths = channel_config['priority_paths']
        priority_units = channel_config['priority_units']

        for lookup_path, unit_type in zip(priority_paths, priority_units):
            if lookup_path in hdf5_file and image_path in hdf5_file:
                try:
                    # Extract image indices and lookup table
                    image_data = hdf5_file[image_path]
                    lookup_data = hdf5_file[lookup_path]

                    if len(image_data.shape) == 3:
                        image_indices = image_data[0].astype(np.int32)
                    elif len(image_data.shape) == 2:
                        image_indices = image_data[:].astype(np.int32)
                    else:
                        continue

                    if len(lookup_data.shape) != 1:
                        continue

                    lookup_table = lookup_data[:].astype(np.float32)

                    # Fast lookup without validation
                    valid_mask = (image_indices >= 0) & (image_indices < len(lookup_table))
                    result = np.full(image_indices.shape, np.nan, dtype=np.float32)
                    result[valid_mask] = lookup_table[image_indices[valid_mask]]

                    # Handle fill values
                    fill_value = lookup_data.attrs.get('_FillValue')
                    if fill_value is not None:
                        if isinstance(fill_value, np.ndarray):
                            fill_value = fill_value[0]
                        result[result == fill_value] = np.nan

                    # Apply quantile-based clipping before min-max scaling
                    # Use 5th and 95th percentiles to remove outliers
                    valid_data = result[~np.isnan(result)]
                    if len(valid_data) > 0:
                        q1, q99 = np.percentile(valid_data, [1, 99])
                        result = np.clip(result, q1, q99)
                        # Use the quantile-clipped range for min-max scaling
                        scale_min, scale_max = q1, q99
                    else:
                        # Fallback to original norm range if no valid data
                        norm_range = channel_config['norm_ranges'][unit_type]
                        scale_min, scale_max = norm_range[0], norm_range[1]

                    # Fast normalization using quantile-clipped range
                    normalized = 2.0 * (result - scale_min) / (scale_max - scale_min) - 1.0
                    normalized[np.isnan(result)] = -1.0

                    return normalized.astype(np.float32)

                except:
                    continue

        return None
    except:
        return None

def fast_resize(data: np.ndarray) -> np.ndarray:
    """Ultra-fast resize using PyTorch interpolation"""
    try:
        data_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()
        resized = F.interpolate(data_tensor, size=config.TARGET_RESOLUTION, mode='bilinear', align_corners=False)
        return resized.squeeze().numpy()
    except:
        return np.full(config.TARGET_RESOLUTION, -1.0, dtype=np.float32)

def process_single_file_ultrafast(file_path: str) -> Optional[np.ndarray]:
    """Process a single HDF5 file with maximum speed"""
    try:
        with h5py.File(file_path, 'r') as hdf5_file:
            channels = []
            
            # Extract all channels
            for channel_name in config.SELECTED_CHANNELS:
                channel_data = fast_extract_channel(hdf5_file, channel_name)
                if channel_data is not None:
                    # Fast resize
                    resized_data = fast_resize(channel_data)
                    channels.append(resized_data)
                else:
                    # Fill with default values if channel missing
                    channels.append(np.full(config.TARGET_RESOLUTION, -1.0, dtype=np.float32))
            
            if len(channels) == len(config.SELECTED_CHANNELS):
                return np.stack(channels, axis=0)
            else:
                return None
                
    except:
        return None

def process_single_file_with_metadata(file_path: str) -> Optional[Dict]:
    """Process a single HDF5 file and return data with metadata"""
    try:
        # Process the file
        frame_data = process_single_file_ultrafast(file_path)
        if frame_data is None:
            return None

        # Extract timestamp from filename
        timestamp = extract_timestamp(file_path)
        if timestamp is None:
            return None

        # Calculate quality score based on valid pixels
        valid_pixels = np.sum(frame_data > -0.99)  # Count non-fill values
        total_pixels = frame_data.size
        quality_score = valid_pixels / total_pixels

        # Create data package
        data_package = {
            'frame_data': frame_data,  # Shape: [C, H, W]
            'metadata': {
                'file_path': file_path,
                'filename': os.path.basename(file_path),
                'timestamp': timestamp,
                'quality_score': quality_score,
                'channels': config.SELECTED_CHANNELS,
                'resolution': config.TARGET_RESOLUTION,
                'processing_time': datetime.now().isoformat()
            },
            'config': {
                'selected_channels': config.SELECTED_CHANNELS,
                'channel_configs': config.CHANNEL_CONFIGS,
                'target_resolution': config.TARGET_RESOLUTION,
                'area_extent': [80.0, 5.0, 100.0, 25.0]  # From original config
            },
            'version': '3.0_single_frame'
        }

        return data_package

    except Exception as e:
        if not config.NO_LOGGING:
            print(f"Error processing {file_path}: {e}")
        return None

def save_single_frame(data_package: Dict, output_dir: str) -> Optional[str]:
    """Save single frame data to PyTorch file"""
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Convert frame data to tensor
        frame_tensor = torch.from_numpy(data_package['frame_data'])

        # Create final data package
        final_package = {
            'frame_data': frame_tensor,  # Shape: [C, H, W]
            'metadata': data_package['metadata'],
            'config': data_package['config'],
            'version': data_package['version']
        }

        # Create filename based on timestamp
        timestamp = data_package['metadata']['timestamp']
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M")
        filename = f"frame_{timestamp_str}.pt"
        output_path = os.path.join(output_dir, filename)

        torch.save(final_package, output_path)
        return output_path

    except Exception as e:
        if not config.NO_LOGGING:
            print(f"Error saving frame: {e}")
        return None

def extract_timestamp(file_path: str) -> Optional[datetime]:
    """Extract timestamp from INSAT-3DS filename format: 3SIMG_DDMMMYYYY_HHMM_L1C_ASIA_MER_V01R00.h5"""
    try:
        filename = os.path.basename(file_path)

        # INSAT-3DS pattern: 3SIMG_01JUN2025_0000_L1C_ASIA_MER_V01R00.h5
        import re
        pattern = r'3SIMG_(\d{2})([A-Z]{3})(\d{4})_(\d{4})_'
        match = re.search(pattern, filename)

        if match:
            day, month_str, year, time_str = match.groups()

            # Convert month abbreviation to number
            month_map = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }

            if month_str in month_map:
                month = month_map[month_str]
                hour = int(time_str[:2])
                minute = int(time_str[2:])

                return datetime(int(year), month, int(day), hour, minute)

        return None
    except:
        return None

def validate_temporal_sequence(timestamps: List[datetime], sequence_id: str = "") -> tuple[bool, str]:
    """Validate strict 30-minute temporal consistency - NO missing frames allowed"""
    if len(timestamps) < 2:
        return True, ""

    # STRICT 30-minute policy: Each frame must be exactly 30 minutes apart
    EXPECTED_INTERVAL_MINUTES = 30.0
    TOLERANCE_MINUTES = 2.0  # Allow ±2 minutes tolerance for timing variations

    intervals = []
    for i in range(1, len(timestamps)):
        interval = (timestamps[i] - timestamps[i-1]).total_seconds() / 60.0
        intervals.append(interval)

    # Check EVERY interval must be approximately 30 minutes
    for i, interval in enumerate(intervals):
        if abs(interval - EXPECTED_INTERVAL_MINUTES) > TOLERANCE_MINUTES:
            # Return detailed rejection reason
            rejection_reason = f"Interval {i+1}: {interval:.1f} min (expected {EXPECTED_INTERVAL_MINUTES} ± {TOLERANCE_MINUTES})"
            return False, rejection_reason

    # All intervals passed - sequence is valid
    return True, ""

def create_sequences_ultrafast(file_list: List[str]) -> List[Dict]:
    """Create sequences with strict 30-minute temporal validation and step size of 1"""
    sequences = []

    # Statistics tracking
    total_potential_sequences = 0
    timestamp_extraction_failures = 0
    temporal_validation_failures = 0
    rejected_sequences_details = []

    total_length = config.TOTAL_SEQUENCE_LENGTH
    step_size = 1  # Maximum overlap - step by 1 file each time

    for i in range(0, len(file_list) - total_length + 1, step_size):
        total_potential_sequences += 1
        sequence_files = file_list[i:i + total_length]
        sequence_id = f"seq_{i:06d}"

        # Extract timestamps for temporal validation
        timestamps = []
        for file_path in sequence_files:
            timestamp = extract_timestamp(file_path)
            if timestamp is not None:
                timestamps.append(timestamp)
            else:
                timestamps = None
                break

        if timestamps is None or len(timestamps) != total_length:
            timestamp_extraction_failures += 1
            continue

        # Validate strict 30-minute temporal sequence
        is_valid, rejection_reason = validate_temporal_sequence(timestamps, sequence_id)
        if is_valid:
            sequences.append({
                'files': sequence_files,
                'timestamps': timestamps,
                'sequence_id': sequence_id,
                'start_time': timestamps[0],
                'end_time': timestamps[-1]
            })
        else:
            temporal_validation_failures += 1
            # Store detailed rejection info
            rejected_sequences_details.append({
                'sequence_id': sequence_id,
                'start_time': timestamps[0].strftime("%Y-%m-%d %H:%M"),
                'reason': rejection_reason
            })

    # Print detailed statistics
    if not config.NO_LOGGING:
        print(f"📊 STRICT 30-MINUTE SEQUENCE VALIDATION RESULTS (Step Size = 1):")
        print(f"   Total potential sequences: {total_potential_sequences}")
        print(f"   Timestamp extraction failures: {timestamp_extraction_failures}")
        print(f"   30-minute validation failures: {temporal_validation_failures}")
        print(f"   Valid 30-minute sequences: {len(sequences)}")
        print(f"   Success rate: {len(sequences)/total_potential_sequences*100:.1f}%")

        # Show first few rejection details
        if rejected_sequences_details:
            print(f"\n❌ SAMPLE REJECTIONS (first 10):")
            for detail in rejected_sequences_details[:10]:
                print(f"   {detail['sequence_id']} ({detail['start_time']}): {detail['reason']}")
            if len(rejected_sequences_details) > 10:
                print(f"   ... and {len(rejected_sequences_details) - 10} more rejections")

    return sequences

def main_single_frame_processing():
    """Process each HDF5 file individually into separate .pt files"""
    start_time = datetime.now()

    print("🚀 SINGLE FRAME SATELLITE PREPROCESSING")
    print("=" * 50)
    print(f"⚡ Processing mode: One .pt file per .h5 file")
    print(f"   - Quantile clipping: 1st-99th percentiles")
    print(f"   - Min-max scaling to [-1, 1]")
    print(f"   - {config.MAX_WORKERS} parallel workers")
    print(f"   - Target resolution: {config.TARGET_RESOLUTION}")
    print()

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Find files
    all_files = []
    for pattern in ['*.h5', '*.hdf5', '*.HDF5']:
        files = glob.glob(os.path.join(config.HDF5_DIR, pattern))
        all_files.extend(files)

    all_files = sorted(list(set(all_files)))
    print(f"📁 Found {len(all_files)} HDF5 files")

    if len(all_files) == 0:
        print("❌ No HDF5 files found")
        return

    # Process files in parallel
    successful_files = []
    failed_files = []

    print(f"🔥 Processing with {config.MAX_WORKERS} workers...")

    with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        # Submit all files
        future_to_file = {
            executor.submit(process_single_file_with_metadata, file_path): file_path
            for file_path in all_files
        }

        with tqdm(total=len(all_files), desc="Processing frames") as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]

                try:
                    data_package = future.result()
                    if data_package is not None:
                        # Save the processed frame
                        output_path = save_single_frame(data_package, config.OUTPUT_DIR)
                        if output_path:
                            successful_files.append(output_path)
                        else:
                            failed_files.append(file_path)
                    else:
                        failed_files.append(file_path)
                except Exception as e:
                    failed_files.append(file_path)
                    if not config.NO_LOGGING:
                        print(f"Error processing {os.path.basename(file_path)}: {e}")

                pbar.update(1)

                # Memory management
                if psutil.virtual_memory().percent > 85:
                    gc.collect()

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    print(f"\n🎉 SINGLE FRAME PROCESSING COMPLETE!")
    print(f"   Successfully processed: {len(successful_files)}/{len(all_files)} files")
    print(f"   Failed: {len(failed_files)} files")
    print(f"   Success rate: {len(successful_files)/len(all_files)*100:.1f}%")
    print(f"   Total time: {processing_time:.1f} seconds")
    print(f"   Speed: {len(successful_files)/processing_time:.2f} files/second")
    print(f"   Output directory: {config.OUTPUT_DIR}")

    if failed_files and not config.NO_LOGGING:
        print(f"\n❌ Failed files (first 10):")
        for failed_file in failed_files[:10]:
            print(f"   {os.path.basename(failed_file)}")
        if len(failed_files) > 10:
            print(f"   ... and {len(failed_files) - 10} more")

def test_fixed_extraction():
    """Test the fixed channel extraction on a sample file"""
    print("🧪 TESTING FIXED CHANNEL EXTRACTION")
    print("="*50)

    # Find a test H5 file
    test_files = glob.glob(os.path.join(config.HDF5_DIR, "*.h5"))
    if not test_files:
        print("❌ No H5 files found for testing")
        return False

    test_file = test_files[0]
    print(f"📁 Testing with: {os.path.basename(test_file)}")

    try:
        with h5py.File(test_file, 'r') as hdf5_file:
            print(f"📋 H5 structure: {list(hdf5_file.keys())}")

            success_count = 0
            for channel_name in config.SELECTED_CHANNELS:
                print(f"\n🔍 Testing {channel_name} extraction...")

                channel_data = fast_extract_channel(hdf5_file, channel_name)

                if channel_data is not None:
                    print(f"  ✅ Success! Shape: {channel_data.shape}")
                    print(f"  📊 Range: [{channel_data.min():.3f}, {channel_data.max():.3f}]")
                    print(f"  📈 Std: {channel_data.std():.6f}")

                    # Check for corruption indicators
                    if channel_name in ['VIS', 'SWIR'] and channel_data.std() < 0.01:
                        print(f"  ⚠️  Low variance detected - possible corruption")
                    else:
                        print(f"  ✅ Good data quality")

                    success_count += 1
                else:
                    print(f"  ❌ Failed to extract {channel_name}")

            print(f"\n📊 EXTRACTION TEST RESULTS:")
            print(f"  ✅ Successful: {success_count}/{len(config.SELECTED_CHANNELS)} channels")

            if success_count == len(config.SELECTED_CHANNELS):
                print(f"  🎉 ALL CHANNELS EXTRACTED SUCCESSFULLY!")
                return True
            else:
                print(f"  ⚠️  Some channels failed extraction")
                return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Test mode
        test_success = test_fixed_extraction()
        if test_success:
            print(f"\n🎉 PREPROCESSING SCRIPT FIX VERIFIED!")
            print(f"✅ Ready for production use")
        else:
            print(f"\n❌ Fix verification failed")
            print(f"🔧 Please check the channel extraction logic")
    else:
        # Normal processing mode
        main_single_frame_processing()
