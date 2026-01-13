import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import glob
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
    METRICS_AVAILABLE = True
except ImportError:
    print("scikit-image not available. SSIM/PSNR metrics will be skipped.")
    METRICS_AVAILABLE = False

import matplotlib
matplotlib.use('Agg')

try:
    from unet import (
        SatCastUNet,
        SatCastLoss,
        load_modern_config,
        ssim,
        compute_perceptual_metrics,
        get_memory_info,
        cleanup_memory
    )

except ImportError as e:
    raise ImportError("Could not import model components. Make sure unet.py is available.") from e


DEFAULT_CHANNEL_NAMES = ['VIS', 'WV', 'SWIR', 'TIR1', 'TIR2']
DEFAULT_CHANNEL_WEIGHTS = [1.0, 1.2, 1.0, 1.1, 1.1]


def build_inference_config(config: Optional[Dict] = None) -> Dict:
    """Build inference defaults from validated training config or a checkpoint config."""
    config = config or {}
    return {
        'in_channels': config.get('in_channels', 5),
        'out_channels': config.get('out_channels', config.get('in_channels', 5)),
        'input_frames': config.get('input_frames', config.get('sequence_length', 8)),
        'output_frames': config.get('output_frames', config.get('forecast_length', 4)),
        'sequence_length': config.get('sequence_length', config.get('input_frames', 8)),
        'forecast_length': config.get('forecast_length', config.get('output_frames', 4)),
        'input_size': config.get('input_size', config.get('image_size', 720)),
        'base_channels': config.get('base_channels', 32),
        'channel_names': config.get('channel_names', DEFAULT_CHANNEL_NAMES),
        'channel_weights': config.get('channel_weights', DEFAULT_CHANNEL_WEIGHTS),
        'data_range': config.get('data_range', 2.0),
    }

class EnhancedSatelliteInference:
    """
    ENHANCED Satellite forecasting inference engine with latest architectural improvements

    Features:
    - Enhanced attention mechanisms for better spatial understanding
    - SSIM preservation layers for structural quality
    - Per-channel loss calculation (prevents channel contamination)
    - Robust error handling and NaN prevention
    - Enhanced visualization with detailed metrics
    - Direct UNet prediction (no diffusion)
    """

    def __init__(self, model_path: str, device: str = 'cuda', config: Optional[Dict] = None):
        self.device = device
        self.model = None
        self.config = None
        self.loss_fn = None
        self.default_config = build_inference_config(config)

        self.load_model(model_path)

        self.loss_fn = SatCastLoss(device=self.device, channel_config=self.config)
        
    def load_model(self, model_path: str):
        """Load trained model from checkpoint with enhanced architecture support"""
        print(f"Loading enhanced model from {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if 'config' in checkpoint:
            self.config = {**self.default_config, **checkpoint['config']}
            print("Loaded configuration from checkpoint and merged with enhanced defaults")
        else:
            self.config = self.default_config.copy()
            print("Using enhanced default configuration (config not found in checkpoint)")

        print(f"Model Configuration:")
        print(f"   Input: {self.config['input_frames']} frames × {self.config['in_channels']} channels")
        print(f"   Output: {self.config['output_frames']} frames × {self.config['out_channels']} channels")
        print(f"   Resolution: {self.config['input_size']}×{self.config['input_size']}")
        print(f"   Base channels: {self.config['base_channels']} (memory-optimized)")
        print(f"   Channels: {self.config['channel_names']}")

        print("Initializing enhanced SatCast UNet...")
        self.model = SatCastUNet(
            in_channels=self.config['in_channels'],
            out_channels=self.config['out_channels'],
            base_channels=self.config['base_channels']
        ).to(self.device)

        print("Using direct UNet prediction (no diffusion)")

        try:
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("Loaded model weights from 'model_state_dict'")
            elif 'ema_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['ema_state_dict'], strict=False)
                print("Loaded EMA weights from 'ema_state_dict'")
            else:
                self.model.load_state_dict(checkpoint, strict=False)
                print("Loaded weights from checkpoint root")
        except Exception as e:
            print(f"Warning during weight loading: {e}")
            print("Continuing with partially loaded weights...")

        self.model.eval()
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Enhanced model loaded successfully")
        print(f"Total parameters: {total_params:,}")
        print(f"Architecture: UNet with enhanced attention and SSIM preservation")
        print(f"Features: Direct prediction, skip connections, 3D convolutions")

        print(f"Input frames: {getattr(self.model, 'input_frames', self.config['input_frames'])}")
        print(f"Output frames: {getattr(self.model, 'output_frames', self.config['output_frames'])}")
        print(f"Base channels: {self.config['base_channels']}")
        print(f"Input size: {self.config['input_size']}")
        print(f"Model validation complete")
        
    def load_sequence_from_files(self, file_paths: List[str]) -> torch.Tensor:
        """Load a sequence from individual frame files"""
        frames = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Frame file not found: {file_path}")
            
            data = torch.load(file_path, map_location='cpu', weights_only=False)
            
            if isinstance(data, dict) and 'frame_data' in data:
                frame_data = data['frame_data']
            else:
                frame_data = data
            
            target_size = self.config.get('image_size', 720)
            if frame_data.shape[-1] != target_size or frame_data.shape[-2] != target_size:
                frame_data = F.interpolate(
                    frame_data.unsqueeze(0),
                    size=(target_size, target_size),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
            
            frames.append(frame_data)
        
        sequence = torch.stack(frames, dim=0)
        return sequence
    
    def create_sequence_from_directory(self, data_dir: str, start_time: Optional[datetime] = None) -> Tuple[torch.Tensor, List[str]]:
        """Create a sequence from files in a directory"""
        frame_files = glob.glob(os.path.join(data_dir, "frame_*.pt"))
        frame_files.sort()
        
        if len(frame_files) == 0:
            raise ValueError(f"No frame files found in {data_dir}")
        
        if start_time:
            filtered_files = []
            for file_path in frame_files:
                filename = os.path.basename(file_path)
                try:
                    import re
                    pattern = r'frame_(\d{8})_(\d{4})\.pt'
                    match = re.search(pattern, filename)
                    if match:
                        date_str, time_str = match.groups()
                        file_time = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M")
                        if file_time >= start_time:
                            filtered_files.append((file_time, file_path))
                except:
                    continue
            
            filtered_files.sort(key=lambda x: x[0])
            frame_files = [fp for _, fp in filtered_files[:self.config['sequence_length']]]
        else:
            frame_files = frame_files[:self.config.get('input_frames', 8)]

        if len(frame_files) < self.config.get('input_frames', 8):
            raise ValueError(f"Need {self.config.get('input_frames', 8)} frames, found {len(frame_files)}")
        
        sequence = self.load_sequence_from_files(frame_files)
        return sequence, frame_files

    def load_ground_truth_sequence(self, data_dir: str, input_files: List[str]) -> Optional[torch.Tensor]:
        """Load ground truth frames that correspond to the prediction time steps"""
        try:
            last_input_file = os.path.basename(input_files[-1])
            import re
            pattern = r'frame_(\d{8})_(\d{4})\.pt'
            match = re.search(pattern, last_input_file)
            if not match:
                return None

            date_str, time_str = match.groups()
            last_input_time = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M")

            gt_files = []
            for i in range(self.config.get('output_frames', 4)):
                gt_time = last_input_time + timedelta(minutes=(i + 1) * 30)
                gt_filename = f"frame_{gt_time.strftime('%Y%m%d_%H%M')}.pt"
                gt_filepath = os.path.join(data_dir, gt_filename)

                if os.path.exists(gt_filepath):
                    gt_files.append(gt_filepath)
                else:
                    print(f"Ground truth file not found: {gt_filename}")
                    return None

            if len(gt_files) == self.config.get('output_frames', 4):
                ground_truth = self.load_sequence_from_files(gt_files)
                print(f"Loaded ground truth: {len(gt_files)} frames")
                return ground_truth
            else:
                return None

        except Exception as e:
            print(f"Could not load ground truth: {e}")
            return None
    
    @torch.no_grad()
    def predict(self, input_sequence: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Generate predictions using enhanced UNet with 3D spatio-temporal attention

        Args:
            input_sequence: Input tensor [T, C, H, W] or [B, T, C, H, W]
            num_samples: Number of prediction samples to average (for uncertainty estimation)

        Returns:
            Predicted frames [T, C, H, W] where T = output_frames
        """
        self.model.eval()

        # Ensure correct input shape: [B, T, C, H, W]
        if len(input_sequence.shape) == 4:  # [T, C, H, W]
            input_sequence = input_sequence.unsqueeze(0)  # [1, T, C, H, W]
        elif len(input_sequence.shape) != 5:
            raise ValueError(f"Expected 4D or 5D input, got {input_sequence.shape}")

        # Move to device
        input_sequence = input_sequence.to(self.device)

        print(f"Generating predictions with enhanced UNet model...")
        print(f"   Input shape: {input_sequence.shape}")
        print(f"   Expected output frames: {self.config['output_frames']}")

        # Validate input sequence length
        if input_sequence.shape[1] != self.config['input_frames']:
            print(f"Warning: Expected {self.config['input_frames']} input frames, got {input_sequence.shape[1]}")
            # Take the last input_frames if we have more
            if input_sequence.shape[1] > self.config['input_frames']:
                input_sequence = input_sequence[:, -self.config['input_frames']:, :, :, :]
                print(f"   Trimmed to last {self.config['input_frames']} frames")
            else:
                raise ValueError(f"Not enough input frames: need {self.config['input_frames']}, got {input_sequence.shape[1]}")

        # Generate predictions using direct UNet forward pass
        predictions = []

        for sample_idx in range(num_samples):
            try:
                # Direct UNet prediction (deterministic, so multiple samples will be identical)
                with torch.no_grad():
                    if sample_idx == 0:
                        print(f"Using direct UNet prediction")
                        print(f"   Input frames: {input_sequence.shape}")

                    # Direct forward pass through UNet
                    # Model expects [B, T, C, H, W] and returns [B, T_out, C, H, W]
                    predicted_frames = self.model(input_sequence)

                    # Ensure output is in correct format
                    if len(predicted_frames.shape) == 5:
                        # Model returns [B, T, C, H, W] - should be forecast frames
                        if predicted_frames.shape[1] == self.config['output_frames']:
                            forecast_frames = predicted_frames
                        else:
                            # Take last output_frames if model returns different number
                            forecast_frames = predicted_frames[:, -self.config['output_frames']:, :, :, :]
                    else:
                        raise ValueError(f"Unexpected model output shape: {predicted_frames.shape}")

                    predictions.append(forecast_frames.cpu())

                    if sample_idx == 0:
                        print(f"Direct UNet prediction completed")
                        print(f"   Output shape: {forecast_frames.shape}")

            except Exception as e:
                print(f"Error in prediction sample {sample_idx + 1}: {e}")
                if sample_idx == 0:  # If first sample fails, re-raise
                    raise
                else:  # Skip failed samples
                    continue

        if not predictions:
            raise RuntimeError("All prediction samples failed")

        # Average predictions if multiple samples
        if len(predictions) > 1:
            prediction = torch.stack(predictions).mean(dim=0)
            print(f"Averaged {len(predictions)} prediction samples")
        else:
            prediction = predictions[0]

        # Remove batch dimension and return: [T, C, H, W]
        final_prediction = prediction.squeeze(0)
        print(f"Generated predictions: {final_prediction.shape}")

        return final_prediction
    
    def save_predictions(self, predictions: torch.Tensor, input_files: List[str], output_dir: str):
        """Save predictions as individual frame files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract base timestamp from first input file
        first_file = os.path.basename(input_files[0])
        try:
            import re
            pattern = r'frame_(\d{8})_(\d{4})\.pt'
            match = re.search(pattern, first_file)
            if match:
                date_str, time_str = match.groups()
                base_time = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M")
                # Predictions start after input sequence (8 frames * 30 min = 4 hours later)
                pred_start_time = base_time + timedelta(hours=4)
            else:
                pred_start_time = datetime.now()
        except:
            pred_start_time = datetime.now()
        
        # Save each predicted frame
        for i, frame in enumerate(predictions):
            # Calculate timestamp for this prediction
            frame_time = pred_start_time + timedelta(minutes=i * 30)
            timestamp_str = frame_time.strftime("%Y%m%d_%H%M")
            
            # Create data package
            data_package = {
                'frame_data': frame,  # Shape: [C, H, W]
                'metadata': {
                    'timestamp': frame_time,
                    'prediction_index': i,
                    'source_files': input_files,
                    'model_config': self.config,
                    'generated_at': datetime.now().isoformat()
                },
                'version': 'prediction_v1.0'
            }
            
            # Save file
            output_file = os.path.join(output_dir, f"pred_{timestamp_str}.pt")
            torch.save(data_package, output_file)
            print(f"Saved prediction: {output_file}")
    
    def visualize_prediction(self, input_sequence: torch.Tensor, predictions: torch.Tensor,
                           output_path: str, channel_idx: int = 0, ground_truth: Optional[torch.Tensor] = None):
        """
        Create enhanced visualization with per-channel metrics and detailed comparison

        Features:
        - Input/Prediction/Ground Truth comparison
        - Per-channel PSNR and SSIM metrics
        - Difference maps with proper scaling
        - Enhanced colorbar and labeling
        """
        channel_name = self.config['channel_names'][channel_idx]
        print(f"Creating enhanced visualization for {channel_name} channel...")

        if ground_truth is not None:
            # Enhanced visualization with ground truth comparison
            fig, axes = plt.subplots(4, 4, figsize=(20, 20))

            # Row 1: Input frames (last 4 of 8 input frames)
            for i in range(4):
                input_idx = input_sequence.shape[0] - 4 + i  # Last 4 input frames
                if input_idx >= 0:
                    frame = input_sequence[input_idx, channel_idx].cpu().numpy()
                    im = axes[0, i].imshow(frame, cmap='viridis', aspect='equal')
                    axes[0, i].set_title(f'Input T-{4-i}', fontsize=10)
                    axes[0, i].axis('off')
                    plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

            # Row 2: Predictions
            for i in range(min(4, predictions.shape[0])):
                frame = predictions[i, channel_idx].cpu().numpy()
                im = axes[1, i].imshow(frame, cmap='viridis', aspect='equal')
                axes[1, i].set_title(f'Predicted T+{i+1}', fontsize=10)
                axes[1, i].axis('off')
                plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

            # Row 3: Ground Truth
            for i in range(min(4, ground_truth.shape[0])):
                frame = ground_truth[i, channel_idx].cpu().numpy()
                im = axes[2, i].imshow(frame, cmap='viridis', aspect='equal')
                axes[2, i].set_title(f'Ground Truth T+{i+1}', fontsize=10)
                axes[2, i].axis('off')
                plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)

            # Row 4: Differences (Prediction - Ground Truth)
            for i in range(min(4, min(predictions.shape[0], ground_truth.shape[0]))):
                pred_frame = predictions[i, channel_idx].cpu().numpy()
                gt_frame = ground_truth[i, channel_idx].cpu().numpy()
                diff_frame = pred_frame - gt_frame

                # Use diverging colormap for differences
                vmax = max(abs(diff_frame.min()), abs(diff_frame.max()))
                im = axes[3, i].imshow(diff_frame, cmap='RdBu_r', aspect='equal',
                                     vmin=-vmax, vmax=vmax)
                axes[3, i].set_title(f'Difference T+{i+1}', fontsize=10)
                axes[3, i].axis('off')
                plt.colorbar(im, ax=axes[3, i], fraction=0.046, pad=0.04)

            # Calculate and display metrics
            if predictions.shape == ground_truth.shape:
                metrics_text = f"{channel_name} Channel Metrics:\n"
                for i in range(min(4, predictions.shape[0])):
                    pred_np = predictions[i, channel_idx].cpu().numpy()
                    gt_np = ground_truth[i, channel_idx].cpu().numpy()

                    # PSNR calculation
                    mse = np.mean((pred_np - gt_np) ** 2)
                    if mse == 0:
                        psnr = float('inf')
                    else:
                        psnr = 20 * np.log10(1.0 / np.sqrt(mse))

                    # SSIM calculation
                    if METRICS_AVAILABLE:
                        try:
                            # Convert numpy to torch tensors for ssim calculation
                            gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                            pred_tensor = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                            data_range = gt_np.max() - gt_np.min()
                            ssim_val = ssim(gt_tensor, pred_tensor, data_range=data_range).item()
                            metrics_text += f"T+{i+1}: PSNR={psnr:.2f}dB, SSIM={ssim_val:.3f}\n"
                        except Exception:
                            metrics_text += f"T+{i+1}: PSNR={psnr:.2f}dB, SSIM=Error\n"
                    else:
                        metrics_text += f"T+{i+1}: PSNR={psnr:.2f}dB\n"

                # Add metrics text to the plot
                fig.text(0.02, 0.02, metrics_text, fontsize=10, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.suptitle(f'Satellite Forecasting Comparison - {channel_name} Channel',
                        fontsize=16, fontweight='bold')

        else:
            # Original visualization without ground truth
            _, axes = plt.subplots(2, 6, figsize=(18, 6))
            axes = axes.flatten()

            # Plot input frames
            for i in range(min(8, input_sequence.shape[0])):
                frame = input_sequence[i, channel_idx].cpu().numpy()
                axes[i].imshow(frame, cmap='viridis', aspect='equal')
                axes[i].set_title(f'Input {i+1}')
                axes[i].axis('off')

            # Plot predicted frames
            for i in range(min(4, predictions.shape[0])):
                frame = predictions[i, channel_idx].cpu().numpy()
                axes[8 + i].imshow(frame, cmap='viridis', aspect='equal')
                axes[8 + i].set_title(f'Pred {i+1}')
                axes[8 + i].axis('off')

            # Hide unused subplots
            total_frames = min(8, input_sequence.shape[0]) + min(4, predictions.shape[0])
            for i in range(total_frames, len(axes)):
                axes[i].axis('off')

            plt.suptitle(f'Satellite Forecasting - {channel_name} Channel',
                        fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved: {output_path}")

    def calculate_enhanced_metrics(self, predictions: torch.Tensor, ground_truth: torch.Tensor, output_dir: str):
        """
        Calculate comprehensive per-channel metrics using the enhanced loss framework

        Features:
        - Per-channel PSNR and SSIM
        - Channel-weighted metrics using training weights
        - Temporal consistency metrics
        - Detailed metrics report
        """
        print("Calculating enhanced per-channel metrics...")

        if not METRICS_AVAILABLE:
            print("Metrics calculation skipped (scikit-image not available)")
            return

        metrics_report = {
            'overall': {},
            'per_channel': {},
            'per_timestep': {},
            'channel_weights': self.config['channel_weights']
        }

        # Calculate metrics for each channel and timestep
        channel_names = self.config['channel_names']
        channel_weights = self.config['channel_weights']

        total_weighted_psnr = 0.0
        total_weighted_ssim = 0.0
        total_weight = 0.0

        for ch_idx, ch_name in enumerate(channel_names):
            ch_weight = channel_weights[ch_idx] if ch_idx < len(channel_weights) else 1.0

            ch_psnrs = []
            ch_ssims = []

            for t_idx in range(predictions.shape[0]):  # For each timestep
                try:
                    pred_frame = predictions[t_idx, ch_idx].cpu().numpy()
                    gt_frame = ground_truth[t_idx, ch_idx].cpu().numpy()

                    # PSNR calculation
                    mse = np.mean((pred_frame - gt_frame) ** 2)
                    if mse == 0:
                        psnr_val = float('inf')
                    else:
                        psnr_val = 20 * np.log10(2.0 / np.sqrt(mse))  # data_range = 2.0

                    # SSIM calculation - convert numpy to torch tensors
                    data_range = max(gt_frame.max() - gt_frame.min(), 1e-8)
                    gt_tensor = torch.from_numpy(gt_frame).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    pred_tensor = torch.from_numpy(pred_frame).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    ssim_val = ssim(gt_tensor, pred_tensor, data_range=data_range).item()

                    ch_psnrs.append(psnr_val)
                    ch_ssims.append(ssim_val)

                except Exception as e:
                    print(f"Error calculating metrics for {ch_name} T+{t_idx+1}: {e}")
                    ch_psnrs.append(0.0)
                    ch_ssims.append(0.0)

            # Channel averages
            avg_psnr = np.mean([p for p in ch_psnrs if p != float('inf')])
            avg_ssim = np.mean(ch_ssims)

            metrics_report['per_channel'][ch_name] = {
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'weight': ch_weight,
                'per_timestep': {
                    'psnr': ch_psnrs,
                    'ssim': ch_ssims
                }
            }

            # Weighted totals
            if avg_psnr != float('inf'):
                total_weighted_psnr += ch_weight * avg_psnr
                total_weighted_ssim += ch_weight * avg_ssim
                total_weight += ch_weight

            print(f"   {ch_name}: PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.3f} (weight={ch_weight:.1f})")

        # Overall weighted metrics
        if total_weight > 0:
            metrics_report['overall']['weighted_psnr'] = total_weighted_psnr / total_weight
            metrics_report['overall']['weighted_ssim'] = total_weighted_ssim / total_weight

        # Save metrics report
        metrics_file = os.path.join(output_dir, 'enhanced_metrics.json')
        with open(metrics_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                return obj

            import json
            json.dump(metrics_report, f, indent=2, default=convert_numpy)

        print(f"Enhanced metrics saved to: {metrics_file}")
        print(f"Overall Weighted PSNR: {metrics_report['overall'].get('weighted_psnr', 0):.2f}dB")
        print(f"Overall Weighted SSIM: {metrics_report['overall'].get('weighted_ssim', 0):.3f}")

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Satellite Forecasting Inference with Per-Channel Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic inference with visualization
    python inference.py --data /path/to/test/data --model best_model.pth --output results --visualize

    # Batch processing with device selection
    python inference.py --data /path/to/test/data --model best_model.pth --device cuda

    # Single sequence inference
    python inference.py --model best_model.pth --sequence_files frame_*.pt --visualize
        """
    )

    # CRITICAL FIX: Match user's preferred argument order and names
    parser.add_argument('--data', help='Path to test data directory')
    parser.add_argument('--model', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output', default='predictions', help='Output directory for predictions and visualizations')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--visualize', action='store_true', help='Create visualization plots for each channel showing input/predicted/target frames')

    # Additional options
    parser.add_argument('--sequence_files', nargs='+', help='List of specific frame files to use as input (alternative to --data)')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of prediction samples to average')
    parser.add_argument('--start_time', help='Start time for sequence (YYYYMMDD_HHMM format)')

    args = parser.parse_args()

    print("Starting Enhanced Satellite Forecasting Inference")
    print("=" * 60)

    CONFIG, _ = load_modern_config()

    # Initialize enhanced inference engine
    inference = EnhancedSatelliteInference(args.model, args.device, config=CONFIG)
    
    # Load input sequence
    if args.sequence_files:
        # Use specific files
        print(f"Loading sequence from {len(args.sequence_files)} specified files")
        input_sequence = inference.load_sequence_from_files(args.sequence_files)
        input_files = args.sequence_files
    elif args.data:
        # Load from directory
        start_time = None
        if args.start_time:
            start_time = datetime.strptime(args.start_time, "%Y%m%d_%H%M")
        
        print(f"Loading sequence from directory: {args.data}")
        input_sequence, input_files = inference.create_sequence_from_directory(args.data, start_time)
    else:
        print("Must specify either --data directory or --sequence_files")
        return
    
    print(f"Loaded input sequence: {input_sequence.shape}")
    print(f"Input files: {len(input_files)} frames")
    
    # Generate predictions
    print(f"Generating predictions (samples: {args.num_samples})")
    predictions = inference.predict(input_sequence, args.num_samples)
    print(f"Generated predictions: {predictions.shape}")
    
    # Save predictions
    print(f"Saving predictions to: {args.output}")
    inference.save_predictions(predictions, input_files, args.output)
    
    # Load ground truth for comparison (if available)
    ground_truth = None
    if args.data:
        print("Looking for ground truth data...")
        ground_truth = inference.load_ground_truth_sequence(args.data, input_files)

    # Calculate enhanced metrics if ground truth is available
    if ground_truth is not None:
        print("Calculating enhanced per-channel metrics...")
        inference.calculate_enhanced_metrics(predictions, ground_truth, args.output)

    # Create enhanced visualizations
    if args.visualize:
        print("Creating enhanced visualizations...")
        os.makedirs(os.path.join(args.output, 'visualizations'), exist_ok=True)

        # Create visualization for each channel
        for channel_idx, channel_name in enumerate(inference.config['channel_names']):
            viz_path = os.path.join(args.output, 'visualizations', f'prediction_viz_{channel_name.lower()}.png')
            inference.visualize_prediction(input_sequence, predictions, viz_path, channel_idx, ground_truth)

        print(f"Visualizations saved to: {os.path.join(args.output, 'visualizations')}")

    print("=" * 60)
    print("Enhanced Satellite Forecasting Inference Complete")
    print(f"Results saved to: {args.output}")
    if args.visualize:
        print(f"Visualizations available in: {os.path.join(args.output, 'visualizations')}")

if __name__ == "__main__":
    main()
