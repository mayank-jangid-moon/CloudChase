"""
MODERN CONFIGURATION SYSTEM WITH PYDANTIC VALIDATION
Based on Code Analysis Recommendations (Section 2.1)

Features:
- Type validation with Pydantic
- Environment variable support
- Nested configuration structures
- Automatic validation and error reporting
"""

from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass
import os
from pathlib import Path


class DataConfig(BaseModel):
    """Data configuration with validation"""
    data_dir: str = Field(default="/teamspace/studios/this_studio/MOSDAC")
    test_data_dir: str = Field(default="/teamspace/studios/this_studio/TESTING")
    eval_output_dir: str = Field(default="/teamspace/studios/this_studio/EVAL")

    train_months: List[str] = Field(default=['mar', 'apr', 'may'])
    val_months: List[str] = Field(default=['may'])
    test_months: List[str] = Field(default=['jun'])

    image_size: int = Field(default=720, ge=720, le=720)
    sequence_length: int = Field(default=8, ge=8, le=8)
    forecast_length: int = Field(default=4, ge=4, le=4)
    temporal_stride: int = Field(default=6, ge=1, le=24)
    cache_gb: int = Field(default=90, ge=1, le=120)
    preload_all_frames: bool = Field(default=True)

    num_workers: int = Field(default=8, ge=0, le=16)
    pin_memory: bool = Field(default=True)
    persistent_workers: bool = Field(default=True)
    prefetch_factor: int = Field(default=4, ge=1, le=8)

    preferred_format: Literal['hdf5', 'zarr', 'tensor'] = Field(default='hdf5')
    use_memory_mapping: bool = Field(default=True)
    cache_preprocessed: bool = Field(default=True)
    
    @validator('data_dir', 'test_data_dir', 'eval_output_dir')
    def validate_paths(cls, v):
        """Validate that paths exist or can be created"""
        path = Path(v)
        if not path.exists():
            print(f"Warning: Path {v} does not exist")
        return str(path.absolute())


class ModelConfig(BaseModel):
    """Model architecture configuration"""
    in_channels: int = Field(default=5, ge=1, le=10)
    out_channels: int = Field(default=5, ge=1, le=10)
    base_channels: int = Field(default=32, ge=16, le=128)
    channel_multipliers: List[int] = Field(default=[1, 2, 4, 6, 8])
    temporal_depth: int = Field(default=8, ge=4, le=16)
    
    use_attention_gates: bool = Field(default=True)
    use_hierarchical_recurrence: bool = Field(default=True)
    norm_type: Literal['group', 'batch', 'layer'] = Field(default='group')
    activation_type: Literal['gelu', 'relu', 'leaky_relu', 'silu'] = Field(default='gelu')
    upsampling_method: Literal['trilinear_conv', 'conv_transpose', 'bilinear'] = Field(default='trilinear_conv')
    
    use_gradient_checkpointing: bool = Field(default=True)
    use_mixed_precision: bool = Field(default=True)
    compile_model: bool = Field(default=True)


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration"""
    type: Literal['sequential', 'cosine', 'plateau', 'linear'] = Field(default='sequential')
    warmup_epochs: int = Field(default=3, ge=0, le=10)
    cosine_t0: int = Field(default=15, ge=5, le=50)
    plateau_patience: int = Field(default=5, ge=2, le=20)
    plateau_factor: float = Field(default=0.5, ge=0.1, le=0.9)


class LossConfig(BaseModel):
    """Multi-component loss configuration"""
    mse_weight: float = Field(default=1.0, ge=0.0)
    perceptual_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    gradient_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    brightness_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    physics_weight_range: List[float] = Field(default=[0.02, 0.08])
    
    @validator('physics_weight_range')
    def validate_weight_range(cls, v):
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError("physics_weight_range must be [min, max] with min < max")
        return v


class TrainingConfig(BaseModel):
    """Training configuration"""
    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=1, ge=1, le=16)
    accumulation_steps: int = Field(default=4, ge=1, le=32)
    
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-2)
    weight_decay: float = Field(default=0.01, ge=0.0, le=0.1)
    optimizer: Literal['adamw', 'adam', 'sgd'] = Field(default='adamw')
    
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    
    eval_every: int = Field(default=5, ge=1, le=50)
    test_eval_every: int = Field(default=10, ge=1, le=50)
    save_every: int = Field(default=10, ge=1, le=50)
    
    target_psnr: float = Field(default=32.5, ge=20.0, le=50.0)
    target_ssim: float = Field(default=0.89, ge=0.5, le=1.0)
    continue_after_target: bool = Field(default=True)


class DataLoaderConfig(BaseModel):
    """DataLoader optimization configuration"""
    num_workers: int = Field(default=8, ge=0, le=32)
    pin_memory: bool = Field(default=True)
    persistent_workers: bool = Field(default=True)
    prefetch_factor: int = Field(default=4, ge=1, le=16)
    multiprocessing_context: Literal['spawn', 'fork', 'forkserver'] = Field(default='spawn')
    drop_last: bool = Field(default=True)


class HardwareConfig(BaseModel):
    """Hardware optimization configuration"""
    device: str = Field(default='cuda')
    mixed_precision: bool = Field(default=True)
    compile_backend: Literal['inductor', 'aot_eager', 'cudagraphs'] = Field(default='inductor')
    
    memory_fraction: float = Field(default=0.95, ge=0.1, le=1.0)
    empty_cache_every: int = Field(default=50, ge=1, le=1000)
    gradient_clip_norm: float = Field(default=1.0, ge=0.1, le=10.0)
    
    enable_tensor_cores: bool = Field(default=True)
    allow_tf32: bool = Field(default=True)
    cudnn_benchmark: bool = Field(default=True)


class MonitoringConfig(BaseModel):
    """Monitoring and logging configuration"""
    log_every: int = Field(default=10, ge=1, le=100)
    save_samples: bool = Field(default=True)
    track_gradients: bool = Field(default=False)
    profile_memory: bool = Field(default=False)
    
    use_wandb: bool = Field(default=False)
    wandb_project: str = Field(default="satellite_nowcasting")
    wandb_entity: Optional[str] = Field(default=None)


class PathsConfig(BaseModel):
    """Paths configuration with environment variable support"""
    checkpoint_dir: str = Field(default="./checkpoints")
    log_dir: str = Field(default="./logs")
    output_dir: str = Field(default="./output")
    
    @validator('*')
    def create_directories(cls, v):
        """Create directories if they don't exist"""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())


class ExperimentalConfig(BaseModel):
    """Experimental features configuration"""
    use_flash_attention: bool = Field(default=False)
    use_xformers: bool = Field(default=False)
    use_deepspeed: bool = Field(default=False)


@dataclass
class SatCastConfig(BaseModel):
    """
    COMPLETE CONFIGURATION SCHEMA WITH PYDANTIC VALIDATION
    Based on Code Analysis Recommendations
    """
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    experimental: ExperimentalConfig = Field(default_factory=ExperimentalConfig)
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        extra = "forbid"
        use_enum_values = True


def load_config_from_yaml(yaml_path: str) -> SatCastConfig:
    """
    Load configuration from YAML file with Pydantic validation
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        Validated SatCastConfig instance
    """
    import yaml
    
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    return SatCastConfig(**yaml_data)


def get_default_config() -> SatCastConfig:
    """Get default configuration with all validation"""
    return SatCastConfig()


if __name__ == "__main__":
    config = get_default_config()
    print("Default configuration validated successfully")
    print(f"Model channels: {config.model.in_channels} -> {config.model.out_channels}")
    print(f"Training targets: PSNR>={config.training.target_psnr}, SSIM>={config.training.target_ssim}")