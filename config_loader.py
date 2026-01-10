import os
from pathlib import Path
from typing import Optional, Dict, Any
import hydra
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import logging

from config_schema import SatCastConfig, DataConfig, ModelConfig, TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigManager:
    """
    MODERN CONFIGURATION MANAGER
    Combines Hydra's flexibility with Pydantic's validation
    """
    
    def __init__(self):
        self.config_store = ConfigStore.instance()
        self._register_configs()
    
    def _register_configs(self):
        """Register Pydantic schemas with Hydra ConfigStore"""
        self.config_store.store(name="base_config", node=SatCastConfig)
        
        self.config_store.store(group="data", name="base", node=DataConfig)
        self.config_store.store(group="model", name="base", node=ModelConfig)
        self.config_store.store(group="training", name="base", node=TrainingConfig)
    
    def load_config(self, 
                   config_path: Optional[str] = None,
                   config_name: str = "config_modern",
                   overrides: Optional[list] = None) -> SatCastConfig:
        """
        Load configuration with Hydra + Pydantic validation
        
        Args:
            config_path: Path to config directory (None for current dir)
            config_name: Name of config file (without .yaml)
            overrides: List of config overrides (e.g., ["model.base_channels=64"])
            
        Returns:
            Validated SatCastConfig instance
        """
        try:
            if config_path is None:
                config_path = str(Path(__file__).parent)
            
            with initialize(config_path=config_path, version_base=None):
                cfg = compose(config_name=config_name, overrides=overrides or [])
                
                cfg_dict = OmegaConf.to_container(cfg, resolve=True)
                
                validated_config = SatCastConfig(**cfg_dict)
                
                logger.info("Configuration loaded and validated successfully")
                return validated_config
                
        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            logger.info("Falling back to default configuration...")
            return SatCastConfig()
    
    def save_config(self, config: SatCastConfig, output_path: str):
        """Save configuration to YAML file"""
        config_dict = config.dict()
        
        omega_cfg = OmegaConf.create(config_dict)
        
        with open(output_path, 'w') as f:
            OmegaConf.save(omega_cfg, f)
        
        logger.info(f"Configuration saved to {output_path}")
    
    def print_config_summary(self, config: SatCastConfig):
        """Print a summary of the configuration"""
        print("\n" + "="*60)
        print("CONFIGURATION SUMMARY")
        print("="*60)
        
        print(f"Data:")
        print(f"   Training months: {config.data.train_months}")
        print(f"   Validation months: {config.data.val_months}")
        print(f"   Test months: {config.data.test_months}")
        print(f"   Image size: {config.data.image_size}x{config.data.image_size}")
        print(f"   Sequence length: {config.data.sequence_length} frames")
        print(f"   Forecast length: {config.data.forecast_length} frames")
        print(f"   Temporal stride: {config.data.temporal_stride}")
        print(f"   Cache size: {config.data.cache_gb}GB")
        print(f"   Preferred format: {config.data.preferred_format}")
        
        print(f"Model:")
        print(f"   Channels: {config.model.in_channels} -> {config.model.out_channels}")
        print(f"   Base channels: {config.model.base_channels}")
        print(f"   Attention gates: {config.model.use_attention_gates}")
        print(f"   Hierarchical recurrence: {config.model.use_hierarchical_recurrence}")
        print(f"   Normalization: {config.model.norm_type}")
        print(f"   Activation: {config.model.activation_type}")
        
        print(f"Training:")
        print(f"   Epochs: {config.training.epochs}")
        print(f"   Batch size: {config.training.batch_size}")
        print(f"   Learning rate: {config.training.learning_rate}")
        print(f"   Optimizer: {config.training.optimizer}")
        print(f"   Target PSNR: {config.training.target_psnr}")
        print(f"   Target SSIM: {config.training.target_ssim}")
        
        print(f"Loss weights:")
        print(f"   MSE: {config.training.loss.mse_weight}")
        print(f"   Perceptual: {config.training.loss.perceptual_weight}")
        print(f"   Gradient: {config.training.loss.gradient_weight}")
        print(f"   Brightness: {config.training.loss.brightness_weight}")
        
        print(f"Hardware:")
        print(f"   Device: {config.hardware.device}")
        print(f"   Mixed precision: {config.hardware.mixed_precision}")
        print(f"   Tensor cores: {config.hardware.enable_tensor_cores}")
        print(f"   Compile model: {config.model.compile_model}")
        
        print("="*60 + "\n")


def create_config_variants():
    """Create configuration variants for different scenarios"""
    
    high_perf_overrides = [
        "model.base_channels=64",
        "training.batch_size=2",
        "training.learning_rate=2e-4",
        "dataloader.num_workers=16",
        "hardware.memory_fraction=0.98"
    ]
    
    memory_efficient_overrides = [
        "model.base_channels=24",
        "training.batch_size=1",
        "training.accumulation_steps=8",
        "model.use_gradient_checkpointing=true",
        "hardware.memory_fraction=0.85"
    ]
    
    fast_training_overrides = [
        "training.epochs=50",
        "training.eval_every=10",
        "training.target_psnr=30.0",
        "training.target_ssim=0.85",
        "model.temporal_depth=6"
    ]
    
    return {
        "high_performance": high_perf_overrides,
        "memory_efficient": memory_efficient_overrides,
        "fast_training": fast_training_overrides
    }


# Global config manager instance
config_manager = ConfigManager()


def load_config_with_overrides(overrides: Optional[list] = None) -> SatCastConfig:
    """Convenience function to load config with overrides"""
    return config_manager.load_config(overrides=overrides)


def load_config_variant(variant: str) -> SatCastConfig:
    """Load a predefined configuration variant"""
    variants = create_config_variants()
    
    if variant not in variants:
        logger.warning(f"Unknown variant '{variant}'. Available: {list(variants.keys())}")
        return SatCastConfig()
    
    overrides = variants[variant]
    logger.info(f"Loading '{variant}' configuration variant...")
    
    return config_manager.load_config(overrides=overrides)


@hydra.main(version_base=None, config_path=".", config_name="config_modern")
def main(cfg: DictConfig) -> None:
    """Example Hydra app using the configuration"""
    
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    validated_config = SatCastConfig(**config_dict)
    
    config_manager.print_config_summary(validated_config)
    
    print(f"Starting training with {validated_config.model.base_channels} base channels")
    print(f"Using {validated_config.data.preferred_format} data format")
    print(f"Target metrics: PSNR>={validated_config.training.target_psnr}, SSIM>={validated_config.training.target_ssim}")


if __name__ == "__main__":
    print("Testing configuration system...")
    
    default_config = SatCastConfig()
    config_manager.print_config_summary(default_config)
    
    for variant_name in ["high_performance", "memory_efficient", "fast_training"]:
        print(f"\nTesting '{variant_name}' variant...")
        variant_config = load_config_variant(variant_name)
        print(f"   Base channels: {variant_config.model.base_channels}")
        print(f"   Batch size: {variant_config.training.batch_size}")
        print(f"   Learning rate: {variant_config.training.learning_rate}")
    
    print("\nConfiguration system test completed successfully")
