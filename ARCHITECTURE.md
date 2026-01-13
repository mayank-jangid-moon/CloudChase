# CloudChase Architecture

CloudChase predicts future INSAT cloud frames from a temporal stack of satellite imagery.
The current repository is a Python-only research/training codebase with three main flows:
preprocessing, model training, and checkpoint inference.

## Runtime Flow

1. Raw INSAT HDF5 files are converted by `preprocessing/preprocess.py` into one `.pt`
   file per timestamp. Each file stores a `[channels, height, width]` tensor plus metadata.
2. `unet.py` builds temporal sequences from those frame files with `SingleFrameDataset`.
   Each training sample is split into 8 input frames and 4 future target frames.
3. `SatCastUNet` predicts the future frames directly using a 3D U-Net with spatial skip
   connections, temporal reduction, attention gates, ConvLSTM decoder recurrence, and a
   tanh output range of `[-1, 1]`.
4. `SatCastLoss` combines reconstruction, SSIM, gradient, perceptual, and physics-inspired
   losses with phase-based weights.
5. `inference.py` loads a checkpoint, reconstructs the matching model configuration, runs
   direct UNet prediction, writes predicted frame files, and optionally generates visual
   comparisons and per-channel metrics.

## File Responsibilities

- `config_schema.py`: Pydantic v2 schema for data, model, training, dataloader, hardware,
  monitoring, path, and experimental configuration. It validates values and normalizes paths.
- `config_loader.py`: Hydra loader and variant factory. It converts YAML plus overrides into
  a validated `SatCastConfig` and can serialize config back to YAML.
- `config_modern.yaml`: Default experiment configuration, including dataset directories,
  sequence lengths, loss weights, hardware settings, output paths, and Hydra run directories.
- `preprocessing/preprocess.py`: INSAT L1C HDF5-to-PT converter. It extracts configured
  channels, normalizes them to `[-1, 1]`, resizes to 720x720, stamps frame metadata, and
  writes single-frame PT files.
- `preprocessing/regrid_insat.py`: NetCDF regridding utility for dense variables and sparse
  AMV wind vectors. It can write regridded NetCDF or interleaved data/mask PyTorch tensors.
- `data_format_optimizer.py`: Storage utility for tensor serialization as HDF5, Zarr, or
  PyTorch files, plus format benchmarking and HDF5 memory-mapped dataset creation.
- `unet.py`: Main model, losses, dataset, training loop, metrics, checkpointing, convergence,
  and test-evaluation logic.
- `inference.py`: Checkpoint loading, sequence assembly, prediction, saving, visualization,
  and metric reporting for trained models.
- `requirements.txt`: Python dependency manifest for the repository.
