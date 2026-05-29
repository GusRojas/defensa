# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Thesis project comparing deep learning models for autonomous drone navigation ("Defensa"). Trains and evaluates 10+ CNN/RNN models on a drone control task, exports them to ONNX, and visualizes training metrics.

## Commands

```bash
# Create virtual environment & install dependencies
uv sync

# Or using pip
pip install -r requirements.txt

# Run model comparison script (generates all plots from training histories)
python scripts/compare_models_final.py --models-dir ./data/mis_modelos --output-dir ./comparison_results

# Run comparison with custom pattern
python scripts/compare_models_final.py --models-dir ./data/mis_modelos --pattern "training_history_DroneNav*.json"

# Run Jupyter notebook
jupyter notebook notebooks/

# Activate virtual environment
source .venv/bin/activate
```

## Project Structure

```
defensa/
├── scripts/
│   └── compare_models_final.py   # Main analysis: loads training_history_*.json,
│                                  # generates comparison plots (loss, metrics,
│                                  # overfitting, convergence, summary tables)
├── data/
│   └── mis_modelos/              # Training history JSON files for 10 models
├── notebooks/
│   └── 001_comapre_models.ipynb  # Exploratory notebook (WIP)
├── main.py                       # Placeholder entry point
├── pyproject.toml                # UV project config (Python 3.13)
├── requirements.txt              # pip-compatible dependency list
└── uv.lock                       # Locked dependency versions
```

## Models (10 architectures compared)

| Model | Type |
|---|---|
| PilotNetRegressor | CNN (NVIDIA PilotNet) |
| MobileNetV3_large | Lightweight CNN |
| MobileNetV3_small | Lightweight CNN (smaller) |
| DroneResNet18 | ResNet-based CNN |
| MLP | Fully connected |
| ConvMLP | CNN + MLP hybrid |
| ConvLSTM | CNN + LSTM |
| DroneNav-ConvLSTM | Drone-specific ConvLSTM variant |
| DroneNavSA-ConvLSTM | ConvLSTM with self-attention |
| DroneNavSA-ConvLSTM_completo | Full self-attention ConvLSTM |

## Training History JSON Format

Each file in `data/mis_modelos/` follows `training_history_<ModelName>.json` with:

- **Metadata**: `batch_size`, `learning_rate`, `weight_decay`, `epochs_trained`, `best_val_loss`, `total_time_hours`
- **Training curves**: `train_loss`, `val_loss`, `train_mae`, `val_mae`, `train_mse`, `val_mse`, `train_accuracy`, `val_accuracy` (each a list of per-epoch values)

## Key Dependencies

- **PyTorch** / **TorchVision** — model definitions and training
- **ONNX** / **ONNX Runtime** — model export and deployment
- **pymavlink** — drone communication protocol (MAVLink), for simulation
- **matplotlib** / **seaborn** — academic publication-quality plots (300 DPI)
- **scikit-learn** — ML evaluation metrics
- **OpenCV** / **Pillow** — image processing for drone camera input
- **h5py** — HDF5 dataset storage

## Academic Plot Style

The `compare_models_final.py` script uses a consistent color palette across all models and outputs at 300 DPI. Default `ylim(0.15, 0.3)` for validation loss plots. Plots are saved to a configurable output directory.
