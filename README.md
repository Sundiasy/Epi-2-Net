# 🦠 Epi²Net

## 🌟 Highlights

- **🧬 Physics-Informed Architecture**: Integrates epidemiological dynamics with deep learning through Neural ODEs
- **🌍 Multi-Region Modeling**: Captures spatial dependencies across regions using Graph Convolutional Networks
- **📊 Real-World Validation**: Tested on COVID-19 data from Italy, England, France, and Spain
- **⚡ High Performance**: Achieves state-of-the-art forecasting accuracy with robust uncertainty quantification
- **🔬 Interpretable**: Combines data-driven learning with mechanistic epidemic models

---

## 📖 Overview

**Epi²Net** is a novel deep learning framework designed for accurate spatiotemporal forecasting of epidemic spread. By combining the flexibility of Neural ODEs with the spatial reasoning capabilities of Graph Convolutional Networks, Epi²Net captures both the temporal dynamics and spatial dependencies inherent in epidemic processes.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Time Series                         │
│               (Cases, Locations, Mobility, etc.)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              RNN Encoder (GRU-based)                         │
│          Extracts initial latent state z₀                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Neural ODE + GCN Dynamics                          │
│    dz/dt = f(z, A, θ) - Graph-aware ODE function            │
│    • Diffusion Convolution for spatial modeling             │
│    • Adaptive graph learning                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              ODE Solver (Dopri5/Euler)                       │
│         Integrates latent dynamics forward in time           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Output Decoder                              │
│           Multi-horizon epidemic forecasts                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Installation

**Install dependencies**
```bash
pip install -r requirements.txt
```

## 📊 Dataset Structure

The project supports epidemic data from multiple countries:

```
dataset/
├── Italy/
│   ├── italy_labels.csv          # Time series data
│   ├── italy_regions.csv         # Region metadata
│   ├── graphs/                   # Daily adjacency matrices
│   │   ├── IT_2020-03-13.csv
│   │   └── ...
│   └── gamma.npy                 # Recovery rate parameter
├── England/
│   └── ...
```

---

## 🎯 Training

### Quick Start

Train on Italy dataset with default configuration:

```bash
python main.py --root_path ./dataset/Italy/ \
               --data_path italy_labels.csv \
               --node_num 105
```

### Advanced Options

```bash
python main.py --root_path ./dataset/England/ \
               --data_path england_labels.csv \
               --node_num 129 
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--root_path` | Root directory of dataset | `./dataset/Italy/` |
| `--data_path` | CSV file with time series | `italy_labels.csv` |
| `--node_num` | Number of regions | `105` |

---

## 📈 Evaluation

The model is evaluated on three forecasting horizons:

- **3-day ahead**: Short-term predictions
- **5-day ahead**: Medium-term predictions
- **7-day ahead**: Long-term predictions

### Metrics

- **MAE** (Mean Absolute Error): Average prediction error
- **MAPE** (Mean Absolute Percentage Error): Relative error
- **RMSE** (Root Mean Square Error): Penalizes large errors

Results are saved in:
- `logs/Epi2Net_results.txt` - Performance summary
- `models/exp_*/` - Saved model checkpoints
- `logs/exp_*/` - Training logs

---

## 📁 Project Structure

```
Epi-2-Net/
├── config.yaml              # Configuration file
├── main.py                  # Main training script
├── epi2net_model.py         # Model architecture
├── epi2net_supervisor.py    # Training/evaluation logic
├── ode_func.py              # Neural ODE function
├── diffeq_solver.py         # ODE solver wrapper
├── utills.py                # Utility functions
├── metrics.py               # Evaluation metrics
├── data_provider/           # Data loading modules
│   ├── data_factory.py      # Data factory
│   ├── data_loader.py       # Dataset loaders
│   └── dataloader.py        # PyTorch DataLoader
├── dataset/                 # Epidemic datasets
├── utils/                   # Additional utilities
│   ├── losses.py            # Loss functions
│   ├── metrics.py           # Metric calculations
│   └── tools.py             # Helper tools
├── logs/                    # Training logs
├── models/                  # Saved models
└── README.md                # This file
```
