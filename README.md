# 🦠 Epi²Net

<!-- <div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org)

*A physics-informed deep learning framework for spatiotemporal epidemic forecasting using Neural Ordinary Differential Equations (Neural ODEs) and Graph Convolutional Networks (GCNs).*

</div> -->

---

## 🌟 Highlights

- **🧬 Physics-Informed Architecture**: Integrates epidemiological dynamics with deep learning through Neural ODEs
- **🌍 Multi-Region Modeling**: Captures spatial dependencies across regions using Graph Convolutional Networks
- **📊 Real-World Validation**: Tested on COVID-19 data from Italy, England, France, and Spain
- **⚡ High Performance**: Achieves state-of-the-art forecasting accuracy with robust uncertainty quantification
- **🔬 Interpretable**: Combines data-driven learning with mechanistic epidemic models

---

## 📖 Overview

**Epi²Net** is a novel deep learning framework designed for accurate spatiotemporal forecasting of epidemic spread. By combining the flexibility of Neural ODEs with the spatial reasoning capabilities of Graph Convolutional Networks, Epi²Net captures both the temporal dynamics and spatial dependencies inherent in epidemic processes.

<!-- ### Key Features

- **Neural ODE Integration**: Continuous-time modeling of epidemic dynamics
- **Graph-Based Spatial Modeling**: Captures inter-regional transmission patterns
- **Latent Dynamics Learning**: Learns hidden epidemic drivers from data
- **Multi-Horizon Forecasting**: Supports 3-day, 5-day, and 7-day ahead predictions
- **Robust Training**: Adaptive learning rate scheduling and gradient clipping -->

---

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

<!-- ---

## 🔬 Methodology

### Neural ODE Framework

Epi²Net models the latent epidemic dynamics as a continuous-time ODE:

$$\frac{dz(t)}{dt} = f_\theta(z(t), A, t)$$

where:
- $z(t)$ is the latent state vector
- $A$ is the graph adjacency matrix
- $f_\theta$ is a neural network parameterized by $\theta$

### Graph Convolutional Dynamics

The ODE function incorporates spatial structure through diffusion convolution:

$$f_\theta(z, A) = \sigma\left(\sum_{k=0}^K \theta_k (D^{-1}A)^k z\right)$$

This captures multi-hop spatial dependencies between regions.

### Training Objective

The model is trained end-to-end by minimizing the forecasting loss:

$$\mathcal{L} = \sum_{i=1}^N \| y_i - \hat{y}_i \|_2^2 + \lambda \|\theta\|_2^2$$ -->
<!-- 
---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📚 Citation

If you use Epi²Net in your research, please cite:

```bibtex
@article{epi2net2024,
  title={Epi²Net: Physics-Informed Neural ODEs for Spatiotemporal Epidemic Forecasting},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- COVID-19 data provided by national health agencies
- Built with [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- Neural ODE implementation based on [torchdiffeq](https://github.com/rtqichen/torchdiffeq)

---

## 📧 Contact

**Sundiasy** - [@Sundiasy](https://github.com/Sundiasy)

Project Link: [https://github.com/Sundiasy/Epi-2-Net](https://github.com/Sundiasy/Epi-2-Net)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for epidemic research and public health

</div> -->