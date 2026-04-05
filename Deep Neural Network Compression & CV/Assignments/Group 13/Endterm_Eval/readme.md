# Deep Compression: 3-Stage CNN Optimization for Edge Deployment

This repository implements the full **Deep Compression** pipeline—Pruning, Trained Quantization, and Huffman Coding—as described in the foundational research by Han et al. (2015). The goal is to shrink a VGG-style CNN (`SmallCIFARNet`) so it can fit entirely within on-chip SRAM, bypassing the energy-expensive "memory wall" of off-chip DRAM.

## 🚀 Key Features
* **Stage 1: Global L1 Pruning:** Achieves 90% sparsity by removing low-magnitude weights globally across the network.
* **Stage 2: Trained Quantization:** Implements K-Means clustering ($k=32$) with vectorized gradient aggregation for centroid fine-tuning.
* **Stage 3: Huffman Coding:** Lossless entropy encoding of quantized weights and relative sparse indices.
* **Hardware Optimized:** Native support for Apple Silicon (M1/M2/M3) via the PyTorch MPS backend.

---

## 🛠️ Installation & Setup

### 1. Clone and Environment
```bash
git clone [https://github.com/your-username/deep-compression-cifar10.git](https://github.com/your-username/deep-compression-cifar10.git)
cd deep-compression-cifar10

# Activate your virtual environment
source .venv/bin/activate 

# Install dependencies
pip install torch torchvision numpy scikit-learn
