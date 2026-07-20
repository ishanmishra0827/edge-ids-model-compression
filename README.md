# Edge IDS Model Compression: Block-Sparse Pruning vs. Inference Latency

This repository provides the official implementation, multi-seed training loops, and native single-board hardware evaluation scripts for replicating the empirical benchmarks reported in **"Block-Sparse Pruning Compresses Models Without Reducing Inference Latency in Intrusion Detection Networks"**.

The core of this study examines the dissociation between storage-level compression and live inference latency when deploying block-sparse models on commodity edge runtimes without sparse-aware execution kernels.

## 🛠️ Hardware & Environment Core
All hardware benchmarks were executed natively on an isolated single-board platform to protect measurement accuracy from shared-host scheduling interference.

* **Target Device:** Raspberry Pi 5 (Broadcom BCM2712 ARM Cortex-A76 @ 2.4 GHz)
* **Operating System:** Raspberry Pi OS (64-bit)
* **Inference Runtime:** TensorFlow Lite (v2.14.0+) utilizing the native XNNPACK CPU Delegate
* **Execution Paradigm:** Deterministic single-sample pipeline (Batch Size = 1) to simulate inline network edge packet routing

## 🗂️ Clean Repository Architecture
The repository is engineered around an explicit three-tier pipeline configuration:
