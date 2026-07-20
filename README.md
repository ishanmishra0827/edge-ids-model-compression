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

```text
├── data/
│   └── preprocess.py            # Feature engineering, encoding alignment, and safe-k SMOTE balancing
├── models/
│   ├── train_seeds.py           # Automated 5-seed training loop with TF-MOT pruning callbacks
│   └── verify_sparsity.py       # Validates structural mask stripping and weight matrix zero-counts
└── benchmarks/
    ├── summarize_results.py     # Aggregates 5-seed telemetry files into scientific Mean ± SD tables
    └── evaluate_and_plot.py     # Executes TFLite hardware stream, plots 23-class CM, and macro ROC curves
```

## 🚀 Execution & Reproduction Workflow

### 1. Environment Initialization
Clone the architecture repository and freeze package dependencies inside a clean virtual workspace:
```bash
git clone https://github.com
cd edge-ids-model-compression
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Standardized Feature Engineering
Execute the dataset ingestion module. This pipeline downloads canonical splits, isolates training-unseen anomalies as an evaluation check, applies aligned One-Hot Encoding, performs training-bounded Min-Max scaling, and applies automated local oversampling:
```bash
python data/preprocess.py
```

### 3. Statistically Validated Multiseed Experiment
Execute the multi-seed optimization matrix. The script systematically loops over five distinct random seeds (`[42, 123, 456, 789, 2024]`), handles weight distribution configurations, strips training wrappers, and compiles compressed deployment models (`.tflite` via `Optimize.DEFAULT` optimization parameters):
```bash
python models/train_seeds.py
```

### 4. Telemetry Extraction & Post-Processing
To evaluate and visualize metrics across your experiment configurations, pass any generated model artifact to the unified evaluation and plotting runner:
```bash
# Evaluate and plot charts for the dense seed 42 run
python benchmarks/evaluate_and_plot.py --model models/MCDNN_dense_seed42.tflite

# Evaluate and plot charts for the pruned seed 42 run
python benchmarks/evaluate_and_plot.py --model models/MCDNN_pruned_seed42.tflite
```

### 5. Aggregate Macro Performance
Once all five random training seeds have finalized their deployment cycles, parse the comprehensive results payload (`multiseed_results.json`) to dynamically print clean LaTeX tables embedded with structural standard deviation measurements (\(\pm\)):
```bash
python benchmarks/summarize_results.py
```
