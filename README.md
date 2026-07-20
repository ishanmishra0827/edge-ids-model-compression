# Edge IDS Model Compression: Block-Sparse Pruning vs. Inference Latency

Official implementation, multi-seed training loop, and native Raspberry Pi
hardware evaluation scripts for the empirical benchmarks reported in
**"Block-Sparse Pruning Compresses Models Without Reducing Inference
Latency in Intrusion Detection Networks."**

This study examines the dissociation between storage-level compression
and live inference latency when deploying block-sparse pruned models on
commodity edge runtimes without sparse-aware execution kernels.

## Hardware & Environment

* **Target Device:** Raspberry Pi 5 (Broadcom BCM2712 ARM Cortex-A76 @ 2.4 GHz)
* **Operating System:** Raspberry Pi OS (64-bit), kernel 6.18.34+rpt-rpi-2712
* **Inference Runtime:** TensorFlow Lite, XNNPACK CPU delegate
* **Execution Paradigm:** Single-sample pipeline (batch size = 1)

## Repository Contents

| File | Purpose |
|---|---|
| `Data_preprocessing.py` | Downloads NSL-KDD train/test partitions, aligns one-hot encoding, applies training-bounded Min-Max scaling, isolates unseen-attack-type test records, applies safe-k SMOTE balancing |
| `multi_seed_experiment.py` | Trains both dense and pruned models across 5 random seeds, exports each to TFLite, evaluates on the real test set, benchmarks latency |
| `verify_sparsity.py` | Confirms actual achieved weight sparsity from a saved pruned model |
| `summarize_multiseed.py` | Aggregates results across all completed seed runs into mean ± std, with ready-to-use LaTeX table output |
| `Model_Evaluation_Final.py` | Generates the full 23-class confusion matrix and classification report used in the paper |

## Reproduction Workflow

### 1. Environment setup
```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Data preprocessing
```bash
python Data_preprocessing.py
```
Produces `x_train.npy`, `y_train.npy`, `x_val.npy`, `y_val.npy`,
`x_test_real.npy`, `y_test_real.npy`, and `unseen_class_report.csv`.

### 3. Multi-seed training
```bash
python multi_seed_experiment.py
```
Trains 10 total models (5 seeds x dense/pruned). Resumable — safe to
interrupt and rerun; already-completed runs are skipped. Saves
incrementally to `multiseed_results.json`.

### 4. Aggregate results
```bash
python summarize_multiseed.py
```
Prints mean ± std across completed seeds and LaTeX-ready table rows.

### 5. Confusion matrix and classification report
```bash
python Model_Evaluation_Final.py
```
