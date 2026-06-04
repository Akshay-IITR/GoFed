# GoFed — Federated Abuse Classification

> **GoFed: Towards Energy-Accuracy Tradeoff in Abuse Classification Based on Optimum Parameter Selection**

---

## Table of Contents

1. [Overview]                      (#1-overview)
2. [Architecture]                  (#2-architecture)
3. [Repository Structure]          (#3-repository-structure)
4. [Requirements]                  (#4-requirements)
5. [Installation]                  (#5-installation)
6. [Running GoFed]                 (#6-running-gofed)
7. [CLI Reference]                 (#7-cli-reference)
8. [Output Files]                  (#8-output-files)
9. [Reproducing Paper Results]     (#9-reproducing-paper-results)
10. [Energy Model]                 (#10-energy-model)
11. [Troubleshooting]              (#11-troubleshooting)

---

## 1. Overview

GoFed is a **personalized, communication-efficient Federated Learning (FL) framework** for abuse classification. It addresses two core challenges in federated NLP:

- **Data heterogeneity** — clients hold non-IID datasets in diverse languages and platforms
- **Communication overhead** — resource-constrained devices cannot afford to upload model updates every round

**Key idea — Local Classifier Accuracy (LCA) gating:**
After each local training round, a client evaluates its model on its own test set. If the resulting LCA exceeds the client's personal best, the updated BiLSTM parameters are transmitted to the server. Otherwise, nothing is sent and the server reuses the stored copy of that client's best-performing parameters for aggregation.

**Results reported in the paper:**
- Up to **10.41 %** improvement in local classifier accuracy over baselines
- Up to **55 %** reduction in communication overhead
- Up to **73 %** reduction in communication energy

---

## 2. Architecture

```
Raw Text
  │
  ▼
DistilBERT (frozen)                    ← embeddings only; NOT transmitted in FL
  │   768-dim contextual embeddings
  ▼
BiLSTM Layer 1  (128 hidden, bidirectional → 256-dim output)
  │
Dropout (rate = f_dropout)
  │
BiLSTM Layer 2  (64 hidden, bidirectional → 128-dim output)
  │
Dropout (rate = s_dropout)
  │
Dense → Sigmoid                        ← binary: Abusive (1) / Non-Abusive (0)
```

> Only **BiLSTM + Dense** weights are exchanged in federated rounds.
> DistilBERT stays frozen on every client and is never transmitted.

---

## 3. Repository Structure

```
GoFed/
├── gofed.py                  # Main implementation — GoFed algorithm + model
├── requirements.txt          # Python package dependencies
├── README.md                 # This file
│
├── Clients/                  # ← create this with your datasets
│   ├── client_1/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── client_2/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── ...
│   ├── client_10/
│   │   ├── train.csv
│   │   └── test.csv
│   └── centralized_test.csv  # global evaluation set
│
└── results/                  # ← created automatically on first run
    ├── gofed_metrics.csv
    ├── gofed_client_details.csv
    └── run_config.json
```

### CSV format

Every `train.csv`, `test.csv`, and `centralized_test.csv` must contain exactly two columns:

| Column | Type | Description |
|--------|------|-------------|
| `Text` | string | Raw social media text |
| `Label` | int or string | Binary class: `0`/`1` or any two string values |

> String labels (e.g. `CAG`/`NAG`, `HOF`/`NOT`) are automatically remapped to `0`/`1` alphabetically.
> To ensure the correct mapping, pre-convert labels to integers before running.

---

## 4. Requirements

| Package | Minimum Version | Purpose |
|---------|:--------------:|---------|
| Python | 3.10 | Runtime |
| torch | 2.0.0 | Model training and tensor operations |
| transformers | 4.36.0 | DistilBERT backbone and tokenizer |
| tokenizers | 0.15.0 | Fast tokenization backend |
| numpy | 1.24.0 | Numerical operations |
| pandas | 1.5.0 | CSV data loading |
| scikit-learn | 1.2.0 | Accuracy and F1-score metrics |

---

## 5. Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/gofed.git
cd gofed
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Using conda (recommended)
conda create -n gofed python=3.11 -y
conda activate gofed

# Or using venv
python -m venv gofed_env
source gofed_env/bin/activate         # Linux / macOS
gofed_env\Scripts\activate            # Windows
```

### Step 3 — Install dependencies

**CPU only:**
```bash
pip install -r requirements.txt
```

**GPU — CUDA 11.8 (Ampere and older, sm_50 to sm_90):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**GPU — CUDA 12.1 (Ampere / Ada Lovelace):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**GPU — CUDA 12.8 Nightly (Blackwell sm_120 — RTX PRO 3000 / RTX 50-series):**
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
```

### Step 4 — Verify installation

```bash
python -c "
import torch, transformers, sklearn
print('PyTorch     :', torch.__version__)
print('Transformers:', transformers.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU         :', torch.cuda.get_device_name(0))
"
```

## 6. Running GoFed

### Similar-task setting

All clients share the same hyperparameters (Section 5.3.2, similar-task scenario in the paper).

```bash
python gofed.py \
    --data_dir              Clients             \
    --output_dir            results/similar     \
    --task_setting          similar             \
    --num_rounds            20                  \
    --num_local_epochs      1                   \
    --participation_ratio   1.0                 \
    --lr                    0.001               \
    --batch_size            10                  \
    --optimizer             RMSprop             \
    --f_dropout             0.1                 \
    --s_dropout             0.3                 \
    --device                cuda
```

### Multi-task setting

Each client uses its own hyperparameters from Table 2 of the paper.
These are loaded automatically — no extra flags are needed.

```bash
python gofed.py \
    --data_dir              Clients             \
    --output_dir            results/multitask   \
    --task_setting          multitask           \
    --num_rounds            20                  \
    --num_local_epochs      1                   \
    --participation_ratio   1.0                 \
    --device                cuda
```

### Partial participation (κ = 0.6, 10 clients)

Reproduces the FedL2 baseline comparison from the paper.

```bash
python gofed.py \
    --data_dir              Clients             \
    --output_dir            results/partial     \
    --task_setting          similar             \
    --num_rounds            20                  \
    --participation_ratio   0.6                 \
    --num_clients           10                  \
    --device                cuda
```

### CPU-only run (for testing without a GPU)

```bash
python gofed.py \
    --data_dir     Clients         \
    --output_dir   results/cpu     \
    --task_setting similar         \
    --num_rounds   5               \
    --batch_size   8               \
    --device       cpu
```

---

## 7. CLI Reference

### Paths

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `Clients` | Root directory containing `client_1/` … `client_N/` subdirs and `centralized_test.csv` |
| `--output_dir` | `gofed_results` | Directory where metrics CSVs and config JSON are saved |
| `--bert_model` | `distilbert-base-multilingual-cased` | HuggingFace model name or local path for BERT backbone |

### Federated settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_clients` | `10` | Total number of federated clients |
| `--num_rounds` | `20` | Number of global FL rounds (T) |
| `--num_local_epochs` | `1` | Local training epochs per round (Eₗ) |
| `--participation_ratio` | `1.0` | Fraction of clients selected per round (κ) |
| `--task_setting` | `similar` | `similar` — shared HP; `multitask` — per-client HP from Table 2 |

### Hyperparameters — similar-task only

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | `0.001` | Learning rate |
| `--batch_size` | `10` | Batch size |
| `--optimizer` | `RMSprop` | Optimizer: `Adam` / `Adamax` / `Nadam` / `RMSprop` / `Adagrad` |
| `--f_dropout` | `0.1` | Dropout after first BiLSTM layer |
| `--s_dropout` | `0.3` | Dropout after second BiLSTM layer |

### Model architecture

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_length` | `256` | Tokenizer maximum sequence length (tokens) |
| `--hidden_size` | `128` | First BiLSTM hidden size; second layer uses `hidden_size // 2` |

### Energy model

| Argument | Default | Description |
|----------|---------|-------------|
| `--cpu_freq_ghz` | `2.2` | Client CPU base clock frequency (GHz) |
| `--gpu_freq_ghz` | `1.395` | Client GPU base clock frequency (GHz) |
| `--phi_cpu` | `2.3e-29` | CPU energy coefficient φ (W/(cycles/s)³) |
| `--phi_gpu` | `2.3e-29` | GPU energy coefficient φ′ per SM (W/(cycles/s)³) |
| `--n_sm` | `84` | Number of GPU streaming multiprocessors |
| `--bandwidth_hz` | `1e6` | Uplink wireless channel bandwidth (Hz) |
| `--tx_power_w` | `0.5` | Client uplink transmit power (W) |
| `--channel_gain` | `1.0` | Wireless channel gain hᵢ |
| `--noise_psd` | `0.025` | Noise power spectral density N₀ (W/Hz); calibrated so E_cm ≈ 4 J per upload, matching Figure 5b |

### Miscellaneous

| Argument | Default | Description |
|----------|---------|-------------|
| `--seed` | `42` | Global random seed for reproducibility |
| `--device` | `cpu` | Compute device: `cpu` or `cuda` |
| `--num_workers` | `0` | DataLoader worker processes |

---

## 8. Output Files

All outputs are written to `--output_dir` and updated after every round.

### `gofed_metrics.csv` — Round-level summary

| Column | Description |
|--------|-------------|
| `round` | Global iteration number (1 … T) |
| `global_accuracy` | Accuracy on `centralized_test.csv` after server aggregation |
| `global_f1` | Weighted F1-score on the centralized test set |
| `comm_overhead_MB` | Total communication volume this round: uplink bytes from improving clients + one Mglobal broadcast (BiLSTM parameters only, in MB) |
| `total_energy_J` | Sum of client computation energy (all selected clients) + uplink communication energy (improving clients only), in Joules |
| `num_clients_sent_update` | Number of clients whose LCA improved and transmitted parameters |
| `LCA_client_1` … `LCA_client_N` | Local Classifier Accuracy per client this round |
| `sent_update_client_1` … | `1` if client transmitted parameters, `0` otherwise |

### `gofed_client_details.csv` — Per-client per-round detail

| Column | Description |
|--------|-------------|
| `round` | Round number |
| `client_id` | Client identifier (e.g. `client_1`) |
| `current_lca` | LCA achieved after local training this round |
| `best_lca` | Best LCA ever recorded for this client (transmission threshold) |
| `sent_update` | `1` if parameters were transmitted to server, `0` otherwise |
| `comp_energy_J` | Local computation energy — Eq. 4 |
| `comm_energy_J` | Uplink communication energy — Eq. 7; `0.0` if no update was sent |

### `run_config.json` — Reproducibility record

All CLI arguments saved to JSON. Re-run with identical settings:

```bash
python gofed.py $(python -c "
import json
cfg = json.load(open('results/run_config.json'))
print(' '.join(f'--{k} {v}' for k, v in cfg.items()))
")
```

---

## 9. Reproducing Paper Results

### Similar-task setting (Table 3, upper block)

```bash
python gofed.py \
    --data_dir              Clients                   \
    --output_dir            results/paper_similar     \
    --task_setting          similar                   \
    --num_rounds            20                        \
    --num_local_epochs      1                         \
    --participation_ratio   1.0                       \
    --num_clients           10                        \
    --lr                    0.001                     \
    --batch_size            10                        \
    --optimizer             RMSprop                   \
    --f_dropout             0.1                       \
    --s_dropout             0.3                       \
    --device                cuda                      \
    --seed                  42
```

### Multi-task setting (Table 3, lower block)

```bash
python gofed.py \
    --data_dir              Clients                   \
    --output_dir            results/paper_multitask   \
    --task_setting          multitask                 \
    --num_rounds            20                        \
    --num_local_epochs      1                         \
    --participation_ratio   1.0                       \
    --num_clients           10                        \
    --device                cuda                      \
    --seed                  42
```

### Expected metrics (Table 3 and Figure 5)

| Method | Avg LCA Similar | Avg LCA Multi | Comm Overhead (MB) | Comm Energy (J) |
|--------|:--------------:|:-------------:|:------------------:|:---------------:|
| FedL3 (κ=1.0, s=10) | ~0.800 | ~0.779 | ~1400 | ~1500 |
| FedProx (κ=0.6) | ~0.801 | ~0.780 | ~1100 | ~1250 |
| GoFed (κ=1.0, s=10) | **~0.812** | **~0.791** | **~770** | **~900** |

---

## 10. Energy Model

GoFed tracks two energy components per client per round, following Section 3.2 of the paper.

### Computation energy (Eq. 3–4)

```
W     = N_fl × |D| × E_l                              total FLOPs workload
t_cp  = max(W / f_cpu, W / f_gpu)                     computation time  (Eq. 3)
E_cp  = (φ_cpu · f³_cpu  +  N_SM · φ_gpu · f³_gpu) × t_cp              (Eq. 4)
```

`φ = 2.3e-29` is calibrated so that the resulting device power `P ≈ 5.5 W` and `E_cp ≈ 3–5 J` per client per round — consistent with Figure 5b of the paper.

### Communication energy (Eq. 6–7)

```
R     = B · log₂(1 + P_tx · h² / N₀)                 Shannon capacity   (Eq. 6)
t_cm  = M_bits / R                                     transmission time
E_cm  = P_tx × t_cm                                    energy             (Eq. 7)
```

`E_cm = 0` for clients that do not improve their LCA and therefore do not transmit.

### Total energy per round

```
E_round = Σᵢ E_cp_i   +   Σⱼ E_cm_j
          ──────────       ──────────
          all selected     improvers only
```

> Server computation and downlink energy are **excluded**, consistent with Eq. 15 of the paper.

---

## 11. Troubleshooting

**`CUDA capability sm_120 not compatible` warning**

Your GPU is a Blackwell card (e.g. RTX PRO 3000 / RTX 50-series). Install the nightly PyTorch build with CUDA 12.8 support:

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

Until a stable build is released, training falls back silently to CPU. Pass `--device cpu` to suppress the warning during that period.

---

**`UNEXPECTED keys: vocab_layer_norm …` warning**

Safe to ignore. The downloaded checkpoint was saved from `DistilBertForMaskedLM` (includes an MLM head); loading it as `DistilBertModel` (encoder only) simply discards the MLM head weights. All encoder weights load correctly.

---

**`Remapped labels ['CAG', 'NAG'] → [0, 1]` warning**

Your CSV contains string labels. They are remapped alphabetically (`CAG → 0`, `NAG → 1`). Verify this matches your intended positive class (abusive = 1). For three-class datasets collapsed to binary (OAG + CAG → 1, NAG → 0), pre-convert labels to integers in the CSV before running.

---

**Out-of-memory on GPU**

Reduce batch size and/or sequence length:

```bash
python gofed.py ... --batch_size 4 --max_length 128
```

---

**`FileNotFoundError: centralized_test.csv not found`**

The global evaluation file must be at `<data_dir>/centralized_test.csv`. Create it from a held-out portion of your data, or use `generate_dummy_data.py` to create a synthetic placeholder for testing.

---

**Slow training on CPU**

DistilBERT embedding generation dominates training time. For pipeline verification, use the lightweight mock model:

```bash
python gofed.py --bert_model ./mock_distilbert ...
```

For real experiments, a CUDA-enabled GPU is strongly recommended.
