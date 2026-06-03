import os
import copy
import time
import math
import argparse
import logging
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S",)
logger = logging.getLogger("GoFed")

MULTITASK_HP = {
    "client_1":  {"lr": 0.0014, "batch_size": 16, "optimizer": "Adamax",  "f_dropout": 0.2, "s_dropout": 0.2},
    "client_2":  {"lr": 0.0012, "batch_size": 5,  "optimizer": "Nadam",   "f_dropout": 0.2, "s_dropout": 0.1},
    "client_3":  {"lr": 0.0012, "batch_size": 10, "optimizer": "Adamax",  "f_dropout": 0.3, "s_dropout": 0.3},
    "client_4":  {"lr": 0.001,  "batch_size": 10, "optimizer": "RMSprop", "f_dropout": 0.1, "s_dropout": 0.3},
    "client_5":  {"lr": 0.0014, "batch_size": 16, "optimizer": "Adamax",  "f_dropout": 0.2, "s_dropout": 0.2},
    "client_6":  {"lr": 0.0012, "batch_size": 5,  "optimizer": "Nadam",   "f_dropout": 0.2, "s_dropout": 0.1},
    "client_7":  {"lr": 0.0014, "batch_size": 16, "optimizer": "Adamax",  "f_dropout": 0.2, "s_dropout": 0.2},
    "client_8":  {"lr": 0.0012, "batch_size": 5,  "optimizer": "Nadam",   "f_dropout": 0.2, "s_dropout": 0.1},
    "client_9":  {"lr": 0.0014, "batch_size": 16, "optimizer": "Adamax",  "f_dropout": 0.2, "s_dropout": 0.2},
    "client_10": {"lr": 0.001,  "batch_size": 10, "optimizer": "RMSprop", "f_dropout": 0.1, "s_dropout": 0.3},}

class AbuseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt",)
        return (enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), torch.tensor(int(self.labels[idx]), dtype=torch.long),)

# ─────────────────────────────────────────────
# Model: DistilBERT encoder + BiLSTM classifier
# ─────────────────────────────────────────────
class BERTBiLSTMClassifier(nn.Module):
    def __init__(self, bert_model_name: str, hidden_size: int = 128, num_layers: int = 2, f_dropout: float = 0.1, s_dropout: float = 0.3, 
        device: torch.device = torch.device("cpu"),):
        super().__init__()
        try:
            self.bert = AutoModel.from_pretrained(bert_model_name)
        except Exception:
            from transformers import DistilBertConfig, DistilBertModel
            cfg = DistilBertConfig.from_pretrained(bert_model_name)
            self.bert = DistilBertModel(cfg)
        # Freeze BERT – only BiLSTM is trained and communicated
        for param in self.bert.parameters():
            param.requires_grad = False

        for param in self.bert.parameters():
            param.requires_grad = False
        bert_hidden = getattr(self.bert.config, "dim", getattr(self.bert.config, "hidden_size", 768))

        self.bilstm1 = nn.LSTM(input_size=bert_hidden, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True,)
        self.dropout1 = nn.Dropout(f_dropout)
        self.bilstm2 = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size // 2, num_layers=1, batch_first=True, bidirectional=True,)
        self.dropout2 = nn.Dropout(s_dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = bert_out.last_hidden_state
        out1, _ = self.bilstm1(seq)            # (batch, seq_len, 256)
        out1 = self.dropout1(out1)
        out2, _ = self.bilstm2(out1)           # (batch, seq_len, 128)
        out2 = self.dropout2(out2)
        out = out2[:, -1, :]                   # (batch, 128)
        logit = self.sigmoid(self.fc(out))     # (batch, 1)
        return logit.squeeze(1)

    def get_bilstm_state_dict(self):
        """Return only the BiLSTM + FC parameters (what gets communicated)."""
        keys = [k for k in self.state_dict() if not k.startswith("bert.")]
        return {k: self.state_dict()[k].clone() for k in keys}

    def set_bilstm_state_dict(self, state_dict):
        """Load only the BiLSTM + FC parameters (received from server)."""
        current = self.state_dict()
        current.update(state_dict)
        self.load_state_dict(current)

def build_optimizer(model, name: str, lr: float):
    # Only optimize BiLSTM + FC params
    params = [p for n, p in model.named_parameters() if not n.startswith("bert.")]
    name = name.strip().lower()
    if name == "adamax":
        return torch.optim.Adamax(params, lr=lr)
    elif name == "nadam":
        return torch.optim.NAdam(params, lr=lr)
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    elif name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr)
    else:
        logger.warning(f"Unknown optimizer '{name}', defaulting to Adam.")
        return torch.optim.Adam(params, lr=lr)

# ─────────────────────────────────────────────
# Compute model parameter size in bits
# ─────────────────────────────────────────────
def bilstm_param_size_bits(model: BERTBiLSTMClassifier, bits_per_param: int = 32) -> int:
    """Number of bits in the BiLSTM + FC (communicated) parameters."""
    total = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("bert."))
    return total * bits_per_param

def bilstm_param_size_mb(model: BERTBiLSTMClassifier) -> float:
    return bilstm_param_size_bits(model) / (8 * 1024 * 1024)

# ─────────────────────────────────────────────
# Energy calculation (Section 3.2)
# ─────────────────────────────────────────────
def compute_computation_energy(workload_flops: float, cpu_freq_ghz: float = 2.2, gpu_freq_ghz: float = 1.395, cpu_flops_per_cycle: float = 8.0,
    gpu_flops_per_cycle: float = 128.0, phi_cpu: float = 1e-27, phi_gpu: float = 1e-27, n_sm: int = 84,) -> float:
    """
    Compute energy (Joules) for one local iteration based on Eq. 3-4 from the paper (simplified).
    E_cp = (phi * f_cl^3 + phi'*f_cl'^3) * t_cp
    """
    f_cpu = cpu_freq_ghz * 1e9
    f_gpu = gpu_freq_ghz * 1e9
    speed_cpu = f_cpu * cpu_flops_per_cycle
    speed_gpu = f_gpu * gpu_flops_per_cycle
    t_cp = max(workload_flops / speed_cpu, workload_flops / speed_gpu)
    phi_prime = n_sm * phi_gpu
    E_cp = (phi_cpu * (cpu_freq_ghz * 1e9) ** 3 + phi_prime * (gpu_freq_ghz * 1e9) ** 3) * t_cp
    return E_cp

def compute_communication_energy(param_bits: int,
    bandwidth_hz: float = 1e6,         # 1 MHz — narrow-band uplink (resource-constrained device)
    tx_power_w: float = 0.5,           # 500 mW transmit power
    channel_gain: float = 1.0,
    noise_psd: float = 1e-9,) -> float:
    snr       = tx_power_w * (channel_gain ** 2) / noise_psd
    data_rate = bandwidth_hz * math.log2(1.0 + snr)     # bits/s  (Eq. 6)
    t_cm      = param_bits / data_rate                  # seconds
    E_cm      = tx_power_w * t_cm                       # Joules  (= P_tx * M/R)
    return E_cm

# ─────────────────────────────────────────────
# Training / evaluation 
# ─────────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        preds = model(input_ids, attention_mask)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        preds = model(input_ids, attention_mask)
        predicted = (preds.cpu().numpy() > 0.5).astype(int)
        all_preds.extend(predicted.tolist())
        all_labels.extend(labels.numpy().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return acc, f1

# ─────────────────────────────────────────────
# GoFed Server
# ─────────────────────────────────────────────
class GoFedServer:
    """
    Maintains a copy of the best (highest LCA) BiLSTM parameters for each client. Aggregates parameters using weighted FedAvg (weighted by dataset size).
    """
    def __init__(self, client_ids: list):
        self.param_copies: dict = {cid: None for cid in client_ids}  # best param copy per client
        self.data_sizes: dict = {cid: 1 for cid in client_ids}
        self.global_params = None  # latest aggregated global params

    def update_copy(self, client_id: str, params: dict, data_size: int):
        """Update stored copy when client sends improved params."""
        self.param_copies[client_id] = copy.deepcopy(params)
        self.data_sizes[client_id] = data_size

    def aggregate(self, participating_clients: list) -> dict:
        """
        FedAvg over participating clients. For clients that did NOT send updates this round, use the stored parameter copy (best LCA copy).
        """
        total_size = sum(self.data_sizes[cid] for cid in participating_clients)
        aggregated = None

        for cid in participating_clients:
            params = self.param_copies[cid]
            if params is None:
                logger.warning(f"No param copy for {cid}; skipping in aggregation.")
                continue
            weight = self.data_sizes[cid] / total_size
            if aggregated is None:
                aggregated = {k: weight * v.float() for k, v in params.items()}
            else:
                for k in aggregated:
                    aggregated[k] += weight * params[k].float()

        self.global_params = aggregated
        return aggregated

# ─────────────────────────────────────────────
# GoFed Client
# ─────────────────────────────────────────────
class GoFedClient:
    def __init__(self, client_id: str, train_loader: DataLoader, test_loader: DataLoader, model: BERTBiLSTMClassifier, optimizer_name: str,
        lr: float, device: torch.device, data_size: int,
    ):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model.to(device)
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.device = device
        self.data_size = data_size

        self.best_lca: float = 0.0          # best LCA seen so far
        self.current_lca: float = 0.0
        self.criterion = nn.BCELoss()

    def receive_global_params(self, global_params: dict):
        """Step 2: Load global BiLSTM parameters sent by server."""
        if global_params is not None:
            self.model.set_bilstm_state_dict(global_params)

    def local_train(self, num_epochs: int):
        """Step 2-3: Train locally for num_epochs."""
        optimizer = build_optimizer(self.model, self.optimizer_name, self.lr)
        for _ in range(num_epochs):
            train_one_epoch(self.model, self.train_loader, optimizer, self.criterion, self.device)

    def compute_lca(self) -> float:
        """Step 3: Evaluate on local test set → LCA."""
        acc, _ = evaluate(self.model, self.test_loader, self.device)
        self.current_lca = acc
        return acc

    def should_send_update(self, iteration: int) -> bool:
        """
        Step 4-5: Compare current LCA with best saved LCA. At t=1, always send (no previous LCA to compare against).
        Returns True if parameters should be sent to server.
        """
        if iteration == 1:
            # First round: always send
            self.best_lca = self.current_lca
            return True
        if self.current_lca > self.best_lca:
            self.best_lca = self.current_lca
            return True
        return False

    def get_params(self) -> dict:
        return self.model.get_bilstm_state_dict()

class MetricsLogger:
    def __init__(self, output_dir: str, n_clients: int):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        # ---- per-round summary CSV ----
        self.summary_path = os.path.join(output_dir, "gofed_metrics.csv")
        client_lca_headers = [f"LCA_client_{i+1}" for i in range(n_clients)]
        client_sent_headers = [f"sent_update_client_{i+1}" for i in range(n_clients)]
        self.summary_headers = (["round", "global_accuracy", "global_f1", "comm_overhead_MB", "total_energy_J", "num_clients_sent_update"]
            + client_lca_headers + client_sent_headers)
        with open(self.summary_path, "w", newline="") as f:
            csv.writer(f).writerow(self.summary_headers)

        # ---- detailed per-client CSV ----
        self.detail_path = os.path.join(output_dir, "gofed_client_details.csv")
        with open(self.detail_path, "w", newline="") as f:
            csv.writer(f).writerow(["round", "client_id", "current_lca", "best_lca", "sent_update", "comp_energy_J", "comm_energy_J"])

        self.n_clients = n_clients

    def log_round(
        self,
        round_num: int,
        global_acc: float,
        global_f1: float,
        comm_overhead_mb: float,
        total_energy: float,
        num_sent: int,
        lca_per_client: dict,    # {client_id: lca}
        sent_per_client: dict,   # {client_id: bool}
        client_details: list,    # list of dicts
    ):
        # Summary row
        n = self.n_clients
        lca_vals = [lca_per_client.get(f"client_{i+1}", float("nan")) for i in range(n)]
        sent_vals = [int(sent_per_client.get(f"client_{i+1}", False)) for i in range(n)]

        row = ([round_num, round(global_acc, 6), round(global_f1, 6), round(comm_overhead_mb, 4), round(total_energy, 6), num_sent]
            + [round(v, 6) if not math.isnan(v) else "" for v in lca_vals] + sent_vals)
        with open(self.summary_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        # Detail rows
        with open(self.detail_path, "a", newline="") as f:
            writer = csv.writer(f)
            for d in client_details:
                writer.writerow([round_num, d["client_id"], round(d["current_lca"], 6),
                    round(d["best_lca"], 6), int(d["sent_update"]), round(d["comp_energy"], 6), round(d["comm_energy"], 6),])

        logger.info(f"Round {round_num:3d} | GlobalAcc={global_acc:.4f} | "
            f"CommOverhead={comm_overhead_mb:.2f}MB | Energy={total_energy:.4f}J | "
            f"ClientsSent={num_sent}")

def load_csv(path: str):
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    text_col = next((c for c in df.columns if c.lower() == "text"), None)
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if text_col is None or label_col is None:
        raise ValueError(f"CSV at {path} must have 'Text' and 'Label' columns. Found: {list(df.columns)}")
    # Drop rows with NaN
    df = df[[text_col, label_col]].dropna()
    # Ensure binary labels 0/1
    unique_labels = sorted(df[label_col].unique())
    if set(unique_labels) - {0, 1}:
        # Map to 0/1 if needed
        label_map = {v: i for i, v in enumerate(unique_labels)}
        df[label_col] = df[label_col].map(label_map)
        logger.warning(f"Remapped labels {unique_labels} → {list(label_map.values())} in {path}")
    return df[text_col].tolist(), df[label_col].tolist()

# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="GoFed: Federated Abuse Classification")

    # ── Paths ──────────────────────────────────
    parser.add_argument("--data_dir", type=str, default="Clients", help="Root directory containing client_1/ … client_10/ subdirs")
    parser.add_argument("--output_dir", type=str, default="gofed_results", help="Directory to save metrics CSV files")
    parser.add_argument("--bert_model", type=str, default="distilbert-base-multilingual-cased", help="HuggingFace model name for BERT backbone")

    # ── Federated settings ─────────────────────
    parser.add_argument("--num_clients", type=int, default=10, help="Total number of federated clients")
    parser.add_argument("--num_rounds", type=int, default=2, help="Number of global federated rounds (T)")
    parser.add_argument("--num_local_epochs", type=int, default=1, help="Number of local training epochs per round (E_l)")
    parser.add_argument("--participation_ratio", type=float, default=0.4, help="Fraction of clients selected per round (kappa)")

    # ── Task setting ───────────────────────────
    parser.add_argument("--task_setting", type=str, default="similar", choices=["similar", "multitask"],
        help="'similar': same HP for all clients; 'multitask': per-client HP from Table 2")

    # ── Similar-task hyperparameters ───────────
    parser.add_argument("--lr",         type=float, default=0.001,  help="Learning rate (similar-task)")
    parser.add_argument("--batch_size", type=int,   default=10,     help="Batch size (similar-task)")
    parser.add_argument("--optimizer",  type=str,   default="RMSprop", help="Optimizer (similar-task)")
    parser.add_argument("--f_dropout",  type=float, default=0.1,    help="First BiLSTM dropout (similar-task)")
    parser.add_argument("--s_dropout",  type=float, default=0.3,    help="Second BiLSTM dropout (similar-task)")

    # ── Model architecture ─────────────────────
    parser.add_argument("--max_length",  type=int, default=256, help="Max tokenizer sequence length")
    parser.add_argument("--hidden_size", type=int, default=128, help="BiLSTM hidden size (first layer)")

    # ── Energy model parameters (Configure it as per the Device) ────────────────
    parser.add_argument("--cpu_freq_ghz",       type=float, default=2.2,   help="CPU clock freq (GHz)")
    parser.add_argument("--gpu_freq_ghz",       type=float, default=1.395, help="GPU clock freq (GHz)")
    parser.add_argument("--bandwidth_hz",        type=float, default=1e6,   help="Wireless uplink bandwidth Hz; 1 MHz for resource-constrained device")
    parser.add_argument("--tx_power_w",          type=float, default=0.5,   help="Transmit power W; 0.5W default")
    parser.add_argument("--channel_gain",        type=float, default=1.0,   help="Channel gain h_i")
    parser.add_argument("--noise_psd",           type=float, default=1e-9,  help="Noise PSD N0")

    # ── Misc ───────────────────────────────────
    parser.add_argument("--seed",      type=int,  default=42,   help="Random seed")
    parser.add_argument("--device",    type=str,  default="cuda", help="'cpu' or 'cuda'")
    parser.add_argument("--num_workers", type=int, default=4,   help="DataLoader workers")

    return parser.parse_args()

# ─────────────────────────────────────────────
# Main GoFed training loop
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Task setting: {args.task_setting}")

    # ── Load tokenizer ─────────────────────────
    logger.info(f"Loading tokenizer: {args.bert_model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    except Exception:
        logger.warning("Fast tokenizer failed; trying slow tokenizer …")
        from transformers import DistilBertTokenizer
        vocab_path = os.path.join(args.bert_model, "vocab.txt")
        if os.path.exists(vocab_path):
            tokenizer = DistilBertTokenizer(vocab_path)
        else:
            tokenizer = DistilBertTokenizer.from_pretrained(args.bert_model)

    # ── Per-client hyperparameters ─────────────
    def get_hp(cid: str):
        if args.task_setting == "multitask":
            hp = MULTITASK_HP.get(cid, {})
            return (hp.get("lr", args.lr), hp.get("batch_size", args.batch_size), hp.get("optimizer", args.optimizer), 
                hp.get("f_dropout", args.f_dropout), hp.get("s_dropout", args.s_dropout),)
        # similar-task: all clients share CLI args
        return args.lr, args.batch_size, args.optimizer, args.f_dropout, args.s_dropout

    # ── Load data and build clients ────────────
    clients: list[GoFedClient] = []
    client_ids = [f"Client_{i+1}" for i in range(args.num_clients)]

    logger.info("Loading client datasets …")
    for cid in client_ids:
        client_dir = os.path.join(args.data_dir, cid)
        train_path = os.path.join(client_dir, "train.csv")
        test_path  = os.path.join(client_dir, "test.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing train.csv or test.csv in {client_dir}\n Expected structure: {args.data_dir}/client_X/train.csv and test.csv")

        train_texts, train_labels = load_csv(train_path)
        test_texts,  test_labels  = load_csv(test_path)
        lr, bs, opt_name, f_drop, s_drop = get_hp(cid)
        train_ds = AbuseDataset(train_texts, train_labels, tokenizer, args.max_length)
        test_ds  = AbuseDataset(test_texts,  test_labels,  tokenizer, args.max_length)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=args.num_workers)
        test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=args.num_workers)

        model = BERTBiLSTMClassifier(bert_model_name=args.bert_model, hidden_size=args.hidden_size, f_dropout=f_drop, s_dropout=s_drop, device=device,)

        client = GoFedClient(
            client_id=cid, train_loader=train_loader, test_loader=test_loader, model=model, optimizer_name=opt_name, lr=lr, device=device, data_size=len(train_texts),)
        clients.append(client)
        logger.info(f"  {cid}: train={len(train_texts)}, test={len(test_texts)}, "
            f"lr={lr}, bs={bs}, opt={opt_name}, f_drop={f_drop}, s_drop={s_drop}")

    # ── Load centralized test set ──────────────
    central_test_path = "centralized.csv"
    if not os.path.exists(central_test_path):
        raise FileNotFoundError(f"centralized_test.csv not found at {central_test_path}")
    central_texts, central_labels = load_csv(central_test_path)
    central_bs = args.batch_size
    central_ds = AbuseDataset(central_texts, central_labels, tokenizer, args.max_length)
    central_loader = DataLoader(central_ds, batch_size=central_bs, shuffle=False, num_workers=args.num_workers)
    logger.info(f"Centralized test set: {len(central_texts)} samples")
    _, _, opt_name_g, f_drop_g, s_drop_g = get_hp("client_1")
    global_eval_model = BERTBiLSTMClassifier(
        bert_model_name=args.bert_model, hidden_size=args.hidden_size, f_dropout=f_drop_g, s_dropout=s_drop_g, device=device,).to(device)

    # ── Init server ────────────────────────────
    server = GoFedServer(client_ids)

    # Initialize server param copies from initial client models (round 0)
    for client in clients:
        server.update_copy(client.client_id, client.get_params(), client.data_size)
    # First aggregation to set global params
    initial_global = server.aggregate(client_ids)

    # Broadcast initial global params to all clients
    for client in clients:
        client.receive_global_params(initial_global)

    # ── Metrics logger ─────────────────────────
    metrics_logger = MetricsLogger(args.output_dir, args.num_clients)

    # ── Compute static param size for comm metrics ──
    param_bits = bilstm_param_size_bits(clients[0].model)
    param_mb   = bilstm_param_size_mb(clients[0].model)
    logger.info(f"BiLSTM param size: {param_mb:.4f} MB ({param_bits} bits)")

    num_bilstm_params = sum(p.numel() for n, p in clients[0].model.named_parameters() if not n.startswith("bert."))
    flops_per_sample = num_bilstm_params * 2  # rough estimate

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting GoFed: {args.num_rounds} rounds, {args.num_clients} clients")
    logger.info(f"Participation ratio κ = {args.participation_ratio}")
    logger.info(f"{'='*60}\n")

    for t in range(1, args.num_rounds + 1):
        logger.info(f"\n--- Round {t}/{args.num_rounds} ---")

        # Step 1: Select clients
        n_selected = max(1, int(args.participation_ratio * args.num_clients))
        selected_indices = np.random.choice(len(clients), size=n_selected, replace=False)
        selected_clients = [clients[i] for i in sorted(selected_indices)]
        selected_ids     = [c.client_id for c in selected_clients]
        logger.info(f"Selected clients ({n_selected}): {selected_ids}")

        # Broadcast current global params to selected clients (Step 1→2)
        current_global = server.global_params
        for client in selected_clients:
            client.receive_global_params(current_global)

        # Per-round tracking
        lca_per_client    = {}
        sent_per_client   = {}
        client_details    = []
        total_comm_mb     = 0.0
        total_energy_j    = 0.0
        clients_sent      = 0

        for client in selected_clients:
            t0 = time.time()

            # Step 2: Local training
            client.local_train(args.num_local_epochs)
            t_local = time.time() - t0

            # Step 3: Evaluate → LCA
            lca = client.compute_lca()
            lca_per_client[client.client_id] = lca
            logger.info(f"  {client.client_id}: LCA={lca:.4f} (best={client.best_lca:.4f})")

            # Step 4-5: Decide whether to send update
            send_update = client.should_send_update(t)
            sent_per_client[client.client_id] = send_update

            # ── Computation energy ──
            dataset_size = client.data_size
            workload = flops_per_sample * dataset_size * args.num_local_epochs
            comp_energy = compute_computation_energy(workload_flops=workload, cpu_freq_ghz=args.cpu_freq_ghz, gpu_freq_ghz=args.gpu_freq_ghz,)
            total_energy_j += comp_energy

            # ── Communication energy & overhead ──
            if send_update:
                # Client sends params to server
                server.update_copy(client.client_id, client.get_params(), client.data_size)
                comm_energy = compute_communication_energy(
                    param_bits=param_bits, bandwidth_hz=args.bandwidth_hz, tx_power_w=args.tx_power_w, channel_gain=args.channel_gain, noise_psd=args.noise_psd,)
                total_energy_j  += comm_energy
                total_comm_mb   += param_mb
                clients_sent    += 1
            else:
                # Server uses saved copy — no communication energy from this client
                comm_energy = 0.0
                logger.info(f"  {client.client_id}: No update sent (LCA not improved)")

            client_details.append({
                "client_id": client.client_id, "current_lca": lca, "best_lca": client.best_lca,
                "sent_update": send_update, "comp_energy": comp_energy, "comm_energy": comm_energy,})

        # Step 6-8: Server aggregation
        aggregated_params = server.aggregate(selected_ids)

        # ── Communication overhead (paper Eq. 17 / 19) ──────────────
        # uplink  = clients_sent x param_mb  (already in total_comm_mb)
        # Mglobal = param_mb x 1  (one global model broadcast, counted once)
        total_comm_mb += param_mb   # Mglobal: one broadcast per round

        # Downlink energy (server → clients) and server computation energy are NOT included in E_tot per paper Eq. 15, which sums
        # only client-side E_cp_i (local training) + E_cm_i (uplink when LCA improves). Downlink and server costs are excluded.

        # Evaluate on centralized test (global model)
        global_eval_model.set_bilstm_state_dict(aggregated_params)
        global_acc, global_f1 = evaluate(global_eval_model, central_loader, device)
        logger.info(f"  Global accuracy: {global_acc:.4f}  F1: {global_f1:.4f}")
        logger.info(f"  Comm overhead: {total_comm_mb:.2f} MB | Energy: {total_energy_j:.4f} J")

        # Log metrics
        metrics_logger.log_round(
            round_num=t,
            global_acc=global_acc,
            global_f1=global_f1,
            comm_overhead_mb=total_comm_mb,
            total_energy=total_energy_j,
            num_sent=clients_sent,
            lca_per_client=lca_per_client,
            sent_per_client=sent_per_client,
            client_details=client_details,
        )

    # ── Final summary ──────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("GoFed training complete.")
    logger.info(f"Results saved to: {args.output_dir}/")
    logger.info(f"  Summary CSV : gofed_metrics.csv")
    logger.info(f"  Details CSV : gofed_client_details.csv")
    logger.info(f"{'='*60}")

    # Save final config
    config_path = os.path.join(args.output_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"  Run config  : run_config.json")

if __name__ == "__main__":
    main()