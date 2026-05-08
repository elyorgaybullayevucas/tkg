# config.py — STORM Model konfiguratsiyasi
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset:   str = "ICEWS18"
    data_dir:  str = "data"

    # ── Embedding o'lchamlari ─────────────────────────────────────────────────
    entity_dim:   int = 256
    relation_dim: int = 256
    delta_dim:    int = 64     # Relative Δt encoding o'lchami
    hidden_dim:   int = 512
    # (eski time_dim, proj_dim lar backward compat uchun saqlanadi)
    time_dim:     int = 64
    proj_dim:     int = 256

    # ── Encoder ───────────────────────────────────────────────────────────────
    num_layers:   int   = 2
    num_heads:    int   = 8
    ffn_dim:      int   = 1024
    dropout:      float = 0.1

    # ── ORION: Temporal Pattern Library ──────────────────────────────────────
    num_patterns:  int   = 128
    w_pattern_div: float = 0.01

    # ── Path sampling ─────────────────────────────────────────────────────────
    num_paths:    int = 8
    max_path_len: int = 3
    max_seq_len:  int = 16

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:      int   = 512
    num_epochs:      int   = 50
    learning_rate:   float = 3e-4
    weight_decay:    float = 1e-4
    grad_clip:       float = 1.0
    label_smoothing: float = 0.1
    num_negative:    int   = 256

    # ── Loss weights ──────────────────────────────────────────────────────────
    w_link:        float = 1.0
    w_contrastive: float = 0.01  # ORION: pattern diversity regularization
    w_self_adv:    float = 0.5
    w_direct:      float = 0.0
    w_ortho_reg:   float = 0.0

    # ── Direct scoring & Diachronic ───────────────────────────────────────────
    use_direct_scoring: bool = False
    use_diachronic:     bool = False

    # ── Temporal History ──────────────────────────────────────────────────────
    use_history:    bool = False
    max_history:    int  = 16
    use_reciprocal: bool = False

    # ── Backward compat (STORM da ishlatilmaydi) ──────────────────────────────
    temperature:        float = 0.3
    momentum:           float = 0.995
    queue_size:         int   = 8192
    use_moco:           bool  = False
    use_time_encoding:  str   = "sinusoidal"

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_every:  int       = 1
    hits_at_k:   List[int] = field(default_factory=lambda: [1, 3, 10])
    filter_flag: bool      = True

    # ── System ────────────────────────────────────────────────────────────────
    device:      str           = "cuda"
    seed:        int           = 42
    num_workers: int           = 4
    save_dir:    str           = "checkpoints"
    log_dir:     str           = "logs"
    resume:      Optional[str] = None
    fp16:        bool          = True

    # Runtime
    num_entities:  int = 0
    num_relations: int = 0
    num_times:     int = 0


DATASET_STATS = {
    "ICEWS14": {"num_entities": 7128,  "num_relations": 230,  "num_times": 365},
    "ICEWS18": {"num_entities": 23033, "num_relations": 256,  "num_times": 7273},
    "WIKI":    {"num_entities": 12554, "num_relations": 24,   "num_times": 232},
    "YAGO":    {"num_entities": 10623, "num_relations": 10,   "num_times": 189},
    "YAGOs":   {"num_entities": 10623, "num_relations": 10,   "num_times": 189},
    "GDELT":   {"num_entities": 7691,  "num_relations": 240,  "num_times": 2975},
}
