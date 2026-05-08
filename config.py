# config.py — TKG Elite Model konfiguratsiyasi
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset:   str = "ICEWS18"
    data_dir:  str = "data"

    # ── Embedding o'lchamlari ─────────────────────────────────────────────────
    entity_dim:   int = 256    # entity embedding
    relation_dim: int = 256    # relation embedding
    time_dim:     int = 64     # vaqt embedding
    hidden_dim:   int = 512    # GRU / Transformer hidden
    proj_dim:     int = 256    # contrastive projection space

    # ── Encoder ───────────────────────────────────────────────────────────────
    encoder_type:   str = "transformer"  # "gru" | "transformer"
    num_layers:     int = 2
    num_heads:      int = 8              # Transformer heads
    ffn_dim:        int = 1024           # Transformer FFN
    dropout:        float = 0.1
    max_seq_len:    int = 16             # max path uzunligi

    # ── Path sampling ─────────────────────────────────────────────────────────
    num_paths:      int = 8
    max_path_len:   int = 3

    # ── Contrastive ───────────────────────────────────────────────────────────
    num_negative:   int = 256
    temperature:    float = 0.3          # optimal: 0.1-0.5
    momentum:       float = 0.995        # MoCo-style EMA encoder
    queue_size:     int = 8192           # negative queue (MoCo)
    use_moco:       bool = True          # MoCo v2 style contrastive

    # ── Temporal encoding ─────────────────────────────────────────────────────
    use_time_encoding: str = "learned"   # "learned" | "sinusoidal" | "both"
    time_granularity:  int = 1           # vaqt granulyarligi

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:     int = 512
    num_epochs:     int = 50
    learning_rate:  float = 3e-4
    weight_decay:   float = 1e-4
    grad_clip:      float = 1.0
    warmup_steps:   int = 1000           # qadamlar soni (epochlar emas)
    label_smoothing: float = 0.1

    # ── Loss weights ──────────────────────────────────────────────────────────
    w_link:         float = 1.0          # link prediction loss
    w_contrastive:  float = 0.5          # contrastive loss
    w_self_adv:     float = 0.5          # self-adversarial negative sampling
    w_direct:       float = 0.0          # DistMult direct scoring (WIKI/YAGO uchun 1.0+)

    # ── Direct scoring ────────────────────────────────────────────────────────
    use_direct_scoring: bool = False     # DistMult-style to'g'ridan scoring
    use_diachronic:     bool = False     # DE-SimplE style temporal gating

    # ── Temporal History (RE-GCN uslubi) ─────────────────────────────────────
    use_history:    bool = False         # Entity tarixini agregatsiyalash
    max_history:    int  = 16            # Har entity uchun max tarix uzunligi
    use_reciprocal: bool = False         # Teskari tripllarni ham o'qitish

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_every:     int = 1
    hits_at_k:      List[int] = field(default_factory=lambda: [1, 3, 10])
    filter_flag:    bool = True

    # ── System ────────────────────────────────────────────────────────────────
    device:         str = "cuda"
    seed:           int = 42
    num_workers:    int = 4
    save_dir:       str = "checkpoints"
    log_dir:        str = "logs"
    resume:         Optional[str] = None
    fp16:           bool = True          # mixed precision

    # Runtime (dataset yuklanganidan keyin to'ldiriladi)
    num_entities:   int = 0
    num_relations:  int = 0
    num_times:      int = 0


DATASET_STATS = {
    "ICEWS14": {"num_entities": 7128,  "num_relations": 230,  "num_times": 365},
    "ICEWS18": {"num_entities": 23033, "num_relations": 256,  "num_times": 7273},
    "WIKI":    {"num_entities": 12554, "num_relations": 24,   "num_times": 232},
    "YAGO":    {"num_entities": 10623, "num_relations": 10,   "num_times": 189},
    "YAGOs":   {"num_entities": 10623, "num_relations": 10,   "num_times": 189},
    "GDELT":   {"num_entities": 7691,  "num_relations": 240,  "num_times": 2975},
}
