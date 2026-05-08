# main.py
"""
STORM Model — ishga tushirish.

Ishlatish:
    python main.py --dataset ICEWS18
    python main.py --dataset WIKI
    python main.py --dataset YAGO --epochs 300
    python main.py --resume checkpoints/ICEWS18_best.pt
"""
import argparse, os, random
import numpy as np
import torch

from config import Config, DATASET_STATS
from data.datamodule import TKGDataModule
from models.elite_tkg_model import CATREModel as EliteTKGModel
from trainers.trainer import EliteTrainer
from utils.logging import get_logger


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_args() -> Config:
    p = argparse.ArgumentParser("CATRE — Cross-scale Adaptive Temporal Reasoning for Extrapolation")

    p.add_argument("--dataset",  default="ICEWS18",
                   choices=["ICEWS14", "ICEWS18", "WIKI", "YAGO", "YAGOs", "GDELT"])
    p.add_argument("--data_dir", default="data")

    p.add_argument("--entity_dim",   type=int,   default=256)
    p.add_argument("--relation_dim", type=int,   default=256)
    p.add_argument("--delta_dim",    type=int,   default=64)
    p.add_argument("--hidden_dim",   type=int,   default=512)
    p.add_argument("--num_heads",    type=int,   default=8)
    p.add_argument("--num_layers",   type=int,   default=2)
    p.add_argument("--ffn_dim",      type=int,   default=1024)
    p.add_argument("--dropout",      type=float, default=0.1)

    p.add_argument("--num_paths",    type=int, default=8)
    p.add_argument("--max_path_len", type=int, default=3)
    p.add_argument("--num_negative", type=int, default=256)

    p.add_argument("--batch_size",      type=int,   default=512)
    p.add_argument("--epochs",          type=int,   default=50,  dest="num_epochs")
    p.add_argument("--lr",              type=float, default=3e-4, dest="learning_rate")
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)

    p.add_argument("--w_link",        type=float, default=1.0)
    p.add_argument("--w_self_adv",    type=float, default=0.5)
    p.add_argument("--w_direct",      type=float, default=0.0)
    p.add_argument("--w_ortho_reg",   type=float, default=0.0)

    p.add_argument("--use_direct_scoring", action="store_true")
    p.add_argument("--use_diachronic",     action="store_true")
    p.add_argument("--use_history",        action="store_true")
    p.add_argument("--max_history",        type=int, default=16)
    p.add_argument("--use_reciprocal",     action="store_true")

    p.add_argument("--device",      default="cuda")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir",    default="checkpoints")
    p.add_argument("--log_dir",     default="logs")
    p.add_argument("--resume",      default=None)
    p.add_argument("--no_fp16",     action="store_true")
    p.add_argument("--eval_every",  type=int, default=1)

    args = p.parse_args()
    cfg  = Config()
    for k, v in vars(args).items():
        if k == "no_fp16":
            cfg.fp16 = not v
        elif k in ("use_direct_scoring", "use_diachronic", "use_history", "use_reciprocal") and v:
            setattr(cfg, k, True)
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def main():
    cfg = parse_args()
    seed_everything(cfg.seed)
    logger = get_logger("main", cfg.log_dir)

    if not torch.cuda.is_available():
        cfg.device = "cpu"
        cfg.fp16   = False
        logger.warning("CUDA mavjud emas — CPU ishlatiladi")

    logger.info(f"Device: {cfg.device}  |  FP16: {cfg.fp16}")

    # ── DataModule ────────────────────────────────────────────────────────────
    logger.info(f"Dataset yuklanmoqda: {cfg.dataset}")
    dm = TKGDataModule(cfg)
    dm.setup()
    logger.info(
        f"Dataset={cfg.dataset} | "
        f"|E|={cfg.num_entities} | "
        f"|R|={cfg.num_relations} | "
        f"|T|={cfg.num_times}"
    )
    logger.info(
        f"Train:{len(dm.train_ds):,}  "
        f"Valid:{len(dm.valid_ds):,}  "
        f"Test:{len(dm.test_ds):,}"
    )

    # ── Dataset-specific sozlamalar ───────────────────────────────────────────
    if cfg.dataset == "GDELT":
        cfg.num_paths    = 3
        cfg.max_path_len = 2
        cfg.batch_size   = 1024
        logger.info("GDELT: num_paths=3, max_path_len=2, batch_size=1024")

    if cfg.dataset in ("WIKI", "YAGO", "YAGOs"):
        cfg.num_paths          = 8
        cfg.max_path_len       = 3
        cfg.num_negative       = 256

        # Multi-scale neighborhood aggregation (CATRE)
        # max_history=64: (B,H,E) = (256,64,256) = 4M elems × 2B = 8MB — GPU safe
        cfg.use_history        = True
        cfg.max_history        = 64
        cfg.batch_size         = 256

        # Direct scoring + diachronic
        cfg.use_direct_scoring = True
        cfg.use_diachronic     = True
        cfg.w_direct           = 2.0

        # Loss weights
        cfg.w_link             = 1.0
        cfg.w_self_adv         = 0.5
        cfg.w_ortho_reg        = 0.02

        # Regularization
        cfg.dropout            = 0.15
        cfg.label_smoothing    = 0.1
        cfg.weight_decay       = 1e-4

        # Training
        cfg.learning_rate      = 5e-4
        cfg.num_epochs         = 500

        logger.info(
            f"{cfg.dataset}: CATRE sozlamalari:\n"
            f"  use_history=True (max_history={cfg.max_history}), "
            f"DirectScoring=True, Diachronic=True, w_direct={cfg.w_direct},\n"
            f"  MultiScaleAggregator (learnable τ₁,τ₂), RelationalMemory (DaeMon-inspired),\n"
            f"  EntityIndependent paths, ThreeSignalFusion,\n"
            f"  epochs={cfg.num_epochs}, LR={cfg.learning_rate}"
        )

    if cfg.dataset == "ICEWS18":
        cfg.use_history        = True
        cfg.max_history        = 64
        cfg.w_self_adv         = 0.5
        cfg.w_ortho_reg        = 0.01
        cfg.num_epochs         = 50
        logger.info("ICEWS18: CATRE sozlamalari: use_history=True, max_history=64")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EliteTKGModel(
        num_entities        = cfg.num_entities,
        num_relations       = cfg.num_relations,
        num_times           = cfg.num_times,
        entity_dim          = cfg.entity_dim,
        relation_dim        = cfg.relation_dim,
        delta_dim           = cfg.delta_dim,
        hidden_dim          = cfg.hidden_dim,
        num_heads           = cfg.num_heads,
        num_layers          = cfg.num_layers,
        ffn_dim             = cfg.ffn_dim,
        num_negative        = cfg.num_negative,
        dropout             = cfg.dropout,
        label_smoothing     = cfg.label_smoothing,
        w_direct            = cfg.w_direct,
        use_direct_scoring  = cfg.use_direct_scoring,
        use_diachronic      = cfg.use_diachronic,
        use_history         = cfg.use_history,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parametrlar: {n_params/1e6:.2f}M")

    # ── Multi-GPU (DataParallel) ───────────────────────────────────────────────
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and cfg.device == "cuda":
        model = torch.nn.DataParallel(model)
        logger.info(f"DataParallel: {n_gpus} ta GPU")
    else:
        logger.info(f"Single GPU/CPU")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = EliteTrainer(
        model         = model,
        cfg           = cfg,
        train_loader  = dm.train_loader(),
        valid_loader  = dm.valid_loader(),
        test_loader   = dm.test_loader(),
        valid_dataset = dm.valid_ds,
        test_dataset  = dm.test_ds,
    )

    test_metrics = trainer.fit()
    logger.info("O'qitish yakunlandi!")
    logger.info(f"Test: {test_metrics}")


if __name__ == "__main__":
    main()
