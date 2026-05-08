# utils/logging.py
import logging, os, sys
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(
        os.path.join(log_dir, f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"),
        encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
