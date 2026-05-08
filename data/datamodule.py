# data/datamodule.py
import torch
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import List, Optional
from config import Config
from data.dataset import TKGEliteDataset


def collate_fn(batch: List[dict]) -> dict:
    B = len(batch)
    P = len(batch[0]["paths"])

    # Max path length
    max_len = max(
        len(step_list)
        for item in batch
        for step_list in item["paths"]
    )
    max_len = max(max_len, 1)

    # Paths tensor: (B, P, L, 3)
    paths_t = torch.zeros(B, P, max_len, 3, dtype=torch.long)
    masks_t = torch.zeros(B, P, max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        for j, path in enumerate(item["paths"]):
            for k, (o, r, t) in enumerate(path):
                paths_t[i, j, k, 0] = o
                paths_t[i, j, k, 1] = r
                paths_t[i, j, k, 2] = t
                masks_t[i, j, k] = True

    # Negative objects: (B, N) — faqat train da
    neg_objs = [item["neg_objects"] for item in batch]
    has_neg = len(neg_objs[0]) > 0
    if has_neg:
        N = min(len(n) for n in neg_objs)
        neg_t = torch.tensor([n[:N] for n in neg_objs], dtype=torch.long)
    else:
        neg_t = torch.empty(B, 0, dtype=torch.long)

    # ── History collation: (B, H_len, 3) ─────────────────────────────────────
    histories = [item["history"] for item in batch]
    max_hist  = max((len(h) for h in histories), default=0)
    if max_hist > 0:
        hist_t = torch.zeros(B, max_hist, 3, dtype=torch.long)
        hist_m = torch.zeros(B, max_hist, dtype=torch.bool)
        for i, hist in enumerate(histories):
            for j, (nb, rl, tm) in enumerate(hist):
                hist_t[i, j, 0] = nb
                hist_t[i, j, 1] = rl
                hist_t[i, j, 2] = tm
                hist_m[i, j]    = True
    else:
        hist_t = torch.zeros(B, 1, 3, dtype=torch.long)
        hist_m = torch.zeros(B, 1, dtype=torch.bool)

    return {
        "subject":      torch.tensor([b["subject"]  for b in batch], dtype=torch.long),
        "relation":     torch.tensor([b["relation"] for b in batch], dtype=torch.long),
        "object":       torch.tensor([b["object"]   for b in batch], dtype=torch.long),
        "time":         torch.tensor([b["time"]     for b in batch], dtype=torch.long),
        "paths":        paths_t,    # (B, P, L, 3)
        "path_masks":   masks_t,    # (B, P, L)
        "neg_objects":  neg_t,      # (B, N)
        "true_objects": [b["true_objects"] for b in batch],
        "history":      hist_t,     # (B, H_len, 3)
        "hist_mask":    hist_m,     # (B, H_len)
    }


class TKGDataModule:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.train_ds: Optional[TKGEliteDataset] = None
        self.valid_ds: Optional[TKGEliteDataset] = None
        self.test_ds:  Optional[TKGEliteDataset] = None

    def setup(self):
        kw = dict(
            data_dir       = self.cfg.data_dir,
            dataset        = self.cfg.dataset,
            num_paths      = self.cfg.num_paths,
            max_path_len   = self.cfg.max_path_len,
            num_negative   = self.cfg.num_negative,
            neg_mode       = "mixed",
            max_history    = self.cfg.max_history,
            use_reciprocal = self.cfg.use_reciprocal,
        )
        self.train_ds = TKGEliteDataset(split="train", **kw)
        self.valid_ds = TKGEliteDataset(split="valid", **kw)
        self.test_ds  = TKGEliteDataset(split="test",  **kw)

        self.cfg.num_entities  = self.train_ds.num_entities
        self.cfg.num_relations = self.train_ds.num_relations
        self.cfg.num_times     = max(
            self.train_ds.num_times,
            self.valid_ds.num_times,
            self.test_ds.num_times, 1
        )

    def _make_balanced_sampler(self, ds: TKGEliteDataset) -> WeightedRandomSampler:
        """
        Relation-balanced sampler: har bir relatsiya bir xil ehtimolda tanlanadi.
        WIKI (rel=1 = 45%) va YAGO (rel=8 = 30%) uchun zarur — aksi holda
        dominant relatsiya loss ni boshqarib ketadi va boshqalari yomonlashadi.
        """
        relations = [r for _, r, _, _ in ds.quads]
        freq      = Counter(relations)
        weights   = [1.0 / freq[r] for r in relations]
        return WeightedRandomSampler(weights, len(weights), replacement=True)

    def _loader(self, ds: TKGEliteDataset, shuffle: bool) -> DataLoader:
        use_balanced = (
            shuffle
            and self.cfg.dataset in ("WIKI", "YAGO", "YAGOs")
        )
        sampler = self._make_balanced_sampler(ds) if use_balanced else None
        return DataLoader(
            ds,
            batch_size  = self.cfg.batch_size,
            shuffle     = (shuffle and sampler is None),
            sampler     = sampler,
            num_workers = self.cfg.num_workers,
            collate_fn  = collate_fn,
            pin_memory  = True,
            drop_last   = shuffle,
        )

    def train_loader(self): return self._loader(self.train_ds, True)
    def valid_loader(self): return self._loader(self.valid_ds, False)
    def test_loader(self):  return self._loader(self.test_ds,  False)
