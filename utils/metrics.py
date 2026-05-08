# utils/metrics.py
from typing import Dict, List
import torch


def compute_ranks(
    scores: torch.Tensor,       # (N, E)
    targets: torch.Tensor,      # (N,)
    filter_mask: torch.Tensor,  # (N, E) bool
    filter_flag: bool = True,
) -> torch.Tensor:              # (N,) float ranks
    N, E = scores.shape
    device = scores.device
    idx = torch.arange(N, device=device)

    if filter_flag:
        tgt_sc = scores[idx, targets].clone()
        scores = scores.clone()
        scores[filter_mask] = float("-inf")
        scores[idx, targets] = tgt_sc

    tgt_sc = scores[idx, targets].unsqueeze(1)      # (N,1)
    ranks = (scores > tgt_sc).sum(dim=1).float() + 1  # (N,)
    return ranks


def ranks_to_metrics(
    ranks: torch.Tensor,
    hits_at_k: List[int] = [1, 3, 10],
) -> Dict[str, float]:
    out = {
        "MRR": (1.0 / ranks).mean().item(),
        "MR":  ranks.mean().item(),
    }
    for k in hits_at_k:
        out[f"Hits@{k}"] = (ranks <= k).float().mean().item()
    return out


def format_metrics(m: Dict[str, float]) -> str:
    skip = {"MR", "n"}
    return "  ".join(
        f"{k}: {v:.4f}" for k, v in m.items() if k not in skip
    )
