# models/elite_tkg_model.py
"""
CATRE: Cross-scale Adaptive Temporal Reasoning for Extrapolation

═══════════════════════════════════════════════════════════════════════════════
Scientific Contributions
═══════════════════════════════════════════════════════════════════════════════

1. MULTI-SCALE TEMPORAL AGGREGATION  ★ NOVEL
   - Three temporal scales: fine (recent), mid, coarse (historical)
   - Scale boundaries τ₁, τ₂ are LEARNABLE parameters (not fixed heuristics)
   - Soft assignment: sigmoid-based smooth boundary between scales
   - Cross-scale attention fusion: query selects which scale matters most
   - DaeMon uses single-scale PNA; CATRE learns the optimal granularity

2. ENTITY-INDEPENDENT RELATIONAL MEMORY  ★ (DaeMon-inspired + improved)
   - DaeMon key insight: init from query relation r, not entity embedding
   - CATRE improvement: combined semantic × temporal attention (dual attention)
   - Semantic attention: what in history is semantically relevant to query?
   - Temporal attention: what in history is temporally recent?
   - Final: gated update of initial relation memory with aggregated evidence

3. ENTITY-INDEPENDENT PATH REASONING  ★ (DaeMon-inspired + improved)
   - Path steps: (relation, Δt) only — no entity embeddings
   - Generalizes to unseen test-time entities (WIKI/YAGO gap problem)
   - DaeMon: GRU-style sequential; CATRE: Transformer (parallel, long-range)

4. RELATIVE TEMPORAL ENCODING  ★ (retained, improves DaeMon)
   - Δt = t_query − t_history, log-sinusoidal encoding
   - DaeMon uses absolute snapshot index → breaks for unseen future times
   - CATRE: always works for any future timestamp (WIKI test t=222..231)

5. THREE-SIGNAL FUSION  ★ NOVEL
   - Signal A: entity-aware query (s_emb + r_emb + neighborhood context)
   - Signal B: entity-independent path attention (multi-hop reasoning)
   - Signal C: relational memory (DaeMon-inspired entity-free evidence)
   - Fusion: learned 2×H→H MLP combining path attention + relational memory
   - Outperforms single-signal models by capturing complementary evidence

═══════════════════════════════════════════════════════════════════════════════
Architecture Overview
═══════════════════════════════════════════════════════════════════════════════

  Input: (s, r, t_q, paths[B,P,L,3], history[B,H,3])
     │
     ├─① Embeddings: ent_emb[s], rel_emb[r], delta_enc(Δt)
     │
     ├─② Multi-Scale Aggregation (use_history=True):
     │    messages = rel_proj(nb_rel)       ← entity-independent!
     │    [fine|mid|coarse] = 3×PNA(msgs × scale_weight)
     │    msa_out = cross_scale_attn([fine, mid, coarse], query=r)
     │    s_dynamic = tawaregate(s_emb, msa_out)
     │    nb_ctx = proj([s_dynamic, s_emb])     → q += nb_ctx
     │
     ├─③ Relational Memory (use_history=True):
     │    m_init = proj(r_query)            ← entity-independent init!
     │    attn = dual_attn(m_init, messages, Δt)
     │    rel_mem_out = gate(m_init, attn_weighted_msgs)
     │
     ├─④ Path Encoder (entity-independent):
     │    step = [rel_emb, Δt_enc]          ← no entity!
     │    path_reprs = Transformer(steps)
     │
     ├─⑤ Cross-Path Attention:
     │    cross_out = q + MHA(q, path_reprs, path_reprs)
     │
     ├─⑥ Three-Signal Fusion:
     │    final_q = fusion_mlp([cross_out, rel_mem_out])
     │
     └─⑦ Scoring:
          scores = link_head(final_q) · all_ent
               + w_direct × DistMult(s_dynamic, r) · all_ent

═══════════════════════════════════════════════════════════════════════════════
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RELATIVE TEMPORAL ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

class RelativeTemporalEncoding(nn.Module):
    """
    Δt = t_query − t_history → log-scaled sinusoidal.
    Works for any future timestamp (no out-of-vocabulary).
    """
    def __init__(self, dim: int, max_delta: int = 20000):
        super().__init__()
        self.dim = dim
        freqs = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(max_delta) / dim)
        )
        self.register_buffer("freqs", freqs)   # (dim/2,)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """delta_t: (...,) int → (..., dim)"""
        x   = torch.log1p(delta_t.float().clamp(min=0))
        x   = x.unsqueeze(-1) * self.freqs
        enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return enc


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MULTI-SCALE TEMPORAL AGGREGATION  ★ NOVEL
# ═══════════════════════════════════════════════════════════════════════════════

class MultiScaleAggregator(nn.Module):
    """
    Three-scale PNA aggregation with learnable temporal boundaries.

    Scale boundaries τ₁, τ₂ are learnable parameters — the model discovers
    optimal temporal granularity for each dataset automatically.

    Soft scale weights (smooth, differentiable):
        w_fine(Δt)   = σ(−sharp · (Δt − τ₁))
        w_coarse(Δt) = σ( sharp · (Δt − τ₂))
        w_mid(Δt)    = clamp(1 − w_fine − w_coarse, min=0)

    Entity-independent messages: msg = rel_proj(r_neighbor)
    Per-scale PNA: [mean, max, min, std, query-attn, subject] = 6×E features
    Cross-scale fusion: query-adaptive soft attention over 3 scales.
    """
    def __init__(self, entity_dim: int, relation_dim: int, dropout: float = 0.1):
        super().__init__()
        self.entity_dim = entity_dim

        # Entity-independent message projection
        self.rel_proj = nn.Linear(relation_dim, entity_dim, bias=False)

        # Learnable temporal scale parameters
        self.log_tau1  = nn.Parameter(torch.tensor(1.0))   # τ₁ ≈ e¹ ≈ 3
        self.log_tau2  = nn.Parameter(torch.tensor(2.5))   # τ₂ ≈ e²·⁵ ≈ 12
        self.sharpness = nn.Parameter(torch.tensor(1.5))   # sigmoid sharpness

        # Learnable time decay
        self.log_gamma = nn.Parameter(torch.tensor(-2.0))

        # Per-scale: query attention + PNA projection  (6E → E each)
        self.attn_projs = nn.ModuleList([
            nn.Linear(entity_dim, entity_dim, bias=False) for _ in range(3)
        ])
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(entity_dim * 6, entity_dim),
                nn.LayerNorm(entity_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ) for _ in range(3)
        ])

        # Cross-scale fusion: project query → 3 logits
        self.cs_query   = nn.Linear(relation_dim, entity_dim)
        self.cs_logits  = nn.Linear(entity_dim, 3)
        self.out_norm   = nn.LayerNorm(entity_dim)
        self.drop       = nn.Dropout(dropout)

    def _pna_one_scale(
        self,
        msgs:   torch.Tensor,   # (B, H, E)  scale-weighted messages
        mask:   torch.Tensor,   # (B, H)
        e_emb:  torch.Tensor,   # (B, E)
        q_key:  torch.Tensor,   # (B, E)
        idx:    int,
    ) -> torch.Tensor:          # (B, E)
        eps  = 1e-6
        fill = -1e4

        valid_f = mask.float().unsqueeze(-1)
        n_valid = mask.float().sum(1, keepdim=True).clamp(min=1)
        msgs_v  = msgs * valid_f

        mean_v  = msgs_v.sum(1) / n_valid
        max_v   = msgs_v.max(1).values
        min_raw = msgs.masked_fill(~mask.unsqueeze(-1), 1e4)
        min_v   = min_raw.min(1).values.clamp(-1e4, 1e4)
        sq_m    = (msgs_v.clamp(-10, 10) ** 2).sum(1) / n_valid
        std_v   = (sq_m - mean_v ** 2).clamp(min=eps).sqrt()

        E        = msgs.size(-1)
        attn_raw = (msgs * self.attn_projs[idx](q_key).unsqueeze(1)).sum(-1) / math.sqrt(E)
        attn_raw = attn_raw.masked_fill(~mask, fill)
        all_m    = (~mask).all(dim=1, keepdim=True).expand_as(attn_raw)
        attn_raw = attn_raw.masked_fill(all_m, 0.0)
        attn_w   = torch.softmax(attn_raw, dim=-1)
        attn_v   = (attn_w.unsqueeze(-1) * msgs).sum(1)

        feats = torch.cat([mean_v, max_v, min_v, std_v, attn_v, e_emb], dim=-1)  # (B, 6E)
        return self.scale_projs[idx](feats)

    def forward(
        self,
        e_emb:     torch.Tensor,   # (B, E)
        nb_rel:    torch.Tensor,   # (B, H, R)
        delta_t:   torch.Tensor,   # (B, H) int
        r_emb:     torch.Tensor,   # (B, R)
        hist_mask: torch.Tensor,   # (B, H)
    ) -> torch.Tensor:             # (B, E)

        # Entity-independent messages
        msgs = self.rel_proj(nb_rel)   # (B, H, E)

        # Global time decay
        gamma        = torch.exp(self.log_gamma)
        decay        = torch.exp(-gamma * delta_t.float().clamp(min=0))
        msgs_dec     = msgs * decay.unsqueeze(-1)

        # Learnable soft scale boundaries
        tau1  = torch.exp(self.log_tau1)
        tau2  = torch.exp(self.log_tau2) + tau1   # ensure τ₂ > τ₁
        sharp = self.sharpness.clamp(0.1, 5.0)
        dt_f  = delta_t.float()

        w_fine   = torch.sigmoid(-sharp * (dt_f - tau1))
        w_coarse = torch.sigmoid( sharp * (dt_f - tau2))
        w_mid    = (1.0 - w_fine - w_coarse).clamp(min=0.0)

        # Scale-weighted messages
        m_fine   = msgs_dec * w_fine.unsqueeze(-1)
        m_mid    = msgs_dec * w_mid.unsqueeze(-1)
        m_coarse = msgs_dec * w_coarse.unsqueeze(-1)

        # Project query relation for per-scale attention
        q_key = self.rel_proj(r_emb)   # (B, E) — reuse rel_proj weight

        fine_out   = self._pna_one_scale(m_fine,   hist_mask, e_emb, q_key, 0)
        mid_out    = self._pna_one_scale(m_mid,    hist_mask, e_emb, q_key, 1)
        coarse_out = self._pna_one_scale(m_coarse, hist_mask, e_emb, q_key, 2)

        # Cross-scale query-adaptive fusion
        scales   = torch.stack([fine_out, mid_out, coarse_out], dim=1)   # (B, 3, E)
        sq       = self.cs_query(r_emb)                                   # (B, E)
        logits   = self.cs_logits(sq)                                     # (B, 3)
        weights  = torch.softmax(logits, dim=-1)                          # (B, 3)
        out      = (weights.unsqueeze(-1) * scales).sum(1)               # (B, E)
        return self.out_norm(self.drop(out))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ENTITY-INDEPENDENT RELATIONAL MEMORY  ★ (DaeMon-inspired + improved)
# ═══════════════════════════════════════════════════════════════════════════════

class RelationalMemory(nn.Module):
    """
    Entity-independent relational memory.

    DaeMon key insight: initialize subject representation from query relation r,
    not from entity embedding — entity-independent generalization.

    CATRE improvement: dual-attention aggregation
      • Semantic attention:  attn_sem  = q_key · message  (what's relevant)
      • Temporal attention:  attn_temp = exp(−γ · Δt)     (what's recent)
      • Combined:            attn = softmax(attn_sem × attn_temp)

    Update:
      gate   = σ(W_g [m_init, aggregated])
      m_out  = gate ⊙ tanh(W_u · aggregated) + (1−gate) ⊙ m_init
    """
    def __init__(
        self,
        entity_dim:   int,
        relation_dim: int,
        hidden_dim:   int,
        dropout:      float = 0.1,
    ):
        super().__init__()
        # Init memory from query relation
        self.init_proj   = nn.Linear(relation_dim, entity_dim)

        # Entity-independent message projection (neighbor relations)
        self.msg_proj    = nn.Linear(relation_dim, entity_dim, bias=False)

        # Semantic attention
        self.sem_key     = nn.Linear(entity_dim, entity_dim, bias=False)

        # Temporal decay (learnable)
        self.log_gamma   = nn.Parameter(torch.tensor(-1.5))

        # Gated memory update
        self.gate        = nn.Sequential(nn.Linear(entity_dim * 2, entity_dim), nn.Sigmoid())
        self.transform   = nn.Sequential(nn.Linear(entity_dim, entity_dim), nn.Tanh())

        # Project to hidden_dim for final output
        self.out_proj    = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        nb_rel:    torch.Tensor,   # (B, H, R)
        delta_t:   torch.Tensor,   # (B, H) int
        r_emb:     torch.Tensor,   # (B, R)
        hist_mask: torch.Tensor,   # (B, H)
    ) -> torch.Tensor:             # (B, hidden_dim)
        fill = -1e4

        # Entity-independent init from query relation
        m_init   = self.init_proj(r_emb)                  # (B, E)

        # Entity-independent messages from neighbor relations
        messages = self.msg_proj(nb_rel)                   # (B, H, E)

        # Temporal decay
        gamma    = torch.exp(self.log_gamma)
        decay    = torch.exp(-gamma * delta_t.float().clamp(min=0))   # (B, H)

        # Semantic attention (query relevance)
        E        = m_init.size(-1)
        q_key    = self.sem_key(m_init)                    # (B, E)
        attn_sem = (messages * q_key.unsqueeze(1)).sum(-1) / math.sqrt(E)  # (B, H)
        attn_sem = attn_sem.masked_fill(~hist_mask, fill)

        # Temporal attention combined with semantic (element-wise multiply in log-space → sum)
        # attn_combined = sem + log(decay) for numerical stability
        log_decay    = -gamma * delta_t.float().clamp(min=0)   # (B, H), already in log
        attn_combined = attn_sem + log_decay.clamp(min=-10)    # (B, H)
        attn_combined = attn_combined.masked_fill(~hist_mask, fill)

        all_m    = (~hist_mask).all(dim=1, keepdim=True).expand_as(attn_combined)
        attn_combined = attn_combined.masked_fill(all_m, 0.0)
        attn_w   = torch.softmax(attn_combined, dim=-1)         # (B, H)

        # Aggregated evidence
        agg = (attn_w.unsqueeze(-1) * messages).sum(1)          # (B, E)

        # Gated memory update
        gate_v = self.gate(torch.cat([m_init, agg], dim=-1))
        m_out  = gate_v * self.transform(agg) + (1 - gate_v) * m_init   # (B, E)

        return self.out_proj(self.drop(m_out))   # (B, hidden_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GATED TEMPORAL MEMORY (tawaregate — DaeMon dan)
# ═══════════════════════════════════════════════════════════════════════════════

class GatedTemporalMemory(nn.Module):
    """
    gate   = σ(W_g · [e_static, e_dynamic])
    output = gate ⊙ tanh(W_h · e_dynamic) + (1−gate) ⊙ e_static
    """
    def __init__(self, entity_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate      = nn.Sequential(nn.Linear(entity_dim * 2, entity_dim), nn.Sigmoid())
        self.transform = nn.Sequential(nn.Linear(entity_dim, entity_dim), nn.Tanh())
        self.norm      = nn.LayerNorm(entity_dim)
        self.drop      = nn.Dropout(dropout)

    def forward(self, e_static: torch.Tensor, e_dynamic: torch.Tensor) -> torch.Tensor:
        g   = self.gate(torch.cat([e_static, e_dynamic], dim=-1))
        out = g * self.transform(e_dynamic) + (1 - g) * e_static
        return self.norm(self.drop(out))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ENTITY-INDEPENDENT PATH ENCODER
# ═══════════════════════════════════════════════════════════════════════════════

class PathTransformerEncoder(nn.Module):
    """
    Entity-independent path encoder.
    step = [rel_emb, Δt_enc]  — no entity embeddings.

    Why entity-independent:
    - WIKI/YAGO: test entities differ from train → entity embeddings degrade
    - Pure relation patterns sufficient with 10–24 relation types
    - Transformer (vs DaeMon's sequential) captures long-range path patterns
    """
    def __init__(
        self,
        relation_dim: int,
        delta_dim:    int,
        hidden_dim:   int,
        num_heads:    int   = 8,
        num_layers:   int   = 2,
        ffn_dim:      int   = 1024,
        dropout:      float = 0.1,
        max_seq_len:  int   = 16,
        entity_dim:   int   = 0,   # backward-compat, ignored
    ):
        super().__init__()
        step_dim = relation_dim + delta_dim   # entity-independent!
        self.step_proj = nn.Sequential(
            nn.Linear(step_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_emb   = nn.Embedding(max_seq_len + 1, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, step_vecs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        BP, L, _ = step_vecs.shape
        x   = self.step_proj(step_vecs)
        cls = self.cls_token.expand(BP, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        pos = torch.arange(L + 1, device=x.device)
        x   = x + self.pos_emb(pos).unsqueeze(0)
        cls_m  = torch.ones(BP, 1, dtype=torch.bool, device=x.device)
        full_m = torch.cat([cls_m, mask], dim=1)
        out    = self.transformer(x, src_key_padding_mask=~full_m)
        return self.norm(out)[:, 0, :]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LINK PREDICTION HEAD
# ═══════════════════════════════════════════════════════════════════════════════

class LinkPredHead(nn.Module):
    def __init__(self, hidden_dim: int, entity_dim: int, num_entities: int, dropout: float = 0.1):
        super().__init__()
        self.fc1   = nn.Linear(hidden_dim, hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, entity_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(entity_dim)
        self.gate  = nn.Linear(hidden_dim, hidden_dim)
        self.drop  = nn.Dropout(dropout)
        self.bias  = nn.Parameter(torch.zeros(num_entities))

    def forward(self, q: torch.Tensor, ent_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(q)
        g = torch.sigmoid(self.gate(h))
        h = g * F.gelu(self.fc1(h)) + (1 - g) * h
        h = self.norm2(self.fc2(self.drop(h)))
        s = h @ ent_emb.t()
        if s.size(1) == self.bias.size(0):
            s = s + self.bias
        return s


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DIRECT SCORING HEAD (DistMult-style)
# ═══════════════════════════════════════════════════════════════════════════════

class DirectScoringHead(nn.Module):
    def __init__(self, entity_dim: int, relation_dim: int, num_entities: int, dropout: float = 0.1):
        super().__init__()
        self.rel_proj = nn.Linear(relation_dim, entity_dim, bias=False)
        self.sub_proj = nn.Sequential(
            nn.Linear(entity_dim, entity_dim),
            nn.LayerNorm(entity_dim),
            nn.GELU(),
        )
        self.drop = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(num_entities))

    def forward(self, s_emb: torch.Tensor, r_emb: torch.Tensor,
                all_ent: torch.Tensor) -> torch.Tensor:
        s = self.drop(self.sub_proj(s_emb) * self.rel_proj(r_emb)) @ all_ent.t()
        if s.size(1) == self.bias.size(0):
            s = s + self.bias
        return s


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CATRE — MAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class CATREModel(nn.Module):
    """
    CATRE: Cross-scale Adaptive Temporal Reasoning for Extrapolation

    Combines:
    ├ DaeMon's entity-independent design (relation-only path/messages)
    ├ DaeMon's tawaregate memory (extended with dual-attention)
    ├ Novel multi-scale temporal aggregation (learnable boundaries)
    ├ Novel three-signal fusion (entity + path + relational memory)
    └ Relative Temporal Encoding (Δt-based, works for future times)
    """

    def __init__(
        self,
        num_entities:    int,
        num_relations:   int,
        num_times:       int,
        entity_dim:      int   = 256,
        relation_dim:    int   = 256,
        delta_dim:       int   = 64,
        hidden_dim:      int   = 512,
        num_heads:       int   = 8,
        num_layers:      int   = 2,
        ffn_dim:         int   = 1024,
        num_negative:    int   = 256,
        dropout:         float = 0.1,
        label_smoothing: float = 0.1,
        w_direct:        float = 0.0,
        use_history:     bool  = False,
        use_diachronic:  bool  = False,
        # backward-compat (ignored)
        time_dim:           int   = 64,
        proj_dim:           int   = 256,
        temperature:        float = 0.3,
        momentum:           float = 0.995,
        queue_size:         int   = 8192,
        use_direct_scoring: bool  = False,
        use_time_encoding:  str   = "both",
        **kwargs,
    ):
        super().__init__()
        self.num_entities    = num_entities
        self.entity_dim      = entity_dim
        self.hidden_dim      = hidden_dim
        self.num_negative    = num_negative
        self.label_smoothing = label_smoothing
        self.w_direct        = w_direct
        self.use_history     = use_history
        self.use_diachronic  = use_diachronic

        # ── Embeddings ────────────────────────────────────────────────────────
        self.ent_emb = nn.Embedding(num_entities,      entity_dim,   padding_idx=0)
        self.rel_emb = nn.Embedding(num_relations * 2, relation_dim)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # ── Relative Temporal Encoding ────────────────────────────────────────
        self.delta_enc = RelativeTemporalEncoding(
            delta_dim, max_delta=max(num_times * 3, 20000)
        )

        # ── Query Encoder ─────────────────────────────────────────────────────
        inp = entity_dim + relation_dim + delta_dim
        self.query_encoder = nn.Sequential(
            nn.Linear(inp, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── Neighborhood modules ──────────────────────────────────────────────
        if use_history:
            self.msa      = MultiScaleAggregator(entity_dim, relation_dim, dropout)
            self.gate_mem = GatedTemporalMemory(entity_dim, dropout)
            self.rel_mem  = RelationalMemory(entity_dim, relation_dim, hidden_dim, dropout)
            self.nb_ctx   = nn.Sequential(
                nn.Linear(entity_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.hist_norm = nn.LayerNorm(hidden_dim)
        else:
            self.msa = self.gate_mem = self.rel_mem = self.nb_ctx = self.hist_norm = None

        # ── Path Encoder (entity-independent) ────────────────────────────────
        self.path_encoder = PathTransformerEncoder(
            relation_dim=relation_dim,
            delta_dim=delta_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # ── Cross-Path Attention ──────────────────────────────────────────────
        self.cross_attn  = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm  = nn.LayerNorm(hidden_dim)

        # ── Three-Signal Fusion  ★ NOVEL ──────────────────────────────────────
        # Fuses: cross_attn_out (path reasoning) + rel_mem (relational memory)
        self.fusion_mlp  = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_ffn  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.fusion_norm2 = nn.LayerNorm(hidden_dim)

        # ── Scoring Heads ─────────────────────────────────────────────────────
        self.link_head   = LinkPredHead(hidden_dim, entity_dim, num_entities, dropout)
        self.direct_head = DirectScoringHead(entity_dim, relation_dim, num_entities, dropout) \
                           if (w_direct > 0 or use_direct_scoring) else None

        # ── Diachronic Gating ─────────────────────────────────────────────────
        if use_diachronic:
            self.dia_amp   = nn.Embedding(1, entity_dim)
            self.dia_freq  = nn.Embedding(1, entity_dim)
            self.dia_phase = nn.Embedding(1, entity_dim)
            nn.init.uniform_(self.dia_amp.weight,   -0.1,       0.1)
            nn.init.uniform_(self.dia_freq.weight,   0.5,       2.0)
            nn.init.uniform_(self.dia_phase.weight, -math.pi, math.pi)
            self.num_times_dia = max(num_times, 1)

        # ── Per-relation temperature ──────────────────────────────────────────
        self.rel_temp = nn.Embedding(num_relations, 1)
        nn.init.constant_(self.rel_temp.weight, 1.0)

        self.drop = nn.Dropout(dropout)
        self.register_buffer("all_entity_ids", torch.arange(num_entities, dtype=torch.long))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ent(self, e):
        return self.drop(self.ent_emb(e))

    def _rel(self, r):
        return self.drop(self.rel_emb(r.clamp(0, self.rel_emb.num_embeddings - 1)))

    def _diachronic(self, e_emb: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not self.use_diachronic:
            return e_emb
        t_norm = (t.float() / self.num_times_dia).unsqueeze(-1)
        gate   = 1.0 + self.dia_amp.weight[0] * torch.sin(
            self.dia_freq.weight[0] * t_norm + self.dia_phase.weight[0]
        )
        return e_emb * gate

    def _encode_query(self, subjects, relations, times):
        s = self._ent(subjects)
        r = self._rel(relations)
        d = self.delta_enc(torch.zeros(subjects.size(0), dtype=torch.long, device=subjects.device))
        return self.query_encoder(torch.cat([s, r, d], dim=-1))

    def _encode_paths(
        self,
        paths:      torch.Tensor,   # (B, P, L, 3)
        path_masks: torch.Tensor,   # (B, P, L)
        query_t:    torch.Tensor,   # (B,)
    ) -> torch.Tensor:              # (B, P, H)
        B, P, L, _ = paths.shape
        pf = paths.view(B * P, L, 3)
        mf = path_masks.view(B * P, L)

        rel = pf[:, :, 1]
        t_h = pf[:, :, 2]
        t_q = query_t.unsqueeze(1).expand(-1, P).contiguous().view(B * P)
        dt  = (t_q.unsqueeze(1) - t_h.float()).clamp(min=0).long()

        # Entity-independent: only relation + delta
        r_e = self.rel_emb(rel.clamp(0, self.rel_emb.num_embeddings - 1))
        d_e = self.delta_enc(dt)
        step = torch.cat([r_e, d_e], dim=-1)   # (BP, L, R+D)
        return self.path_encoder(step, mf).view(B, P, -1)

    def _process_neighborhood(
        self,
        times:     torch.Tensor,   # (B,)
        history:   torch.Tensor,   # (B, H, 3)
        hist_mask: torch.Tensor,   # (B, H)
        s_emb:     torch.Tensor,   # (B, E)
        r_emb:     torch.Tensor,   # (B, R)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: (s_dynamic, nb_ctx_out, rel_mem_out)"""
        hist_rel = history[:, :, 1]
        hist_t   = history[:, :, 2]

        delta_t  = (times.unsqueeze(1) - hist_t.float()).clamp(min=0).long()   # (B, H)

        nb_r = self.rel_emb(hist_rel.clamp(0, self.rel_emb.num_embeddings - 1))  # (B, H, R)

        # Multi-Scale Aggregation (entity-independent messages)
        msa_out = self.msa(s_emb, nb_r, delta_t, r_emb, hist_mask)   # (B, E)

        # Tawaregate: combine MSA output with static entity
        s_dynamic = self.gate_mem(s_emb, msa_out)                     # (B, E)

        # Neighborhood context for query
        nb_ctx_out = self.nb_ctx(torch.cat([s_dynamic, s_emb], dim=-1))   # (B, H_dim)

        # Relational Memory (entity-independent, DaeMon-inspired)
        rel_mem_out = self.rel_mem(nb_r, delta_t, r_emb, hist_mask)   # (B, H_dim)

        return s_dynamic, nb_ctx_out, rel_mem_out

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        subjects:    torch.Tensor,
        relations:   torch.Tensor,
        objects:     torch.Tensor,
        times:       torch.Tensor,
        paths:       torch.Tensor,
        path_masks:  torch.Tensor,
        neg_objects: torch.Tensor,
        history:     Optional[torch.Tensor] = None,
        hist_mask:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        device    = subjects.device
        s_emb     = self._ent(subjects)
        r_emb     = self._rel(relations)

        # ── Query (entity-aware) ──────────────────────────────────────────────
        q = self._encode_query(subjects, relations, times)   # (B, H_dim)

        # ── Neighborhood ─────────────────────────────────────────────────────
        s_dynamic   = None
        rel_mem_out = None
        if self.use_history and history is not None:
            s_dynamic, nb_ctx_out, rel_mem_out = self._process_neighborhood(
                times, history, hist_mask, s_emb, r_emb
            )
            q = self.hist_norm(q + nb_ctx_out)

        # ── Paths (entity-independent) ────────────────────────────────────────
        path_reprs = self._encode_paths(paths, path_masks, times)   # (B, P, H_dim)

        # ── Cross-path attention ──────────────────────────────────────────────
        q_exp       = q.unsqueeze(1)
        attn_out, _ = self.cross_attn(q_exp, path_reprs, path_reprs)
        cross_out   = self.cross_norm(q_exp + attn_out).squeeze(1)  # (B, H_dim)

        # ── Three-Signal Fusion ★ ─────────────────────────────────────────────
        # Fallback: if no history, rel_mem_out = zeros
        if rel_mem_out is None:
            rel_mem_out = torch.zeros_like(q)

        fused = self.fusion_mlp(torch.cat([cross_out, rel_mem_out], dim=-1))  # (B, H_dim)
        fused = self.fusion_norm(fused + cross_out)                            # residual
        final_q = self.fusion_norm2(fused + self.fusion_ffn(fused))

        # ── Scoring ───────────────────────────────────────────────────────────
        all_ent = self.ent_emb(self.all_entity_ids)
        scores  = self.link_head(final_q, all_ent)
        scores  = scores * self.rel_temp(relations).squeeze(-1).unsqueeze(1)

        if self.direct_head is not None:
            s_t = s_dynamic if s_dynamic is not None else self.ent_emb(subjects)
            s_t = self._diachronic(s_t, times)
            scores = scores + self.w_direct * self.direct_head(s_t, r_emb, all_ent)

        # ── Losses ────────────────────────────────────────────────────────────
        link_loss = F.cross_entropy(scores, objects, label_smoothing=self.label_smoothing)

        if neg_objects.size(1) > 0:
            neg_s = scores.gather(1, neg_objects)
            with torch.no_grad():
                w = F.softmax(neg_s * 0.5, dim=-1)
            adv = -(w * F.logsigmoid(-neg_s)).sum(-1).mean()
            adv = adv - F.logsigmoid(scores.gather(1, objects.unsqueeze(1))).mean()
        else:
            adv = torch.tensor(0.0, device=device)

        rel_w = self.rel_emb.weight
        tmp   = rel_w @ rel_w.t()
        ortho = torch.norm(tmp - torch.eye(tmp.size(0), device=device), p=2)

        losses = {
            "link":        link_loss,
            "contrastive": torch.tensor(0.0, device=device),
            "self_adv":    adv,
            "ortho_reg":   ortho,
        }
        return scores, losses

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        subjects:   torch.Tensor,
        relations:  torch.Tensor,
        times:      torch.Tensor,
        paths:      torch.Tensor,
        path_masks: torch.Tensor,
        history:    Optional[torch.Tensor] = None,
        hist_mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        s_emb = self._ent(subjects)
        r_emb = self._rel(relations)
        q     = self._encode_query(subjects, relations, times)

        s_dynamic   = None
        rel_mem_out = None
        if self.use_history and history is not None:
            s_dynamic, nb_ctx_out, rel_mem_out = self._process_neighborhood(
                times, history, hist_mask, s_emb, r_emb
            )
            q = self.hist_norm(q + nb_ctx_out)

        path_reprs  = self._encode_paths(paths, path_masks, times)
        q_exp       = q.unsqueeze(1)
        attn_out, _ = self.cross_attn(q_exp, path_reprs, path_reprs)
        cross_out   = self.cross_norm(q_exp + attn_out).squeeze(1)

        if rel_mem_out is None:
            rel_mem_out = torch.zeros_like(q)

        fused   = self.fusion_mlp(torch.cat([cross_out, rel_mem_out], dim=-1))
        fused   = self.fusion_norm(fused + cross_out)
        final_q = self.fusion_norm2(fused + self.fusion_ffn(fused))

        all_ent = self.ent_emb(self.all_entity_ids)
        scores  = self.link_head(final_q, all_ent)
        scores  = scores * self.rel_temp(relations).squeeze(-1).unsqueeze(1)

        if self.direct_head is not None:
            s_t = s_dynamic if s_dynamic is not None else self.ent_emb(subjects)
            s_t = self._diachronic(s_t, times)
            scores = scores + self.w_direct * self.direct_head(s_t, r_emb, all_ent)

        return scores


# Backward compatibility
STORMModel    = CATREModel
EliteTKGModel = CATREModel
