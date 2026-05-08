# models/elite_tkg_model.py
"""
STORM: Sequential Temporal Oracle with Relational Memory

═══════════════════════════════════════════════════════════════════════════════
Scientific Contributions
═══════════════════════════════════════════════════════════════════════════════

1. RELATIVE TEMPORAL ENCODING (RTE)
   - Muammo: mavjud modellar mutlaq vaqt indeksini ishlatadi → WIKI test
     t=222..231 train da ko'rilmagan, learned[222..231] random init = shovqin
   - Yechim: Δt = t_query − t_history, log-scaled sinusoidal encoding
   - Natija: har qanday kelajak vaqtni to'g'ri ifodalaydi; differensiable

2. CROSS-SNAPSHOT ATTENTION (CSA)
   - Muammo: DaeMon RNN-like ketma-ket ishlaydi → uzoq tarix siqiladi,
     gradient yo'qoladi, erta snapshot lar unutiladi
   - Yechim: barcha snapshot larni Transformer bilan attend qilish;
     time-bias: Δt ga qarab vaqt og'irligi
   - Natija: uzoq masofadagi temporal bog'liqliklarni ushlaydi

3. QUERY-ADAPTIVE PNA AGGREGATION
   - Muammo: DaeMon PNA og'irliklari har qanday query uchun bir xil
   - Yechim: aggregatsiya diqqat og'irliklari = f(relation_query, message)
   - Natija: har so'rov uchun eng mos qo'shnilarga e'tibor qaratadi

4. DUAL-PATH TEMPORAL REASONING
   - Yo'l A: Multi-hop path reasoning (Transformer encoder)
   - Yo'l B: Temporal neighborhood aggregation (PNA + CSA + tawaregate)
   - Birlashtirish: cross-attention → ikki yo'l adaptiv birlashadi
   - Natija: lokal (1-hop tarix) va struktural (ko'p-hop) naqshlarni ushlaydi

5. TAWAREGATE MEMORY (DaeMon dan olingan, kengaytirilgan)
   - gate = σ(W_g · [e_static, e_dynamic])
   - output = gate ⊙ e_dynamic + (1−gate) ⊙ e_static
   - Kengaytirish: Δt decay bilan og'irlantirilgan

═══════════════════════════════════════════════════════════════════════════════
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RELATIVE TEMPORAL ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

class RelativeTemporalEncoding(nn.Module):
    """
    Δt = t_query − t_history ni log-scaled sinusoidal bilan kodlaydi.

    Mutlaq vaqt indeksidan farqi:
    - Ko'rilmagan kelajak vaqtlar uchun ishlaydi (out-of-vocabulary yo'q)
    - Silliq va differensiable
    - Qisqa va uzoq vaqt masofalarini bir xil yaxshi ifodalaydi

    Encoding: sin/cos(log(1+Δt) × freq_i),  log-spaced frequencies
    """
    def __init__(self, dim: int, max_delta: int = 20000):
        super().__init__()
        self.dim = dim
        freqs = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(max_delta) / dim)
        )
        self.register_buffer("freqs", freqs)  # (dim/2,)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """delta_t: (...,) int → (..., dim)"""
        x   = torch.log1p(delta_t.float().clamp(min=0))  # (...,)
        x   = x.unsqueeze(-1) * self.freqs               # (..., dim/2)
        enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (..., dim)
        return enc


# ═══════════════════════════════════════════════════════════════════════════════
# 2. QUERY-ADAPTIVE PNA AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

class QueryAdaptivePNA(nn.Module):
    """
    Query-adaptive Principal Neighbourhood Aggregation.

    DaeMon PathAggNet dan farqlari:
    - Aggregatsiya diqqat og'irliklari query ga bog'liq
    - Δt asosida vaqt-decay og'irlash
    - PNA: [mean, max, min, std] × 3 degree scale = 12 aggregator + attn + subject = 14×E
    - Yangi: min va std aggregatorlari (DaeMon da yo'q)
    """
    def __init__(
        self,
        entity_dim:   int,
        relation_dim: int,
        delta_dim:    int,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.entity_dim = entity_dim

        # DistMult: relation projection
        self.rel_proj  = nn.Linear(relation_dim, entity_dim, bias=False)

        # Query-adaptive attention
        self.attn_proj = nn.Linear(entity_dim, entity_dim, bias=False)

        # Time-decay: learnable
        self.log_gamma = nn.Parameter(torch.tensor(-2.0))

        # PNA: 14 × entity_dim
        self.pna_proj = nn.Sequential(
            nn.Linear(entity_dim * 14, entity_dim * 2),
            nn.LayerNorm(entity_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(entity_dim * 2, entity_dim),
            nn.LayerNorm(entity_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        e_emb:      torch.Tensor,  # (B, E)
        nb_emb:     torch.Tensor,  # (B, H, E)
        nb_rel_emb: torch.Tensor,  # (B, H, R)
        delta_t:    torch.Tensor,  # (B, H) int
        query_emb:  torch.Tensor,  # (B, R)
        hist_mask:  torch.Tensor,  # (B, H)
    ) -> torch.Tensor:             # (B, E)
        B, H, E = nb_emb.shape
        eps  = 1e-6
        fill = -1e4  # fp16 safe

        # DistMult messages
        messages = nb_emb * self.rel_proj(nb_rel_emb)              # (B, H, E)

        # Time-decay
        gamma        = torch.exp(self.log_gamma)
        decay        = torch.exp(-gamma * delta_t.float().clamp(min=0))  # (B, H)
        messages_dec = messages * decay.unsqueeze(-1)               # (B, H, E)

        # Query-adaptive attention
        q_key      = self.attn_proj(query_emb)                      # (B, E)
        attn_raw   = (messages_dec * q_key.unsqueeze(1)).sum(-1) / math.sqrt(E)  # (B, H)
        attn_raw   = attn_raw.masked_fill(~hist_mask, fill)
        attn_w     = torch.softmax(attn_raw, dim=-1)                # (B, H)

        # Masked aggregators
        valid_f = hist_mask.float().unsqueeze(-1)                   # (B, H, 1)
        n_valid = hist_mask.float().sum(1, keepdim=True).clamp(min=1)
        msgs_v  = messages_dec * valid_f                            # (B, H, E)

        mean_v = msgs_v.sum(1) / n_valid                            # (B, E)
        max_v  = msgs_v.max(1).values                               # (B, E)

        msgs_min = messages_dec.masked_fill(~hist_mask.unsqueeze(-1), 1e4)
        min_v    = msgs_min.min(1).values.clamp(-1e4, 1e4)          # (B, E)

        sq_mean = (msgs_v.clamp(-10, 10) ** 2).sum(1) / n_valid
        std_v   = (sq_mean - mean_v ** 2).clamp(min=eps).sqrt()     # (B, E)

        attn_v = (attn_w.unsqueeze(-1) * messages_dec).sum(1)       # (B, E)

        # Degree scaling (log-scale based on n_valid)
        deg    = n_valid.squeeze(1)                                  # (B,)
        log_d  = torch.log(deg.clamp(min=1)).unsqueeze(-1)          # (B, 1)
        mean_d = log_d.mean().clamp(min=1)
        s1 = torch.ones_like(mean_v)
        sd = (log_d / mean_d).expand_as(mean_v)
        si = (mean_d / log_d.clamp(min=0.01)).expand_as(mean_v)

        feats = torch.cat([
            mean_v * s1, mean_v * sd, mean_v * si,
            max_v  * s1, max_v  * sd, max_v  * si,
            min_v  * s1, min_v  * sd, min_v  * si,
            std_v  * s1, std_v  * sd, std_v  * si,
            attn_v,
            e_emb,
        ], dim=-1)  # (B, 14E)

        return self.pna_proj(feats)  # (B, E)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CROSS-SNAPSHOT ATTENTION (CSA) — yangi ilmiy hissa
# ═══════════════════════════════════════════════════════════════════════════════

class CrossSnapshotAttention(nn.Module):
    """
    Transformer-style diqqat temporal tarix bo'ylab.

    DaeMon dan farqi:
    - DaeMon: ketma-ket RNN-like → erta tarix siqiladi
    - CSA: barcha tarixiy faktlarni bir vaqtda attend qiladi
    - Time-bias: Δt ni K larga qo'shish → vaqtga sezgir diqqat
    - Gradient yo'qolish muammosi yo'q (direct attention)

    Query:  (relation, entity_static) — nima qidirilmoqda
    Keys:   messages + Δt encoding — tarixiy faktlar
    Values: neighbor embeddings
    """
    def __init__(
        self,
        entity_dim:   int,
        relation_dim: int,
        delta_dim:    int,
        num_heads:    int   = 4,
        dropout:      float = 0.1,
    ):
        super().__init__()
        assert entity_dim % num_heads == 0
        self.entity_dim = entity_dim
        self.num_heads  = num_heads
        self.head_dim   = entity_dim // num_heads
        self.scale      = math.sqrt(self.head_dim)

        self.key_proj   = nn.Linear(entity_dim + delta_dim, entity_dim)
        self.val_proj   = nn.Linear(entity_dim, entity_dim)
        self.query_proj = nn.Linear(relation_dim + entity_dim, entity_dim)
        self.out_proj   = nn.Sequential(
            nn.Linear(entity_dim, entity_dim),
            nn.LayerNorm(entity_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        messages:   torch.Tensor,  # (B, H, E)
        nb_emb:     torch.Tensor,  # (B, H, E)
        delta_enc:  torch.Tensor,  # (B, H, D)
        query_emb:  torch.Tensor,  # (B, R)
        e_static:   torch.Tensor,  # (B, E)
        hist_mask:  torch.Tensor,  # (B, H)
    ) -> torch.Tensor:             # (B, E)
        B, H, E = messages.shape
        fill = -1e4  # fp16 max ~65504, -1e9 overflow qiladi

        # K: message + time encoding
        keys   = self.key_proj(torch.cat([messages, delta_enc], dim=-1))  # (B, H, E)
        values = self.val_proj(nb_emb)                                     # (B, H, E)

        # Q: relation + static entity
        query  = self.query_proj(
            torch.cat([query_emb, e_static], dim=-1)
        ).unsqueeze(1)  # (B, 1, E)

        def split_heads(x):  # (B, S, E) → (B, nh, S, hd)
            return x.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(query)   # (B, nh, 1, hd)
        K = split_heads(keys)    # (B, nh, H, hd)
        V = split_heads(values)  # (B, nh, H, hd)

        attn = (Q @ K.transpose(-2, -1)) / self.scale       # (B, nh, 1, H)
        mask = ~hist_mask.unsqueeze(1).unsqueeze(2)          # (B, 1, 1, H)
        attn = attn.masked_fill(mask, fill)

        # Check all-masked → avoid nan
        all_masked = (~hist_mask).all(dim=1, keepdim=True)   # (B, 1)
        if all_masked.any():
            attn = attn.masked_fill(
                all_masked.unsqueeze(1).unsqueeze(2).expand_as(attn), 0.0
            )

        attn = torch.softmax(attn, dim=-1)                   # (B, nh, 1, H)
        out  = (attn @ V).squeeze(2)                         # (B, nh, hd)
        out  = out.transpose(1, 2).contiguous().view(B, E)   # (B, E)

        return self.out_proj(out)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GATED TEMPORAL MEMORY (tawaregate — DaeMon dan kengaytirilgan)
# ═══════════════════════════════════════════════════════════════════════════════

class GatedTemporalMemory(nn.Module):
    """
    DaeMon tawaregate mexanizmi (kengaytirilgan).

        gate   = σ(W_g · [e_static, e_dynamic])
        output = gate ⊙ tanh(W_h · e_dynamic) + (1−gate) ⊙ e_static

    Kengaytirish: LayerNorm + residual = barqarorroq gradient
    """
    def __init__(self, entity_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate      = nn.Sequential(
            nn.Linear(entity_dim * 2, entity_dim),
            nn.Sigmoid(),
        )
        self.transform = nn.Sequential(
            nn.Linear(entity_dim, entity_dim),
            nn.Tanh(),
        )
        self.norm    = nn.LayerNorm(entity_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, e_static: torch.Tensor, e_dynamic: torch.Tensor) -> torch.Tensor:
        gate   = self.gate(torch.cat([e_static, e_dynamic], dim=-1))
        out    = gate * self.transform(e_dynamic) + (1 - gate) * e_static
        return self.norm(self.dropout(out))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PATH ENCODER — Multi-hop Transformer
# ═══════════════════════════════════════════════════════════════════════════════

class PathTransformerEncoder(nn.Module):
    """
    Multi-hop temporal yo'lni Transformer bilan kodlaydi.
    Har bir qadam: [entity, relation, Δt_encoding] → step_vec
    CLS token → yo'lning umumiy vektori
    """
    def __init__(
        self,
        entity_dim:   int,
        relation_dim: int,
        delta_dim:    int,
        hidden_dim:   int,
        num_heads:    int   = 8,
        num_layers:   int   = 2,
        ffn_dim:      int   = 1024,
        dropout:      float = 0.1,
        max_seq_len:  int   = 16,
    ):
        super().__init__()
        step_dim = entity_dim + relation_dim + delta_dim
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
        self.fc1     = nn.Linear(hidden_dim, hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, entity_dim)
        self.norm1   = nn.LayerNorm(hidden_dim)
        self.norm2   = nn.LayerNorm(entity_dim)
        self.gate    = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.bias    = nn.Parameter(torch.zeros(num_entities))

    def forward(self, q: torch.Tensor, ent_emb: torch.Tensor) -> torch.Tensor:
        h      = self.norm1(q)
        g      = torch.sigmoid(self.gate(h))
        h      = g * F.gelu(self.fc1(h)) + (1 - g) * h
        h      = self.norm2(self.fc2(self.dropout(h)))
        scores = h @ ent_emb.t()
        if scores.size(1) == self.bias.size(0):
            scores = scores + self.bias
        return scores


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
        self.dropout  = nn.Dropout(dropout)
        self.bias     = nn.Parameter(torch.zeros(num_entities))

    def forward(
        self,
        s_emb:       torch.Tensor,  # (B, E)
        r_emb:       torch.Tensor,  # (B, R)
        all_ent_emb: torch.Tensor,  # (E_count, E)
    ) -> torch.Tensor:
        scores = self.dropout(self.sub_proj(s_emb) * self.rel_proj(r_emb)) @ all_ent_emb.t()
        if scores.size(1) == self.bias.size(0):
            scores = scores + self.bias
        return scores


# ═══════════════════════════════════════════════════════════════════════════════
# 8. STORM — ASOSIY MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class STORMModel(nn.Module):
    """
    STORM: Sequential Temporal Oracle with Relational Memory

    ┌──────────────────────────────────────────────────────────────┐
    │  Kirish: (s, r, t_q, paths, history)                        │
    │                                                              │
    │  ① Entity / Relation Embeddings (learned, static)           │
    │  ② Relative Δt Encoding — log-sinusoidal (YANGI)            │
    │                                                              │
    │  ③ Parallel Dual-Path:                                      │
    │    A) Structural Path Reasoning (Transformer, multi-hop)    │
    │    B) Temporal Neighborhood Aggregation:                    │
    │       B1. Query-Adaptive PNA (YANGI — 14×E aggregation)     │
    │       B2. Cross-Snapshot Attention (YANGI — Transformer)    │
    │       B3. Tawaregate Memory (DaeMon → kengaytirilgan)       │
    │                                                              │
    │  ④ Cross-Path Fusion (cross-attention A ↔ B)                │
    │                                                              │
    │  ⑤ Dual Scoring:                                            │
    │    - Path: link_head(fused_q) · all_ent                     │
    │    - Direct: DistMult(s_dynamic, r) · all_ent               │
    │                                                              │
    │  ⑥ Loss: CE + self-adversarial + orthogonal_reg             │
    └──────────────────────────────────────────────────────────────┘
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
        # backward-compat ignored params
        time_dim:        int   = 64,
        proj_dim:        int   = 256,
        temperature:     float = 0.3,
        momentum:        float = 0.995,
        queue_size:      int   = 8192,
        use_direct_scoring: bool = False,
        use_time_encoding:  str  = "both",
        **kwargs,
    ):
        super().__init__()
        self.num_entities    = num_entities
        self.entity_dim      = entity_dim
        self.hidden_dim      = hidden_dim
        self.num_negative    = num_negative
        self.label_smoothing = label_smoothing
        self.w_direct        = w_direct if use_direct_scoring else w_direct
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

        # ── Path Encoder ──────────────────────────────────────────────────────
        self.path_encoder = PathTransformerEncoder(
            entity_dim=entity_dim, relation_dim=relation_dim,
            delta_dim=delta_dim, hidden_dim=hidden_dim,
            num_heads=num_heads, num_layers=num_layers,
            ffn_dim=ffn_dim, dropout=dropout,
        )

        # ── Neighborhood modules ──────────────────────────────────────────────
        if use_history:
            self.pna      = QueryAdaptivePNA(entity_dim, relation_dim, delta_dim, dropout)
            self.csa      = CrossSnapshotAttention(
                entity_dim, relation_dim, delta_dim,
                num_heads=min(4, num_heads), dropout=dropout,
            )
            self.gate_mem = GatedTemporalMemory(entity_dim, dropout)
            self.nb_ctx   = nn.Sequential(
                nn.Linear(entity_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.hist_norm = nn.LayerNorm(hidden_dim)
        else:
            self.pna = self.csa = self.gate_mem = self.nb_ctx = self.hist_norm = None

        # ── Cross-Path Fusion ─────────────────────────────────────────────────
        self.cross_attn   = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.fusion_norm  = nn.LayerNorm(hidden_dim)
        self.fusion_ffn   = nn.Sequential(
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

        # ── Relation temperature ──────────────────────────────────────────────
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

    def _encode_query(
        self,
        subjects:  torch.Tensor,
        relations: torch.Tensor,
        times:     torch.Tensor,
    ) -> torch.Tensor:
        s_emb  = self._ent(subjects)
        r_emb  = self._rel(relations)
        d_zero = self.delta_enc(torch.zeros(subjects.size(0), dtype=torch.long, device=subjects.device))
        return self.query_encoder(torch.cat([s_emb, r_emb, d_zero], dim=-1))

    def _encode_paths(
        self,
        paths:      torch.Tensor,  # (B, P, L, 3)
        path_masks: torch.Tensor,  # (B, P, L)
        query_t:    torch.Tensor,  # (B,)
    ) -> torch.Tensor:             # (B, P, H)
        B, P, L, _ = paths.shape
        paths_f = paths.view(B * P, L, 3)
        masks_f = path_masks.view(B * P, L)

        ent = paths_f[:, :, 0]
        rel = paths_f[:, :, 1]
        t_h = paths_f[:, :, 2]

        t_q  = query_t.unsqueeze(1).expand(-1, P).contiguous().view(B * P)  # (BP,)
        dt   = (t_q.unsqueeze(1) - t_h.float()).clamp(min=0).long()         # (BP, L)

        e_emb = self.ent_emb(ent)
        r_emb = self.rel_emb(rel.clamp(0, self.rel_emb.num_embeddings - 1))
        d_emb = self.delta_enc(dt)

        step  = torch.cat([e_emb, r_emb, d_emb], dim=-1)
        repr_ = self.path_encoder(step, masks_f)
        return repr_.view(B, P, -1)

    def _process_neighborhood(
        self,
        subjects:  torch.Tensor,
        relations: torch.Tensor,
        times:     torch.Tensor,
        history:   torch.Tensor,  # (B, H, 3)
        hist_mask: torch.Tensor,  # (B, H)
        s_emb:     torch.Tensor,  # (B, E)
        r_emb:     torch.Tensor,  # (B, R)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns: (s_dynamic (B,E), ctx (B,hidden_dim))"""
        hist_ent = history[:, :, 0]
        hist_rel = history[:, :, 1]
        hist_t   = history[:, :, 2]

        delta_t  = (times.unsqueeze(1) - hist_t.float()).clamp(min=0).long()
        delta_enc = self.delta_enc(delta_t)

        nb_e = self.ent_emb(hist_ent)
        nb_r = self.rel_emb(hist_rel.clamp(0, self.rel_emb.num_embeddings - 1))

        # A) Query-Adaptive PNA
        pna_out = self.pna(s_emb, nb_e, nb_r, delta_t, r_emb, hist_mask)

        # B) Cross-Snapshot Attention
        messages = nb_e * self.pna.rel_proj(nb_r)
        csa_out  = self.csa(messages, nb_e, delta_enc, r_emb, s_emb, hist_mask)

        # C) Tawaregate memory: fuse PNA + CSA then gate with static
        combined   = (pna_out + csa_out) * 0.5
        s_dynamic  = self.gate_mem(s_emb, combined)

        ctx = self.nb_ctx(torch.cat([s_dynamic, s_emb], dim=-1))
        return s_dynamic, ctx

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

        B      = subjects.size(0)
        device = subjects.device

        s_emb = self._ent(subjects)
        r_emb = self._rel(relations)

        # ── Query ─────────────────────────────────────────────────────────────
        q = self._encode_query(subjects, relations, times)

        # ── Neighborhood ──────────────────────────────────────────────────────
        s_dynamic = None
        if self.use_history and history is not None:
            s_dynamic, ctx = self._process_neighborhood(
                subjects, relations, times, history, hist_mask, s_emb, r_emb
            )
            q = self.hist_norm(q + ctx)

        # ── Paths ─────────────────────────────────────────────────────────────
        path_reprs = self._encode_paths(paths, path_masks, times)

        # ── Cross-path fusion ─────────────────────────────────────────────────
        q_exp = q.unsqueeze(1)
        attn_out, _ = self.cross_attn(q_exp, path_reprs, path_reprs)
        fused = self.fusion_norm(q_exp + attn_out)
        fused = self.fusion_norm2(fused + self.fusion_ffn(fused))
        final_q = fused.squeeze(1)

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

        rel_w  = self.rel_emb.weight
        tmp    = rel_w @ rel_w.t()
        ortho  = torch.norm(tmp - torch.eye(tmp.size(0), device=device), p=2)

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

        q = self._encode_query(subjects, relations, times)

        s_dynamic = None
        if self.use_history and history is not None:
            s_dynamic, ctx = self._process_neighborhood(
                subjects, relations, times, history, hist_mask, s_emb, r_emb
            )
            q = self.hist_norm(q + ctx)

        path_reprs  = self._encode_paths(paths, path_masks, times)
        q_exp       = q.unsqueeze(1)
        attn_out, _ = self.cross_attn(q_exp, path_reprs, path_reprs)
        fused       = self.fusion_norm(q_exp + attn_out)
        fused       = self.fusion_norm2(fused + self.fusion_ffn(fused))
        final_q     = fused.squeeze(1)

        all_ent = self.ent_emb(self.all_entity_ids)
        scores  = self.link_head(final_q, all_ent)
        scores  = scores * self.rel_temp(relations).squeeze(-1).unsqueeze(1)

        if self.direct_head is not None:
            s_t = s_dynamic if s_dynamic is not None else self.ent_emb(subjects)
            s_t = self._diachronic(s_t, times)
            scores = scores + self.w_direct * self.direct_head(s_t, r_emb, all_ent)

        return scores


# Backward compatibility
EliteTKGModel = STORMModel
