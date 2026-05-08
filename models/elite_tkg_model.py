# models/elite_tkg_model.py
"""
EliteTKGModel — Temporal Knowledge Graph Completion uchun mukammal model.

Arxitektura:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. TEMPORAL EMBEDDING MODULE
   - Entity / Relation embedding (Xavier init)
   - Vaqt embedding: Learned + Sinusoidal (har ikkalasi)
   - Time-aware entity representation: ent + time_transform(time)

2. QUERY ENCODER
   - (s, r, t) → MLP → query vector q

3. PATH ENCODER (Transformer)
   - Har bir path: [(o1,r1,t1), (o2,r2,t2), ...] ketma-ketligi
   - Positional + temporal encoding
   - Multi-head self-attention
   - Path-level agregatsiya: CLS token

4. CONTRASTIVE HEAD (MoCo v2 style)
   - Online encoder: q → query_proj → normalize
   - Momentum encoder: entity emb → key_proj → normalize
   - Negative Queue (8192 ta saqlangan key)
   - InfoNCE loss, temperature=0.3

5. LINK PREDICTION HEAD
   - q → score_mlp → entity_dim space
   - ConvE-style interaction: reshape → conv → flatten → score
   - Dot-product + bilinear bilan yakuniy balllar

6. SELF-ADVERSARIAL NEGATIVE SAMPLING
   - Negative lossni negative ballarga qarab og'irlashtirish
   - Murakkab negativlarga ko'proq e'tibor

7. LOSS KOMBINATSIYASI
   - L_total = w1*L_link + w2*L_contrastive + w3*L_self_adv
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import math
import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TEMPORAL EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════

class SinusoidalTimeEncoding(nn.Module):
    """
    Vaqtni sinusoidal encoding bilan ifodalaydi (Transformer positional encoding uslubi).
    t → [sin(t/10000^(2i/d)), cos(t/10000^(2i/d))]
    """
    def __init__(self, dim: int, max_time: int = 10000):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_time, dim)
        position = torch.arange(0, max_time, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:dim // 2])
        self.register_buffer("pe", pe)  # (max_time, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (...,) integer time indices
        t_clamped = t.clamp(0, self.pe.size(0) - 1)
        return self.pe[t_clamped]  # (..., dim)


class TemporalEmbedding(nn.Module):
    """
    Entity, Relation, Time uchun yagona embedding moduli.

    Time = Learned embedding + Sinusoidal encoding (concat → project)
    Entity_t = Entity_emb + time_gate(time_emb)  (gated addition)
    """
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        num_times: int,
        entity_dim: int,
        relation_dim: int,
        time_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.entity_dim   = entity_dim
        self.relation_dim = relation_dim
        self.time_dim     = time_dim

        # Entity & Relation embedding
        self.ent_emb = nn.Embedding(num_entities,  entity_dim,   padding_idx=0)
        self.rel_emb = nn.Embedding(num_relations * 2, relation_dim)  # +inverse

        # Time embedding: learned
        self.time_emb_learned = nn.Embedding(num_times, time_dim)

        # Time embedding: sinusoidal
        self.time_emb_sin = SinusoidalTimeEncoding(time_dim, max_time=num_times + 10)

        # Learned + Sinusoidal → time_dim
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim * 2, time_dim),
            nn.LayerNorm(time_dim),
            nn.GELU(),
        )

        # Temporal transformation: entity + time → entity (gating)
        self.time_gate = nn.Sequential(
            nn.Linear(entity_dim + time_dim, entity_dim),
            nn.Sigmoid(),
        )
        self.time_transform = nn.Linear(time_dim, entity_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Init
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        nn.init.xavier_uniform_(self.time_emb_learned.weight)

    def get_time_emb(self, t: torch.Tensor) -> torch.Tensor:
        """t: (...,) → (..., time_dim)"""
        learned = self.time_emb_learned(t.clamp(0, self.time_emb_learned.num_embeddings - 1))
        sinusoidal = self.time_emb_sin(t)
        combined = torch.cat([learned, sinusoidal], dim=-1)
        return self.time_proj(combined)

    def get_entity_emb(self, e: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.ent_emb(e))

    def get_relation_emb(self, r: torch.Tensor) -> torch.Tensor:
        r_safe = r.clamp(0, self.rel_emb.num_embeddings - 1)
        return self.dropout(self.rel_emb(r_safe))

    def get_temporal_entity_emb(self, e: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Vaqtga bog'liq entity representation.
        e_t = e + sigmoid(W[e; t]) * tanh(W_t * t)
        """
        e_emb = self.ent_emb(e)             # (..., E)
        t_emb = self.get_time_emb(t)        # (..., T)

        gate = self.time_gate(
            torch.cat([e_emb, t_emb], dim=-1)
        )                                    # (..., E) sigmoid
        transform = torch.tanh(
            self.time_transform(t_emb)
        )                                    # (..., E)

        return self.dropout(e_emb + gate * transform)  # (..., E)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PATH ENCODER — Transformer
# ═══════════════════════════════════════════════════════════════════════════════

class PathTransformerEncoder(nn.Module):
    """
    Multi-hop path ni Transformer bilan kodlaydi.

    Input:  [(o1,r1,t1), (o2,r2,t2), ...] ketma-ketligi
    Output: path representation vektori

    CLS token → path ning umumiy ma'nosi
    """
    def __init__(
        self,
        entity_dim: int,
        relation_dim: int,
        time_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Step input: [entity, relation, time] concat → hidden_dim
        step_dim = entity_dim + relation_dim + time_dim
        self.step_proj = nn.Sequential(
            nn.Linear(step_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Positional encoding
        self.pos_emb = nn.Embedding(max_seq_len + 1, hidden_dim)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,        # Pre-LN: barqarorroq
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # FutureWarning ni bartaraf etish
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        step_vecs: torch.Tensor,   # (BP, L, step_dim)
        mask: torch.Tensor,        # (BP, L) True=valid
    ) -> torch.Tensor:             # (BP, hidden_dim)
        BP, L, _ = step_vecs.shape

        # Step projection
        x = self.step_proj(step_vecs)       # (BP, L, H)

        # CLS token prepend
        cls = self.cls_token.expand(BP, -1, -1)  # (BP, 1, H)
        x = torch.cat([cls, x], dim=1)            # (BP, L+1, H)

        # Positional encoding
        positions = torch.arange(L + 1, device=x.device)
        x = x + self.pos_emb(positions).unsqueeze(0)  # (BP, L+1, H)

        # Attention mask: CLS har doim valid
        cls_mask = torch.ones(BP, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)  # (BP, L+1)
        attn_mask = ~full_mask  # True = IGNORE (PyTorch convention)

        # Transformer
        out = self.transformer(x, src_key_padding_mask=attn_mask)  # (BP, L+1, H)
        out = self.norm(out)

        # CLS token output = path representation
        return out[:, 0, :]  # (BP, H)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. QUERY ENCODER
# ═══════════════════════════════════════════════════════════════════════════════

class QueryEncoder(nn.Module):
    """
    (s, r, t) → query vector

    Subject + Relation + Time concat → Deep MLP → q
    """
    def __init__(
        self,
        entity_dim: int,
        relation_dim: int,
        time_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        inp_dim = entity_dim + relation_dim + time_dim
        self.encoder = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        s_emb: torch.Tensor,    # (B, E)
        r_emb: torch.Tensor,    # (B, R)
        t_emb: torch.Tensor,    # (B, T)
    ) -> torch.Tensor:          # (B, H)
        x = torch.cat([s_emb, r_emb, t_emb], dim=-1)
        return self.encoder(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LINK PREDICTION HEAD — ConvE uslubi
# ═══════════════════════════════════════════════════════════════════════════════

class LinkPredHead(nn.Module):
    """
    q → score barcha entitylar bilan.

    ConvE-inspired: reshape + conv + fc → entity_dim → dot-product
    """
    def __init__(
        self,
        hidden_dim: int,
        entity_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Simple but effective: highway connection bilan
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, entity_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(entity_dim)
        self.dropout = nn.Dropout(dropout)

        # Highway gate
        self.gate = nn.Linear(hidden_dim, hidden_dim)

        # Bias per entity
        self.bias = None  # Dataset yuklanganidan keyin o'rnatiladi

    def set_entity_count(self, n: int, device):
        self.bias = nn.Parameter(torch.zeros(n, device=device))

    def forward(
        self,
        q: torch.Tensor,            # (B, H)
        ent_emb: torch.Tensor,      # (E_count, entity_dim)
    ) -> torch.Tensor:              # (B, E_count)
        # Highway network
        h = self.norm1(q)
        gate = torch.sigmoid(self.gate(h))
        h = gate * F.gelu(self.fc1(h)) + (1 - gate) * h  # residual
        h = self.dropout(h)

        # Project to entity space
        h = self.norm2(self.fc2(h))  # (B, entity_dim)

        # Dot-product with all entities
        scores = h @ ent_emb.t()  # (B, E_count)

        if self.bias is not None and scores.size(1) == self.bias.size(0):
            scores = scores + self.bias.unsqueeze(0)

        return scores


# ═══════════════════════════════════════════════════════════════════════════════
# 4a. NEIGHBORHOOD AGGREGATOR — DaeMon-inspired DistMult + PNA
# ═══════════════════════════════════════════════════════════════════════════════

class NeighborhoodAggregator(nn.Module):
    """
    DaeMon-inspired neighborhood aggregation for entity s.

    Arxitektura:
    - DistMult message function: m_i = emb(o_i) ⊙ W_r(emb(r_i))
    - Time decay: decay_i = exp(-γ * (t - t_i))
    - PNA aggregation: [mean, max, attn_weighted]
    - Query-conditioned attention: a_i ∝ e_s · m_i
    - Output: dynamic entity embedding (B, entity_dim)
    - Context vector for query fusion (B, hidden_dim)

    GRU dan farqi: GRU ketma-ketlik tartibini modellaydi (noto'g'ri
    inductive bias). Bu modul esa qo'shnilarni TO'G'RI agregatsiyalaydi
    (DaeMon/PathAggNet kabi).
    """
    def __init__(
        self,
        entity_dim:   int,
        relation_dim: int,
        hidden_dim:   int,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.entity_dim = entity_dim

        # DistMult: relation projection for messages (entity_dim space)
        self.rel_proj = nn.Linear(relation_dim, entity_dim, bias=False)

        # Query-conditioned attention: e_s × W_q → attn over neighbors
        self.attn_q = nn.Linear(entity_dim, entity_dim, bias=False)

        # Time decay: learnable log-scale
        self.log_gamma = nn.Parameter(torch.tensor(-3.0))

        # PNA output: [mean, max, attn, e_s] → entity_dim
        self.out_proj = nn.Sequential(
            nn.Linear(entity_dim * 4, entity_dim * 2),
            nn.LayerNorm(entity_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(entity_dim * 2, entity_dim),
            nn.LayerNorm(entity_dim),
        )

        # Context projection for query fusion: entity_dim → hidden_dim
        self.ctx_proj = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        e_emb:         torch.Tensor,  # (B, entity_dim) — static subject embedding
        hist_ent_emb:  torch.Tensor,  # (B, H, entity_dim) — neighbor embeddings
        hist_rel_emb:  torch.Tensor,  # (B, H, relation_dim) — relation embeddings
        hist_time_raw: torch.Tensor,  # (B, H) — raw integer timestamps
        query_time:    torch.Tensor,  # (B,) — raw query timestamps
        hist_mask:     torch.Tensor,  # (B, H) bool — True = valid
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            dynamic_emb: (B, entity_dim) — neighborhood-aware entity representation
            ctx:         (B, hidden_dim) — context vector for query fusion
        """
        B, H, E = hist_ent_emb.shape

        # DistMult messages: e_neighbor ⊙ W_rel(r_i)
        rel_projected = self.rel_proj(hist_rel_emb)   # (B, H, E)
        messages      = hist_ent_emb * rel_projected  # (B, H, E)

        # Time decay: exp(-γ * |t - t_i|)
        time_diff = (query_time.float().unsqueeze(1) - hist_time_raw.float()).clamp(min=0)
        gamma     = torch.exp(self.log_gamma)
        decay     = torch.exp(-gamma * time_diff)     # (B, H)

        # Decay applied to messages
        messages_dec = messages * decay.unsqueeze(-1)  # (B, H, E)

        # Query-conditioned attention
        q_repr   = self.attn_q(e_emb)                           # (B, E)
        attn_raw = torch.bmm(
            q_repr.unsqueeze(1),                                 # (B, 1, E)
            messages_dec.transpose(1, 2)                         # (B, E, H)
        ).squeeze(1) / math.sqrt(E)                              # (B, H)

        fill_val = -1e4 if attn_raw.dtype == torch.float16 else -1e9
        attn_raw = attn_raw.masked_fill(~hist_mask, fill_val)
        attn     = torch.softmax(attn_raw, dim=-1)               # (B, H)

        # PNA: mean, max, attention-weighted aggregation
        valid_f  = hist_mask.float().unsqueeze(-1)               # (B, H, 1)
        n_valid  = hist_mask.float().sum(1, keepdim=True).clamp(min=1)
        valid_msgs = messages * valid_f                          # (B, H, E)

        mean_msg = valid_msgs.sum(1) / n_valid                   # (B, E)
        max_msg  = valid_msgs.max(1).values                      # (B, E)
        attn_msg = (attn.unsqueeze(-1) * messages).sum(1)        # (B, E)

        # Combine with original embedding
        combined     = torch.cat([mean_msg, max_msg, attn_msg, e_emb], dim=-1)  # (B, 4E)
        dynamic_emb  = self.out_proj(combined)                   # (B, E)
        dynamic_emb  = self.dropout(dynamic_emb)

        # Context vector for query fusion
        ctx = self.ctx_proj(dynamic_emb)                         # (B, hidden_dim)

        return dynamic_emb, ctx


# ═══════════════════════════════════════════════════════════════════════════════
# 4b. DIRECT SCORING HEAD — DistMult uslubi (WIKI/YAGO uchun)
# ═══════════════════════════════════════════════════════════════════════════════

class DirectScoringHead(nn.Module):
    """
    DistMult-uslubida to'g'ridan-to'g'ri baholash.

    score(s, r, o, t) = (s_t ⊙ W_r * r) · o_static

    Nima uchun kerak?
    - WIKI/YAGO da kam relatsiya (10–24) → yo'llar informatif emas
    - DistMult entity embedding larini to'g'ri o'rganadi
    - Path encoder bilan birgalikda: yakunda ancha yaxshiroq natija
    """
    def __init__(
        self,
        entity_dim:   int,
        relation_dim: int,
        dropout:      float = 0.1,
    ):
        super().__init__()
        # relation_dim → entity_dim proektsiyasi
        self.rel_proj = nn.Linear(relation_dim, entity_dim, bias=False)
        # Subject temporal embedding ni entity_dim ga moslashtirish
        self.sub_proj = nn.Sequential(
            nn.Linear(entity_dim, entity_dim),
            nn.LayerNorm(entity_dim),
            nn.GELU(),
        )
        self.dropout  = nn.Dropout(dropout)
        # Per-entity bias
        self.bias: Optional[nn.Parameter] = None

    def set_entity_count(self, n: int, device):
        self.bias = nn.Parameter(torch.zeros(n, device=device))

    def forward(
        self,
        s_t_emb:     torch.Tensor,  # (B, entity_dim) — temporal subject
        r_emb:       torch.Tensor,  # (B, relation_dim)
        all_ent_emb: torch.Tensor,  # (E, entity_dim)
    ) -> torch.Tensor:              # (B, E)
        # DistMult: (s_t * W_r(r)) · o
        s_proj = self.sub_proj(s_t_emb)          # (B, E)
        r_proj = self.rel_proj(r_emb)             # (B, E)
        interaction = self.dropout(s_proj * r_proj)  # (B, E) element-wise
        scores = interaction @ all_ent_emb.t()    # (B, E_count)
        if self.bias is not None and scores.size(1) == self.bias.size(0):
            scores = scores + self.bias.unsqueeze(0)
        return scores


class DiachronicGating(nn.Module):
    """
    DE-SimplE uslubida vaqtga bog'liq entity gating.

    e_t = e_static * (1 + amp * sin(freq * t + phase))

    Vaqt bo'yicha ravon o'zgarish → test temporal umumlashuvini yaxshilaydi.
    """
    def __init__(self, entity_dim: int, num_times: int, dropout: float = 0.1):
        super().__init__()
        d = entity_dim
        # Amplitude: har bir entity uchun learned
        self.amp   = nn.Embedding(1, d)          # global amplitude
        self.freq  = nn.Embedding(1, d)          # global frequency
        self.phase = nn.Embedding(1, d)          # global phase
        self.dropout = nn.Dropout(dropout)
        nn.init.uniform_(self.amp.weight,   -0.1, 0.1)
        nn.init.uniform_(self.freq.weight,   0.5, 2.0)
        nn.init.uniform_(self.phase.weight, -math.pi, math.pi)
        self.num_times = max(num_times, 1)

    def forward(
        self,
        e_emb: torch.Tensor,  # (..., entity_dim)
        t:     torch.Tensor,  # (...,) integer timestamps
    ) -> torch.Tensor:        # (..., entity_dim)
        t_norm = t.float() / self.num_times           # 0..1 ga normalize
        # Broadcast: (..., 1) × (1, E) → (..., E)
        t_exp = t_norm.unsqueeze(-1)
        amp   = self.amp.weight[0]    # (E,)
        freq  = self.freq.weight[0]   # (E,)
        phase = self.phase.weight[0]  # (E,)
        gate  = 1.0 + amp * torch.sin(freq * t_exp + phase)  # (..., E)
        return self.dropout(e_emb * gate)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MOCO-STYLE CONTRASTIVE MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class PathLevelContrastive(nn.Module):
    """
    Path-Level Contrastive Learning (CTPL maqolasiga mos).

    Oldingi muammo:
      query  = final_q  (path encoder orqali, 512-dim)
      positive = ent_emb(o) (oddiy entity embedding, 256-dim)
      → Ikki turli space → CL ≈ random baseline (8.0), hech narsa o'rganmaydi

    Tuzatish (haqiqiy path-level):
      query   = final_q  (s→...→? so'rovi, path + query fusion)
      positive = path_repr(s→o) ning o'rtachasi  ← TO'G'RI YO'L representation
      negative = path_repr(s→neg) lar            ← NOTO'G'RI yo'l representation

    Endi ikkalasi ham bir xil space (hidden_dim) da!
    Model "to'g'ri yo'l representation" va "noto'g'ri yo'l" ni ajratishni o'rganadi.
    """
    def __init__(
        self,
        hidden_dim: int,
        proj_dim:   int,
        queue_size: int   = 4096,
        momentum:   float = 0.995,
        temperature: float = 0.3,
    ):
        super().__init__()
        self.proj_dim    = proj_dim
        self.queue_size  = queue_size
        self.momentum    = momentum
        self.temperature = temperature

        # Query projection: final_q (H) → proj_dim
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        # Key projection: path_repr (H) → proj_dim
        # Online va momentum — ikkalasi ham bir xil H-dim inputdan
        self.key_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        # Momentum encoder (EMA, gradient yo'q)
        self.key_proj_m = copy.deepcopy(self.key_proj)
        for p in self.key_proj_m.parameters():
            p.requires_grad_(False)

        # Negative queue: path representations
        self.register_buffer(
            "queue",
            F.normalize(torch.randn(proj_dim, queue_size), dim=0)
        )
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _update_momentum_encoder(self):
        for p_m, p in zip(self.key_proj_m.parameters(), self.key_proj.parameters()):
            p_m.data = p_m.data * self.momentum + p.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _enqueue(self, keys: torch.Tensor):
        K = keys.size(0)
        ptr = int(self.queue_ptr)
        if ptr + K <= self.queue_size:
            self.queue[:, ptr:ptr + K] = keys.t()
        else:
            end = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:end].t()
            self.queue[:, :K - end] = keys[end:].t()
        self.queue_ptr[0] = (ptr + K) % self.queue_size

    def forward(
        self,
        final_q:      torch.Tensor,  # (B, H)   — query (fused)
        pos_path_repr: torch.Tensor, # (B, H)   — to'g'ri yo'l representation
        neg_path_repr: torch.Tensor, # (B, N, H) — noto'g'ri yo'llar
    ) -> torch.Tensor:
        B = final_q.size(0)

        # Query projection (online, gradient oqadi)
        q_proj = F.normalize(self.query_proj(final_q), dim=-1)  # (B, d)

        # Positive key projection (momentum, gradient yo'q)
        self._update_momentum_encoder()
        with torch.no_grad():
            pos_proj = self.key_proj_m(pos_path_repr)           # (B, d)
            pos_proj = F.normalize(pos_proj, dim=-1)

        # Negative key projection (online)
        B_, N, H = neg_path_repr.shape
        neg_flat = neg_path_repr.view(B_ * N, H)
        neg_proj = self.key_proj(neg_flat)                       # (B*N, d)
        neg_proj = F.normalize(neg_proj, dim=-1)
        neg_proj = neg_proj.view(B_, N, -1)                      # (B, N, d)

        # InfoNCE loss
        pos_logit     = (q_proj * pos_proj).sum(-1, keepdim=True) / self.temperature  # (B,1)
        neg_logit_bat = torch.bmm(
            q_proj.unsqueeze(1), neg_proj.transpose(1, 2)
        ).squeeze(1) / self.temperature                           # (B, N)
        neg_logit_q   = (q_proj @ self.queue.clone().detach()) / self.temperature     # (B, Q)

        logits = torch.cat([pos_logit, neg_logit_bat, neg_logit_q], dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=final_q.device)
        loss   = F.cross_entropy(logits, labels)

        self._enqueue(pos_proj.detach())
        return loss


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ASOSIY MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class EliteTKGModel(nn.Module):
    """
    Temporal Knowledge Graph Completion uchun to'liq model.

    Forward:
      1. (s,r,t) → TemporalEmbedding → QueryEncoder → q
      2. paths → PathTransformerEncoder → path_repr
      3. q + path_repr fusion → link prediction
      4. MoCo contrastive loss
      5. Self-adversarial loss
    """

    def __init__(
        self,
        num_entities:  int,
        num_relations: int,
        num_times:     int,
        entity_dim:    int = 256,
        relation_dim:  int = 256,
        time_dim:      int = 64,
        hidden_dim:    int = 512,
        proj_dim:      int = 256,
        num_heads:     int = 8,
        num_layers:    int = 2,
        ffn_dim:       int = 1024,
        num_negative:  int = 256,
        dropout:       float = 0.1,
        temperature:   float = 0.3,
        momentum:      float = 0.995,
        queue_size:    int = 8192,
        max_seq_len:   int = 16,
        label_smoothing: float = 0.1,
        use_direct_scoring: bool = False,
        use_diachronic:     bool = False,
        w_direct:           float = 0.0,
        use_history:        bool = False,
    ):
        super().__init__()
        self.num_entities       = num_entities
        self.num_relations      = num_relations
        self.entity_dim         = entity_dim
        self.hidden_dim         = hidden_dim
        self.num_negative       = num_negative
        self.label_smoothing    = label_smoothing
        self.use_direct_scoring = use_direct_scoring
        self.use_diachronic     = use_diachronic
        self.w_direct           = w_direct
        self.use_history        = use_history

        # ── 1. Embedding ──────────────────────────────────────────────────────
        self.embeddings = TemporalEmbedding(
            num_entities=num_entities,
            num_relations=num_relations,
            num_times=num_times,
            entity_dim=entity_dim,
            relation_dim=relation_dim,
            time_dim=time_dim,
            dropout=dropout,
        )

        # ── 2. Query Encoder ──────────────────────────────────────────────────
        self.query_encoder = QueryEncoder(
            entity_dim=entity_dim,
            relation_dim=relation_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # ── 3. Path Encoder ───────────────────────────────────────────────────
        step_dim = entity_dim + relation_dim + time_dim
        self.path_encoder = PathTransformerEncoder(
            entity_dim=entity_dim,
            relation_dim=relation_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # ── 4. Fusion: q + path_repr → final_q ───────────────────────────────
        # Cross-attention: q nima so'rashi kerak, path_repr javob beradi
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.fusion_norm2 = nn.LayerNorm(hidden_dim)

        # ── 5. Link Prediction Head ───────────────────────────────────────────
        self.link_head = LinkPredHead(
            hidden_dim=hidden_dim,
            entity_dim=entity_dim,
            dropout=dropout,
        )

        # ── 6. Path-Level Contrastive ─────────────────────────────────────────
        self.contrastive = PathLevelContrastive(
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            queue_size=queue_size,
            momentum=momentum,
            temperature=temperature,
        )

        # Negative entity embedding → hidden_dim (proxy path representation)
        # entity_dim → hidden_dim, negative lar uchun ishlatiladi
        self.neg_path_proj = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ── 7. Relation-specific scaling ─────────────────────────────────────
        # Har bir relation uchun o'z temperature si
        self.rel_temperature = nn.Embedding(num_relations, 1)
        nn.init.constant_(self.rel_temperature.weight, 1.0)

        # ── 8. Neighborhood Aggregator (DaeMon-inspired) ────────────────────────
        if use_history:
            self.history_encoder = NeighborhoodAggregator(
                entity_dim   = entity_dim,
                relation_dim = relation_dim,
                hidden_dim   = hidden_dim,
                dropout      = dropout,
            )
            # Context → query fusion: simple residual + norm
            self.hist_gate = None                           # Artiq kerak emas
            self.hist_norm = nn.LayerNorm(hidden_dim)
        else:
            self.history_encoder = None
            self.hist_gate       = None
            self.hist_norm       = None

        # ── 9. Direct Scoring Head (WIKI/YAGO uchun) ──────────────────────────
        if use_direct_scoring:
            self.direct_head = DirectScoringHead(entity_dim, relation_dim, dropout)
        else:
            self.direct_head = None

        # ── 10. Diachronic Gating (vaqt bo'yicha smooth entity embedding) ─────
        if use_diachronic:
            self.diachronic = DiachronicGating(entity_dim, num_times, dropout)
        else:
            self.diachronic = None

        self.dropout = nn.Dropout(dropout)

        # All entity indices (eval uchun)
        self.register_buffer(
            "all_entity_ids",
            torch.arange(num_entities, dtype=torch.long)
        )

    # ── Forward yordamchi metodlar ────────────────────────────────────────────

    def _encode_query(self, s, r, t):
        """(B,) → (B, H)"""
        s_emb = self.embeddings.get_temporal_entity_emb(s, t)  # (B, E)
        r_emb = self.embeddings.get_relation_emb(r)             # (B, R)
        t_emb = self.embeddings.get_time_emb(t)                 # (B, T)
        return self.query_encoder(s_emb, r_emb, t_emb)          # (B, H)

    def _encode_paths(self, paths, path_masks):
        """
        paths:      (B, P, L, 3)  [entity, relation, time]
        path_masks: (B, P, L)
        → path_repr: (B, P, H)
        """
        B, P, L, _ = paths.shape

        # Flatten: (B*P, L, 3)
        paths_flat = paths.view(B * P, L, 3)
        masks_flat = path_masks.view(B * P, L)

        num_times = self.embeddings.time_emb_learned.num_embeddings
        ent = paths_flat[:, :, 0]                                     # (BP, L)
        rel = paths_flat[:, :, 1]                                     # (BP, L)
        t   = paths_flat[:, :, 2].clamp(0, num_times - 1)            # (BP, L)

        # Step embeddings
        e_emb = self.embeddings.get_entity_emb(ent)          # (BP, L, E)
        r_emb = self.embeddings.get_relation_emb(rel)         # (BP, L, R)
        t_emb = self.embeddings.get_time_emb(t)               # (BP, L, T)

        step_vecs = torch.cat([e_emb, r_emb, t_emb], dim=-1)  # (BP, L, E+R+T)

        # Path encoding
        path_repr = self.path_encoder(step_vecs, masks_flat)  # (BP, H)
        return path_repr.view(B, P, -1)                        # (B, P, H)

    def _fuse_query_paths(self, q, path_reprs):
        """
        Cross-attention: q (query) path_reprs ni attend qiladi.
        q:           (B, H)
        path_reprs:  (B, P, H)
        → fused: (B, H)
        """
        q_expanded = q.unsqueeze(1)  # (B, 1, H)

        # Cross-attention: query=q, key=value=paths
        attn_out, _ = self.cross_attn(q_expanded, path_reprs, path_reprs)
        # attn_out: (B, 1, H)

        # Residual + norm
        fused = self.fusion_norm(q_expanded + attn_out)          # (B, 1, H)
        fused = self.fusion_norm2(fused + self.fusion_ffn(fused)) # (B, 1, H)
        return fused.squeeze(1)                                    # (B, H)

    def _self_adversarial_loss(self, scores, targets, neg_objs):
        """
        Self-adversarial negative sampling loss.
        Negative ballarga softmax og'irligi beriladi:
        L = -Σ p(neg_i) * log σ(-score(neg_i))
        Bu murakkab negativlarga ko'proq e'tibor beradi.
        """
        B = scores.size(0)
        device = scores.device

        if neg_objs.size(1) == 0:
            return torch.tensor(0.0, device=device)

        # Negative scores: (B, N)
        N = neg_objs.size(1)
        neg_scores = scores.gather(1, neg_objs)  # (B, N)

        # Self-adversarial weight: softmax over negative scores
        with torch.no_grad():
            weights = F.softmax(neg_scores * 0.5, dim=-1)  # (B, N)

        # Log sigmoid loss
        neg_loss = -(weights * F.logsigmoid(-neg_scores)).sum(dim=-1).mean()

        # Positive loss
        pos_scores = scores.gather(1, targets.unsqueeze(1))  # (B, 1)
        pos_loss = -F.logsigmoid(pos_scores).mean()

        return pos_loss + neg_loss

    # ── Asosiy forward ────────────────────────────────────────────────────────

    def _compute_neighborhood(
        self,
        subjects:  torch.Tensor,   # (B,)
        times:     torch.Tensor,   # (B,)
        history:   torch.Tensor,   # (B, H, 3) — [entity, relation, time]
        hist_mask: torch.Tensor,   # (B, H)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        NeighborhoodAggregator: entity s uchun dinamik embedding va kontekst.

        Returns:
            dynamic_emb: (B, entity_dim) — neighborhood-aware entity embedding
            ctx:         (B, hidden_dim) — context for query fusion
        """
        num_times = self.embeddings.time_emb_learned.num_embeddings

        hist_ent = history[:, :, 0]                            # (B, H) int
        hist_rel = history[:, :, 1]                            # (B, H) int
        hist_tm  = history[:, :, 2]                            # (B, H) raw int (time decay uchun)

        # Subject static embedding
        s_emb = self.embeddings.get_entity_emb(subjects)       # (B, E)

        # Neighbor embeddings (lookup)
        nb_emb = self.embeddings.get_entity_emb(hist_ent)      # (B, H, E)
        r_emb  = self.embeddings.get_relation_emb(hist_rel)    # (B, H, R)

        return self.history_encoder(
            s_emb,       # (B, E)
            nb_emb,      # (B, H, E)
            r_emb,       # (B, H, R)
            hist_tm,     # (B, H) raw timestamps
            times,       # (B,) raw query timestamps
            hist_mask,   # (B, H)
        )  # → (dynamic_emb (B,E), ctx (B,H_dim))

    def forward(
        self,
        subjects:   torch.Tensor,    # (B,)
        relations:  torch.Tensor,    # (B,)
        objects:    torch.Tensor,    # (B,)
        times:      torch.Tensor,    # (B,)
        paths:      torch.Tensor,    # (B, P, L, 3)
        path_masks: torch.Tensor,    # (B, P, L)
        neg_objects: torch.Tensor,   # (B, N) — negative entities
        history:    Optional[torch.Tensor] = None,  # (B, H, 3)
        hist_mask:  Optional[torch.Tensor] = None,  # (B, H)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            scores:   (B, num_entities) — link prediction
            losses:   {"link": t, "contrastive": t, "self_adv": t}
        """
        B = subjects.size(0)
        device = subjects.device

        # ── 1. Query encoding ─────────────────────────────────────────────────
        q = self._encode_query(subjects, relations, times)  # (B, H)

        # ── 1b. Neighborhood Aggregation ─────────────────────────────────────
        s_dynamic = None   # (B, entity_dim) — dynamic entity emb (neighborhoods)
        if self.history_encoder is not None and history is not None:
            s_dynamic, ctx = self._compute_neighborhood(
                subjects, times, history, hist_mask
            )                                                # ctx: (B, H_dim)
            # Simple residual fusion: q += neighborhood context
            q = self.hist_norm(q + ctx)                      # (B, H)

        # ── 2. Path encoding ──────────────────────────────────────────────────
        path_reprs = self._encode_paths(paths, path_masks)  # (B, P, H)

        # ── 3. Fusion ─────────────────────────────────────────────────────────
        final_q = self._fuse_query_paths(q, path_reprs)     # (B, H)

        # ── 4. Link prediction ────────────────────────────────────────────────
        all_ent_emb = self.embeddings.get_entity_emb(self.all_entity_ids)  # (E, dim)
        scores = self.link_head(final_q, all_ent_emb)                       # (B, E)

        # Relation-specific temperature scaling
        rel_temp = self.rel_temperature(relations).squeeze(-1)  # (B,)
        scores = scores * rel_temp.unsqueeze(1)

        # ── 4b. Direct scoring (DistMult) — WIKI/YAGO uchun ──────────────────
        if self.direct_head is not None and self.w_direct > 0.0:
            # CRITICAL: Use DYNAMIC entity embedding if available (neighborhood-aware)
            # This is the key improvement over static embeddings
            if s_dynamic is not None:
                s_t_emb = s_dynamic                                         # (B, E) DYNAMIC
            else:
                s_t_emb = self.embeddings.get_temporal_entity_emb(subjects, times)  # (B, E)
            r_emb_d = self.embeddings.get_relation_emb(relations)               # (B, R)
            # Diachronic gating (agar yoqilgan bo'lsa)
            if self.diachronic is not None:
                s_t_emb = self.diachronic(s_t_emb, times)
            direct_scores = self.direct_head(s_t_emb, r_emb_d, all_ent_emb)   # (B, E)
            scores = scores + self.w_direct * direct_scores

        # ── 5. Link prediction loss (label smoothing bilan) ───────────────────
        link_loss = F.cross_entropy(
            scores, objects,
            label_smoothing=self.label_smoothing,
        )

        # ── 6. Path-Level Contrastive loss ───────────────────────────────────
        # Positive: s→o yo'llarining path representation (mean pooling)
        # Negative: s→neg_i yo'llarining path representation
        # Ikkalasi ham bir xil hidden_dim space da → haqiqiy path-level CL

        # Positive path representation:
        # path_reprs (B, P, H) — bu allaqachon s→o yo'llari encode qilingan
        # (lekin oxirgi hop olib tashlangan, shuning uchun to'g'ri)
        pos_path_repr = path_reprs.mean(dim=1)  # (B, H) — P ta yo'lning o'rtachasi

        # Negative path representations:
        # Negative entitylar uchun ham path_reprs tuzish kerak.
        # Soddalashtirilgan: query encoder orqali negative entity embedding larini
        # hidden_dim ga ko'taramiz (proxy negative representation)
        if neg_objects.size(1) > 0:
            N = neg_objects.size(1)
            # Neg entity temporal embedding → (B, N, entity_dim)
            neg_ent_t = self.embeddings.get_temporal_entity_emb(
                neg_objects.view(-1),
                times.unsqueeze(1).expand(-1, N).contiguous().view(-1)
            ).view(B, N, -1)                                    # (B, N, E)
            # Entity_dim → hidden_dim (neg_proj layer)
            neg_path_repr = self.neg_path_proj(neg_ent_t)       # (B, N, H)
        else:
            neg_idx = torch.randint(0, self.num_entities, (B, 64), device=device)
            neg_ent_t = self.embeddings.get_temporal_entity_emb(
                neg_idx.view(-1), times.unsqueeze(1).expand(-1, 64).contiguous().view(-1)
            ).view(B, 64, -1)
            neg_path_repr = self.neg_path_proj(neg_ent_t)       # (B, 64, H)

        contrastive_loss = self.contrastive(final_q, pos_path_repr, neg_path_repr)

        # ── 7. Self-adversarial loss ──────────────────────────────────────────
        if neg_objects.size(1) > 0:
            self_adv_loss = self._self_adversarial_loss(scores, objects, neg_objects)
        else:
            self_adv_loss = torch.tensor(0.0, device=device)

        losses = {
            "link":        link_loss,
            "contrastive": contrastive_loss,
            "self_adv":    self_adv_loss,
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
    ) -> torch.Tensor:          # (B, num_entities)
        q = self._encode_query(subjects, relations, times)

        # Neighborhood aggregation
        s_dynamic = None
        if self.history_encoder is not None and history is not None:
            s_dynamic, ctx = self._compute_neighborhood(subjects, times, history, hist_mask)
            q = self.hist_norm(q + ctx)

        path_reprs  = self._encode_paths(paths, path_masks)
        final_q     = self._fuse_query_paths(q, path_reprs)
        all_ent_emb = self.embeddings.get_entity_emb(self.all_entity_ids)
        scores      = self.link_head(final_q, all_ent_emb)
        rel_temp    = self.rel_temperature(relations).squeeze(-1)
        scores      = scores * rel_temp.unsqueeze(1)

        # Direct scoring — DYNAMIC entity embedding when available
        if self.direct_head is not None and self.w_direct > 0.0:
            if s_dynamic is not None:
                s_t_emb = s_dynamic
            else:
                s_t_emb = self.embeddings.get_temporal_entity_emb(subjects, times)
            r_emb_d = self.embeddings.get_relation_emb(relations)
            if self.diachronic is not None:
                s_t_emb = self.diachronic(s_t_emb, times)
            direct_scores = self.direct_head(s_t_emb, r_emb_d, all_ent_emb)
            scores = scores + self.w_direct * direct_scores

        return scores
