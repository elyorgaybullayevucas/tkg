# models/elite_tkg_model.py
"""
ORION: Ontology-aware Relational pattern Inference with Ordered Networks

═══════════════════════════════════════════════════════════════════════════════
  Haqiqiy Ilmiy Yangiliklar (vs DaeMon va barcha mavjud modellar)
═══════════════════════════════════════════════════════════════════════════════

1. TEMPORAL PATTERN LIBRARY (TPL)  ★ MUTLOQO YANGI
   ─────────────────────────────────────────────────
   K ta o'rgatiluvchi "pattern" vektori — abstract vaqtinchalik xulq-atvor
   shablonlari. Hech qayerda (DaeMon, RE-GCN, TITer da) bunday yo'q.

   Asosiy g'oya: TKG da takrorlanuvchi strukturaviy naqshlar mavjud:
     • "Entity A, so'ngra B"  •  "Relation r1 dan keyin r2 keladi"
   Bu naqshlar entity ga bog'liq emas → entity-independent generalizatsiya.

   Ishlatish:
     patterns ∈ R^{K×H}  (K ta learnable vektor)
     query = f(r_query, history_context)
     retrieved = softmax(query · patterns^T / √H) · patterns

   + Pattern diversity regularization:
     L_div = mean(‖P_i · P_j‖²)  (patternlar bir-biridan farq qilsin)

2. RELATION PROFILE ENCODING (RPE)  ★ YANGI
   ──────────────────────────────────────────
   Entity s ning vaqt-og'irlangan relation faoliyat profili:
     profile[r] = Σ_{i: r_i=r} exp(−γ·Δt_i)

   Asosiy g'oya: "Qanday relationlarda faol bo'lgan?" → kelgusi prediction uchun
   YAGO (10R): 10-o'lchamli vektor barcha relation faoliyatni to'liq tasvirlaydi
   WIKI (24R): 24-o'lchamli vektor ham juda informativ

   Entity-independent: scatter_add over relation indices (entity index ishlatilmaydi)

3. ENTITY-INDEPENDENT HISTORY TRANSFORMER  ★ YANGI (DaeMon yaxshilangan)
   ─────────────────────────────────────────────────────────────────────────
   DaeMon: sequential tawaregate iteratsiyasi → erta tarix yo'qoladi
   ORION:  Transformer over {(rel_i, Δt_enc_i)} → parallel, long-range attention

   step = [rel_emb, delta_enc]  — entity embedding yo'q!
   CLS token → history context vektori

4. RELATIVE TEMPORAL ENCODING (RTE)  ★ (CATRE dan olingan, DaeMon ustidan)
   ─────────────────────────────────────────────────────────────────────────
   DaeMon: mutlaq snapshot indeksi → test t=222..231 ko'rilmagan (WIKI)
   ORION:  Δt = t_query − t_history, log-sinusoidal → kelajak vaqtlari uchun ham

5. THREE-SIGNAL FUSION  ★ YANGI ARXITEKTURA
   ────────────────────────────────────────
   A: cross_out    = query + path_attention  (entity-aware paths)
   B: hist_signal  = history_transformer + relation_profile + tawaregate
   C: pattern_out  = TPL retrieval           (abstract patterns)

   final_q = fusion_mlp(cat[A, B, C]) → 3×H → H

═══════════════════════════════════════════════════════════════════════════════
  Arxitektura sxemasi
═══════════════════════════════════════════════════════════════════════════════

  Input: (s, r, t_q, paths[B,P,L,3], history[B,H,3])
      │
      ├─① ent_emb[s], rel_emb[r], delta_enc(Δt)
      │
      ├─② History Branch (use_history=True):
      │    nb_rel, delta_t  ←  history[:,:,1], history[:,:,2]
      │    profile_enc   = RelationProfile(nb_rel, delta_t, mask)   [NOVEL]
      │    hist_out      = HistTransformer(rel_emb_hist, delta_enc)  [NOVEL]
      │    s_dynamic     = GatedTemporalMemory(s_emb, hist_out[:E]) [DaeMon]
      │    nb_ctx        = proj([s_dynamic, s_emb]) → H
      │    hist_signal   = layer_norm(profile_enc + nb_ctx)
      │
      ├─③ Path Branch (entity-independent):
      │    step = [rel_emb_path, delta_enc_path]   ← entity yo'q!
      │    path_reprs = PathTransformer(steps)  → (B, P, H)
      │
      ├─④ Query + Cross-path Attention:
      │    q = query_encoder([s_emb, r_emb, delta_zero])
      │    q += hist_signal  (if history)
      │    cross_out = q + MHA(q, path_reprs, path_reprs)
      │
      ├─⑤ Pattern Library Retrieval:
      │    pattern_query = cross_out + hist_signal
      │    pattern_out = TPL(pattern_query)                         [NOVEL]
      │
      └─⑥ Three-Signal Fusion + Scoring:
           final_q = fusion_mlp([cross_out, hist_signal, pattern_out])
           scores  = link_head(final_q) · all_ent
                   + w_direct × DistMult(s_dynamic, r)

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
    """Δt = t_query − t_history → log-scaled sinusoidal. Kelajak vaqtlari uchun ham ishlaydi."""
    def __init__(self, dim: int, max_delta: int = 20000):
        super().__init__()
        freqs = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(max_delta) / dim))
        self.register_buffer("freqs", freqs)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        x   = torch.log1p(delta_t.float().clamp(min=0))
        x   = x.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RELATION PROFILE ENCODING  ★ YANGI
# ═══════════════════════════════════════════════════════════════════════════════

class RelationProfile(nn.Module):
    """
    Entity s ning vaqt-og'irlangan relation faoliyat profili.

    profile[r] = Σ_{i: r_i mod R = r} exp(−γ · Δt_i) · valid_i

    Entity-independent: entity indeksi ishlatilmaydi, faqat relation.
    YAGO (10R): 10-dim vektor → barcha faollikni to'liq ifodalaydi.
    WIKI (24R): 24-dim vektor → juda informativ.
    """
    def __init__(self, num_relations: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_relations = num_relations
        self.log_gamma = nn.Parameter(torch.tensor(-2.0))
        self.proj = nn.Sequential(
            nn.Linear(num_relations, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        nb_rel:    torch.Tensor,   # (B, H) relation indices (may include inverse: r+R)
        delta_t:   torch.Tensor,   # (B, H) int
        hist_mask: torch.Tensor,   # (B, H) bool
    ) -> torch.Tensor:             # (B, hidden_dim)
        B = nb_rel.size(0)
        gamma = torch.exp(self.log_gamma)
        decay = torch.exp(-gamma * delta_t.float().clamp(min=0)) * hist_mask.float()

        # Inverse relations → base relation index
        rel_idx = (nb_rel % self.num_relations).clamp(0, self.num_relations - 1)

        profile = torch.zeros(B, self.num_relations, device=nb_rel.device)
        profile.scatter_add_(1, rel_idx, decay)

        # L1-normalize to distribution
        profile = profile / profile.sum(1, keepdim=True).clamp(min=1e-8)
        return self.proj(profile)   # (B, hidden_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TEMPORAL PATTERN LIBRARY  ★ MUTLOQO YANGI
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalPatternLibrary(nn.Module):
    """
    K ta o'rgatiluvchi temporal pattern vektorlari.

    Har bir pattern = abstract vaqtinchalik xulq-atvor shabloni.
    Entity-independent: patternlar entity ga bog'liq emas.
    Query: (relation + history_context) → relevant patternlarni oladi.

    Pattern diversity regularization:
        L_div = E[sim(p_i, p_j)²]  (i≠j)
        Patternlar bir-biridan farq qilsin (collapse oldini oladi)

    Ilmiy yangilik: TKG extrapolation da bunday parametrik pattern
    kutubxonasi hech qayerda qo'llanilmagan.
    """
    def __init__(self, num_patterns: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_patterns = num_patterns
        self.hidden_dim   = hidden_dim
        self.scale        = math.sqrt(hidden_dim)

        # Learnable pattern bank
        self.patterns   = nn.Parameter(torch.randn(num_patterns, hidden_dim) * 0.02)

        # Query/Key projections
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, query_vec: torch.Tensor) -> torch.Tensor:
        """query_vec: (B, H) → (B, H)"""
        q        = self.query_proj(query_vec)           # (B, H)
        k        = self.key_proj(self.patterns)         # (K, H)
        attn     = F.softmax((q @ k.t()) / self.scale, dim=-1)  # (B, K)
        retrieved = attn @ self.patterns                # (B, H)
        return self.out_proj(retrieved)

    def diversity_loss(self) -> torch.Tensor:
        """Pattern collapse oldini olish: off-diagonal similarity minimizatsiya."""
        P_n = F.normalize(self.patterns, dim=-1)        # (K, H) — unit vectors
        sim = P_n @ P_n.t()                             # (K, K)
        off = sim[~torch.eye(self.num_patterns, dtype=torch.bool, device=sim.device)]
        return off.pow(2).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TEMPORAL TRANSFORMER (entity-independent, ikkala paths va history uchun)
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalTransformer(nn.Module):
    """
    Entity-independent Transformer encoder.
    step = [rel_emb, delta_enc]  — entity embedding yo'q!

    Ikkita joyda ishlatiladi:
      - HistTransformer: (B, H_nb, R+D) → (B, hidden_dim)
      - PathEncoder:     (B*P, L, R+D)  → (B*P, hidden_dim)

    DaeMon sequential tawaregate dan farqi:
      - Parallel attention (barcha tarix bir vaqtda)
      - Long-range dependencies (gradient yo'qolmaydi)
      - delta_enc positional info sifatida kiritiladi
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
        max_seq_len:  int   = 128,
        entity_dim:   int   = 0,   # backward-compat, ignored
    ):
        super().__init__()
        step_dim = relation_dim + delta_dim
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

    def forward(
        self,
        rel_emb:   torch.Tensor,   # (B, L, R)
        delta_enc: torch.Tensor,   # (B, L, D)
        mask:      torch.Tensor,   # (B, L) bool — True = valid
    ) -> torch.Tensor:             # (B, hidden_dim)
        B, L, _ = rel_emb.shape
        step  = torch.cat([rel_emb, delta_enc], dim=-1)   # (B, L, R+D)
        x     = self.step_proj(step)
        cls   = self.cls_token.expand(B, -1, -1)
        x     = torch.cat([cls, x], dim=1)                # (B, L+1, H)
        pos   = torch.arange(L + 1, device=x.device)
        x     = x + self.pos_emb(pos).unsqueeze(0)
        cls_m = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        out   = self.transformer(x, src_key_padding_mask=~torch.cat([cls_m, mask], dim=1))
        return self.norm(out)[:, 0, :]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GATED TEMPORAL MEMORY (tawaregate — DaeMon dan)
# ═══════════════════════════════════════════════════════════════════════════════

class GatedTemporalMemory(nn.Module):
    """gate = σ(W[e_s, e_d]);  out = gate⊙tanh(W_h·e_d) + (1−gate)⊙e_s"""
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
            nn.Linear(entity_dim, entity_dim), nn.LayerNorm(entity_dim), nn.GELU()
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
# 8. ORION — ASOSIY MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class ORIONModel(nn.Module):
    """
    ORION: Ontology-aware Relational pattern Inference with Ordered Networks

    Yangi komponentlar:
      TPL   — Temporal Pattern Library: K learnable patterns
      RPE   — Relation Profile Encoding: time-decayed relation activity
      HT    — History Transformer: entity-independent, parallel attention
      RTE   — Relative Temporal Encoding: Δt-based
      3SF   — Three-Signal Fusion: path + history + pattern
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
        num_patterns:    int   = 128,
        dropout:         float = 0.1,
        label_smoothing: float = 0.1,
        w_direct:        float = 0.0,
        w_pattern_div:   float = 0.01,
        use_history:     bool  = False,
        use_diachronic:  bool  = False,
        max_history:     int   = 64,
        # backward-compat ignored
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
        self.num_entities       = num_entities
        self.num_base_relations = num_relations   # ortho_reg uchun kerak
        self.entity_dim         = entity_dim
        self.hidden_dim         = hidden_dim
        self.num_negative       = num_negative
        self.label_smoothing    = label_smoothing
        self.w_direct           = w_direct
        self.w_pattern_div      = w_pattern_div
        self.use_history        = use_history
        self.use_diachronic     = use_diachronic

        # ── Embeddings ────────────────────────────────────────────────────────
        self.ent_emb = nn.Embedding(num_entities,      entity_dim,   padding_idx=0)
        self.rel_emb = nn.Embedding(num_relations * 2, relation_dim)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # ── Relative Temporal Encoding ─────────────────────────────────────────
        self.delta_enc = RelativeTemporalEncoding(delta_dim, max(num_times * 3, 20000))

        # ── Query Encoder ──────────────────────────────────────────────────────
        self.query_encoder = nn.Sequential(
            nn.Linear(entity_dim + relation_dim + delta_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),     nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── History Branch (entity-independent) ───────────────────────────────
        if use_history:
            # Relation Profile: entity's time-decayed relation activity [NOVEL]
            self.relation_profile = RelationProfile(num_relations, hidden_dim, dropout)

            # History Transformer: Transformer over (rel, Δt) [NOVEL]
            self.hist_transformer = TemporalTransformer(
                relation_dim=relation_dim,
                delta_dim=delta_dim,
                hidden_dim=hidden_dim,
                num_heads=min(4, num_heads),
                num_layers=2,
                ffn_dim=hidden_dim * 2,
                dropout=dropout,
                max_seq_len=max_history + 2,
            )

            # Tawaregate: fuse static entity with history context [DaeMon]
            # hist_transformer output projected to entity_dim for gating
            self.hist_to_entity = nn.Linear(hidden_dim, entity_dim)
            self.gate_mem       = GatedTemporalMemory(entity_dim, dropout)

            # Neighborhood context: [s_dynamic, s_emb] → H
            self.nb_ctx   = nn.Sequential(
                nn.Linear(entity_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            )
            self.hist_norm       = nn.LayerNorm(hidden_dim)  # hist_signal normalisation
            self.query_hist_norm = nn.LayerNorm(hidden_dim)  # query+hist injection (alohida!)
        else:
            self.relation_profile = self.hist_transformer = self.hist_to_entity = None
            self.gate_mem = self.nb_ctx = self.hist_norm = self.query_hist_norm = None

        # ── Path Encoder (entity-independent) ─────────────────────────────────
        self.path_encoder = TemporalTransformer(
            relation_dim=relation_dim,
            delta_dim=delta_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_seq_len=16,
        )

        # ── Temporal Pattern Library [NOVEL] ──────────────────────────────────
        self.pattern_lib = TemporalPatternLibrary(num_patterns, hidden_dim, dropout)

        # ── Cross-Path Attention ───────────────────────────────────────────────
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # ── Three-Signal Fusion [NOVEL] ────────────────────────────────────────
        # Signals: cross_out (H) + hist_signal (H) + pattern_out (H) = 3H
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.fusion_norm  = nn.LayerNorm(hidden_dim)
        self.fusion_ffn   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.fusion_norm2 = nn.LayerNorm(hidden_dim)

        # ── Scoring Heads ──────────────────────────────────────────────────────
        self.link_head   = LinkPredHead(hidden_dim, entity_dim, num_entities, dropout)
        self.direct_head = DirectScoringHead(entity_dim, relation_dim, num_entities, dropout) \
                           if (w_direct > 0 or use_direct_scoring) else None

        # ── Diachronic Gating ──────────────────────────────────────────────────
        if use_diachronic:
            self.dia_amp   = nn.Embedding(1, entity_dim)
            self.dia_freq  = nn.Embedding(1, entity_dim)
            self.dia_phase = nn.Embedding(1, entity_dim)
            nn.init.uniform_(self.dia_amp.weight,   -0.1,       0.1)
            nn.init.uniform_(self.dia_freq.weight,   0.5,       2.0)
            nn.init.uniform_(self.dia_phase.weight, -math.pi, math.pi)
            self.num_times_dia = max(num_times, 1)

        # ── Per-relation temperature ───────────────────────────────────────────
        self.rel_temp = nn.Embedding(num_relations, 1)
        nn.init.constant_(self.rel_temp.weight, 1.0)

        self.drop = nn.Dropout(dropout)
        self.register_buffer("all_entity_ids", torch.arange(num_entities, dtype=torch.long))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ent(self, e):   return self.drop(self.ent_emb(e))
    def _rel(self, r):   return self.drop(self.rel_emb(r.clamp(0, self.rel_emb.num_embeddings - 1)))

    def _diachronic(self, e: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not self.use_diachronic: return e
        t_n  = (t.float() / self.num_times_dia).unsqueeze(-1)
        gate = 1.0 + self.dia_amp.weight[0] * torch.sin(
            self.dia_freq.weight[0] * t_n + self.dia_phase.weight[0]
        )
        return e * gate

    def _encode_query(self, s, r, device):
        s_e = self._ent(s)
        r_e = self._rel(r)
        d_z = self.delta_enc(torch.zeros(s.size(0), dtype=torch.long, device=device))
        return self.query_encoder(torch.cat([s_e, r_e, d_z], dim=-1)), s_e, r_e

    def _encode_paths(self, paths, path_masks, query_t):
        # paths: (B, P, L, 3);  path_masks: (B, P, L)
        B, P, L, _ = paths.shape
        pf = paths.view(B * P, L, 3)
        mf = path_masks.view(B * P, L)

        rel = pf[:, :, 1]
        t_h = pf[:, :, 2]
        t_q = query_t.unsqueeze(1).expand(-1, P).contiguous().view(B * P)
        dt  = (t_q.unsqueeze(1) - t_h.float()).clamp(min=0).long()

        r_e = self.rel_emb(rel.clamp(0, self.rel_emb.num_embeddings - 1))
        d_e = self.delta_enc(dt)
        out = self.path_encoder(r_e, d_e, mf)
        return out.view(B, P, -1)

    def _process_history(self, times, history, hist_mask, s_emb, r_emb):
        """
        Returns:
          s_dynamic  (B, E): gated entity representation
          hist_signal (B, H): combined history + profile signal
        """
        hist_rel = history[:, :, 1]
        hist_t   = history[:, :, 2]
        delta_t  = (times.unsqueeze(1) - hist_t.float()).clamp(min=0).long()   # (B, H)
        delta_enc = self.delta_enc(delta_t)                                     # (B, H, D)

        nb_r  = self.rel_emb(hist_rel.clamp(0, self.rel_emb.num_embeddings - 1))  # (B, H, R)

        # [A] Relation Profile — entity's relation activity signature [NOVEL]
        profile_enc = self.relation_profile(hist_rel, delta_t, hist_mask)      # (B, H_dim)

        # [B] History Transformer — parallel attention over all history [NOVEL]
        hist_out = self.hist_transformer(nb_r, delta_enc, hist_mask)           # (B, H_dim)

        # [C] Tawaregate — DaeMon: gate static entity with history context
        hist_e   = self.hist_to_entity(hist_out)                               # (B, E)
        s_dynamic = self.gate_mem(s_emb, hist_e)                               # (B, E)

        # Neighborhood context
        nb_ctx_out = self.nb_ctx(torch.cat([s_dynamic, s_emb], dim=-1))        # (B, H_dim)

        # Combined history signal: Transformer context + relation profile
        hist_signal = self.hist_norm(nb_ctx_out + profile_enc)                 # (B, H_dim)
        return s_dynamic, hist_signal

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

        device = subjects.device
        q, s_emb, r_emb = self._encode_query(subjects, relations, device)

        # ── History branch ────────────────────────────────────────────────────
        s_dynamic   = None
        hist_signal = torch.zeros(subjects.size(0), self.hidden_dim, device=device)
        if self.use_history and history is not None:
            s_dynamic, hist_signal = self._process_history(
                times, history, hist_mask, s_emb, r_emb
            )
            q = self.query_hist_norm(q + hist_signal)  # alohida norm — hist_norm bilan aralashmasin

        # ── Path encoding ─────────────────────────────────────────────────────
        path_reprs = self._encode_paths(paths, path_masks, times)   # (B, P, H)

        # ── Cross-path attention ──────────────────────────────────────────────
        q_exp       = q.unsqueeze(1)
        attn_out, _ = self.cross_attn(q_exp, path_reprs, path_reprs)
        cross_out   = self.cross_norm(q_exp + attn_out).squeeze(1)  # (B, H)

        # ── Pattern Library retrieval [NOVEL] ─────────────────────────────────
        # Query for patterns: path-attended query + history signal
        pattern_query = cross_out + hist_signal                     # (B, H)
        pattern_out   = self.pattern_lib(pattern_query)             # (B, H)

        # ── Three-Signal Fusion [NOVEL] ───────────────────────────────────────
        fused   = self.fusion_mlp(torch.cat([cross_out, hist_signal, pattern_out], dim=-1))
        fused   = self.fusion_norm(fused + cross_out)               # residual
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

        # ortho_reg: faqat asosiy relatsiyalar, normalize qilingan (katta tensordan qochish)
        n_base = min(self.num_base_relations, 64)   # Ko'pi bilan 64 ta (ICEWS tezligi uchun)
        rel_w  = F.normalize(self.rel_emb.weight[:n_base], dim=-1)
        tmp    = rel_w @ rel_w.t()                  # (n_base, n_base)
        eye_n  = torch.eye(n_base, device=device)
        ortho  = (tmp - eye_n).pow(2).mean()        # MSE, norm2 emas — stabil

        # Pattern diversity regularization [NOVEL]
        pattern_div = self.pattern_lib.diversity_loss()

        losses = {
            "link":        link_loss,
            "contrastive": pattern_div,   # pattern diversity (trainer contrastive slot)
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

        device = subjects.device
        q, s_emb, r_emb = self._encode_query(subjects, relations, device)

        s_dynamic   = None
        hist_signal = torch.zeros(subjects.size(0), self.hidden_dim, device=device)
        if self.use_history and history is not None:
            s_dynamic, hist_signal = self._process_history(
                times, history, hist_mask, s_emb, r_emb
            )
            q = self.query_hist_norm(q + hist_signal)  # alohida norm

        path_reprs  = self._encode_paths(paths, path_masks, times)
        q_exp       = q.unsqueeze(1)
        attn_out, _ = self.cross_attn(q_exp, path_reprs, path_reprs)
        cross_out   = self.cross_norm(q_exp + attn_out).squeeze(1)

        pattern_query = cross_out + hist_signal
        pattern_out   = self.pattern_lib(pattern_query)

        fused   = self.fusion_mlp(torch.cat([cross_out, hist_signal, pattern_out], dim=-1))
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
STORMModel    = ORIONModel
CATREModel    = ORIONModel
EliteTKGModel = ORIONModel
