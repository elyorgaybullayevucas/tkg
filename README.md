# TKG Elite Model
## Contrastive Temporal Path Learning for TKG Completion

---

## Mukammal Arxitektura

```
┌─────────────────────────────────────────────────────────────────┐
│                    ELITE TKG MODEL                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. TEMPORAL EMBEDDING MODULE                                   │
│     ├─ Entity Embedding (Xavier init)           [E × entity_dim]│
│     ├─ Relation Embedding (+ inverse relations) [2R × rel_dim]  │
│     ├─ Time: Learned + Sinusoidal → concat → proj [T × time_dim]│
│     └─ Temporal Entity: e_t = e + sigmoid(W[e;t])·tanh(W_t·t)  │
│                                                                 │
│  2. QUERY ENCODER (s, r, t) → q                                 │
│     └─ [s_t; r; t] → Linear → LN → GELU → Linear → H           │
│                                                                 │
│  3. PATH ENCODER (Transformer)                                  │
│     ├─ CLS token + Positional encoding                          │
│     ├─ Multi-head Self-Attention (Pre-LN)                       │
│     └─ CLS output → path representation                        │
│                                                                 │
│  4. CROSS-ATTENTION FUSION                                      │
│     └─ q attends path_reprs → final_q (B, H)                   │
│                                                                 │
│  5. LINK PREDICTION HEAD                                        │
│     ├─ Highway network: gate * f(q) + (1-gate) * q             │
│     ├─ Project → entity_dim space                               │
│     ├─ Dot-product with ALL entity embeddings                   │
│     └─ Relation-specific temperature scaling                    │
│                                                                 │
│  6. MoCo v2 CONTRASTIVE                                         │
│     ├─ Online encoder (gradient flows)                          │
│     ├─ Momentum encoder (EMA, no gradient)                      │
│     ├─ Negative Queue (8192 entries)                            │
│     └─ InfoNCE loss, temperature=0.3                            │
│                                                                 │
│  7. SELF-ADVERSARIAL NEGATIVE SAMPLING                          │
│     └─ Weight negatives by softmax(scores) — hard negatives     │
│                                                                 │
│  LOSS = 1.0·L_link + 0.5·L_contrastive + 0.5·L_self_adv        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Nima yangi?

| Xususiyat | Eski model | Elite model |
|-----------|-----------|-------------|
| Time embedding | Learned only | Learned + Sinusoidal |
| Entity representation | Static | Temporal-gated |
| Path encoder | GRU | Transformer (Pre-LN) |
| Query-Path fusion | Mean pooling | Cross-attention |
| Contrastive | Simple InfoNCE | MoCo v2 + Queue |
| Negative sampling | Random | Mixed type-constrained |
| Loss | 2 component | 3 component + label smoothing |
| Temperature | 0.07 (xato!) | 0.3 (optimal) |
| LR Schedule | SequentialLR | OneCycleLR |
| Mixed precision | ❌ | ✅ FP16 |
| Relation scaling | ❌ | ✅ per-relation temperature |

---

## O'rnatish va Ishlatish

```bash
# Dataset ko'chirish
cp -r /path/to/original/data/ICEWS18 data/
cp -r /path/to/original/data/ICEWS14 data/
# ... boshqa datasetlar

# ICEWS18 (default)
python main.py --dataset ICEWS18

# ICEWS14 (kichikroq, tezroq)
python main.py --dataset ICEWS14 --epochs 50

# GPU xotirasi kam bo'lsa
python main.py --dataset ICEWS18 \
    --batch_size 256 \
    --hidden_dim 256 \
    --entity_dim 128 \
    --queue_size 4096

# To'liq konfiguratsiya
python main.py \
    --dataset ICEWS18 \
    --entity_dim 256 \
    --hidden_dim 512 \
    --num_heads 8 \
    --num_layers 2 \
    --num_paths 8 \
    --num_negative 256 \
    --temperature 0.3 \
    --batch_size 512 \
    --epochs 50 \
    --lr 3e-4

# Davom ettirish
python main.py --resume checkpoints/ICEWS18_best.pt
```

---

## Kutilayotgan natijalar (ICEWS18)

| Metrika | Taxminiy |
|---------|---------|
| MRR     | 0.38–0.45 |
| Hits@1  | 0.28–0.35 |
| Hits@3  | 0.44–0.52 |
| Hits@10 | 0.60–0.68 |

---

## Fayl tuzilmasi

```
tkg_elite/
├── main.py                       ← Ishga tushirish
├── config.py                     ← Sozlamalar
├── data/
│   ├── dataset.py                ← TKGEliteDataset
│   ├── datamodule.py             ← DataLoader
│   ├── ICEWS14/ ICEWS18/ WIKI/
│   ├── YAGO/ YAGOs/ GDELT/
├── models/
│   └── elite_tkg_model.py       ← ASOSIY MODEL (780 qator)
├── trainers/
│   └── trainer.py               ← Mixed precision trainer
└── utils/
    ├── logging.py
    ├── metrics.py                ← MRR, Hits@K
    └── paths.py                  ← Temporal BFS path sampler
```
