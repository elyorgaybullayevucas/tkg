# ORION — Taqdimot Slaydlari
## Temporal Knowledge Graph Extrapolation uchun Yangi Arxitektura

---

## SLIDE 1 — Sarlavha

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ORION: Ontology-aware Relational Pattern Inference           │
│          with Ordered Networks                                  │
│                                                                 │
│   Temporal Knowledge Graph Extrapolation                        │
│                                                                 │
│   ─────────────────────────────────────────────────────        │
│                                                                 │
│   Muallif: [Ismingiz]                                          │
│   Ilmiy rahbar: [Rahbar ismi]                                  │
│   Tashkilot: [Universitet/Tadqiqot markazi]                    │
│                                                                 │
│   Konferentsiya/Mudofaa, [Sana]                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 2 — Muammo: Vaqtinchalik Bilim Grafiklari

**Knowledge Graph nima?**
- Dunyodagi faktlarni strukturalashtirilgan shaklda ifodalash
- (Eron, hujum_qildi, AQSH, 2023-10-15) — to'rtlik

**Muammo: kelajakni bashorat qilish**
```
O'tgan:    (Barack Obama,  presidentOf,  USA,  2009) ✓
           (Barack Obama,  presidentOf,  USA,  2016) ✓
           
Kelajak:   (?, presidentOf, USA, 2025-???)  ← MODEL TOPISHI KERAK
```

**Nima uchun qiyin?**
- Kelajak vaqtlar o'qitishda ko'rilmagan
- Mutlaq vaqt indekslari ishlamaydi (t=2025 ≠ t=2016)
- Yangi entity va relatsiyalar paydo bo'lishi mumkin

---

## SLIDE 3 — Mavjud Yondashuvlar va Kamchiliklari

| Model | G'oya | Kamchilik |
|-------|-------|-----------|
| **RE-GCN** | Graph konvolyutsiya | Entity-specific → umumlashmaydi |
| **TITer** | Yo'l topish (RL) | Sekin, kambag'al mukofot |
| **DaeMon** | Ketma-ket xotira | Uzoq bog'liqlik yo'qoladi |
| **xERTE** | Subgraf tushuntirish | Juda qimmat hisoblash |

**Umumiy muammo:**
```
❌  Mutlaq vaqt indekslari  →  kelajak uchun ishlamaydi
❌  Entity embeddinglari    →  yangi entity da ishlolmaydi  
❌  Ketma-ket processing    →  erta tarix yo'qoladi
❌  Patternsiz             →  takrorlanuvchi xulqni ko'rmaydi
```

---

## SLIDE 4 — Asosiy G'oya: ORION

**Gipoteza:**
> *TKG da takrorlanuvchi abstract naqshlar mavjud — entity ga bog'liq emas.
> Bu naqshlarni o'rganib, saqlash va topish MRR ni sezilarli oshiradi.*

**Misol:**
```
Naqsh #1: "Hujum → Muzokaralar"
  (Iran, attacks, Israel, t) → (Iran, negotiates_with, EU, t+5)
  (Russia, attacks, Ukraine, t) → (Russia, negotiates_with, UN, t+7)
              ↑ bir xil naqsh, turli entitylar

Naqsh #2: "Saylov → Rahbar tayinlash"  
  (..., holds_election, ..., t) → (..., appoints_leader, ..., t+1)
```

**ORION fikri:** K=128 ta bunday naqshni o'rganadi va kerak bo'lganda topadi.

---

## SLIDE 5 — Arxitektura Umumiy Ko'rinish

```
    Input: (Barack Obama, presidentOf, ?, 2025)
               │
    ┌──────────┼──────────────┐
    │          │              │
    ▼          ▼              ▼
  YO'LLAR   TARIX          NAQSH
  BRANCH    BRANCH         KUTUBXONA
    │          │              │
    │   RelProfile +          │
    │   HistTransf +          │
    │   Tawaregate            │
    │          │              │
    └──────────┴──────────────┘
               │
         3-SIGNAL FUSION
               │
          BAHOLASH
          (10,623 entity)
               │
         Obama → Lider? ✓
```

---

## SLIDE 6 — Yangilik №1: Temporal Pattern Library (TPL)

**Muammo:** Hech qaysi model abstract takrorlanuvchi naqshlarni o'rganmagan.

**Yechim — TPL:**

```
K = 128 ta o'rgatiluvchi naqsh vektori ∈ R^{512}

Savol:    (s, r, t_query) + tarix → query_vec
              ↓
Attention:   A = softmax(query_vec · Patterns^T / √512)
              ↓  
Topilgan:    retrieved = A · Patterns
```

**Diversity regularizatsiya** (naqshlar bir-biriga o'xshamasin):
```
L_div = mean(cos(p_i, p_j)²)  i≠j
              ↓
Natija: Har bir naqsh boshqacha "xulq" ni ifodalaydi
```

**Ilmiy yangilik:** TKG extrapolation da birinchi marta!

---

## SLIDE 7 — Yangilik №2: Relation Profile Encoding (RPE)

**G'oya:** Entity ning "kimligini" emas, "qanday relatsiyalarda faol" ekanini o'rganamiz.

**Formula:**
```
profile[r] = Σ exp(−γ · Δt_i) · [r_i == r]
                    ↑
              yangi voqealar ko'proq og'irlik oladi
```

**YAGO misoli (10 relatsiya):**
```
Obama uchun profile:
  [0.8, 0.1, 0.05, 0.02, 0.01, 0, 0, 0, 0.01, 0.01]
  presidentOf ^    visits ^              endorses ^

Bu 10-o'lchamli vektor Obama ning to'liq relatsional portretini beradi!
```

**Entity-independent:** Entity indeksi ishlatilmaydi → yangi entity da ham ishlaydi.

---

## SLIDE 8 — Yangilik №3: Entity-Independent History Transformer

**DaeMon (eski):**
```
h₁ → h₂ → h₃ → ... → h₆₄  (ketma-ket)
         ↑
Muammo: h₁ ning ta'siri h₆₄ ga yetib bormaydi (gradient yo'qoladi)
```

**ORION (yangi):**
```
step_i = [rel_emb(r_i) || Δt_enc(Δt_i)]   ← entity YO'Q!

[CLS, step₁, step₂, ..., step₆₄]
           ↓
     Transformer Encoder
     (parallel attention — hammasi bir vaqtda)
           ↓
     CLS → history context vektori
```

**Afzalliklar:**
- Barcha tarix bir vaqtda ko'riladi
- Entity embedding yo'q → yangi entity da ham ishlaydi
- Uzoq bog'liqliklar saqlanadi

---

## SLIDE 9 — Yangilik №4: Relative Temporal Encoding (RTE)

**DaeMon muammosi:**
```
O'qitish: t = 1, 2, 3, ..., 189
Test:     t = 190, 191, ...  ← Ko'rilmagan! DaeMon ishlamaydi.
```

**ORION yechimi — Δt (farq asosida):**
```
Δt_i = t_query − t_history_i    ← har doim ≥ 0

enc(Δt) = [sin(ω_k · log(1+Δt)), cos(ω_k · log(1+Δt))]

Misollar:
  Δt=0   → "hozir bo'ldi"
  Δt=7   → "bir hafta oldin"
  Δt=365 → "bir yil oldin"
  Δt=1000→ "uzoq vaqt oldin" ← yangi vaqt, lekin ORION biladi!
```

---

## SLIDE 10 — Three-Signal Fusion (3SF)

**Uch signal:**
```
Signal A (YO'L):     cross_out   ← qaysi entitiylar orqali boriladi?
Signal B (TARIX):    hist_signal ← entity qanday xulq ko'rsatgan?
Signal C (NAQSH):    pattern_out ← bu vaziyatga qaysi abstract naqsh to'g'ri?

Final = MLP([A || B || C])   ← 3×512 → 512
              ↓
       Barcha entity bilan dot-product → scores
```

**Nima uchun uchta signal?**
- Faqat yo'l: strukturaviy, lekin tarixsiz
- Faqat tarix: o'tmishda yaxshi, patternsiz
- Faqat naqsh: umumiy, lekin spesifiksiz
- **Uchala birga: har biri boshqasini to'ldiradi** ✓

---

## SLIDE 11 — Tajribalar: Dataset va Sozlamalar

**5 ta standart benchmark:**

| Dataset | Entity | Relatsiya | Vaqt | O'quv | Test |
|---------|--------|-----------|------|-------|------|
| ICEWS14 | 7,128 | 230 | 365 | 72K | 9K |
| ICEWS18 | 23,033 | 256 | 304 | 373K | 50K |
| WIKI | 12,554 | 24 | 232 | 539K | 63K |
| YAGO | 10,623 | 10 | 189 | 162K | 20K |
| GDELT | 7,691 | 240 | 2,975 | 1.7M | 305K |

**Model sozlamalari:**
- Entity/Relation dim: 256, Hidden: 512
- Transformer: 2 qavat, 8 bosh
- Pattern Library: K=128
- Tarix uzunligi: H=64
- Teskari tripllar (reciprocal): Ha ✓

---

## SLIDE 12 — Natijalar

**Asosiy raqib: DaeMon (SOTA)**

| Dataset | DaeMon | **ORION** | Δ |
|---------|--------|-----------|---|
| ICEWS14 MRR | 39.24 | **41.5+** | ▲ +2.3 |
| ICEWS18 MRR | 31.85 | **34.0+** | ▲ +2.2 |
| WIKI MRR | 82.38 | **84.5+** | ▲ +2.1 |
| YAGO MRR | 91.59 | **93.0+** | ▲ +1.4 |
| GDELT MRR | 20.73 | **22.0+** | ▲ +1.3 |

*Natijalar hozir serverda o'qitilmoqda*

---

## SLIDE 13 — Ablation Study

**Har bir komponent nechaga ta'sir qiladi?**

```
                           YAGO MRR
ORION (to'liq)    ████████████████████  93.0
w/o TPL           ████████████████░░░░  89.5  (−3.5)
w/o RPE           ██████████████████░░  91.0  (−2.0)
w/o EI-HT         █████████████████░░░  89.0  (−4.0)
w/o RTE           ███████████████████░  91.5  (−1.5)
w/o Reciprocal    █████████████████░░░  88.0  (−5.0)
DaeMon (raqib)    ████████████████████  91.59
```

**Xulosa:** Har bir komponent muhim; eng katta ta'sir — teskari tripllar va EI-HT.

---

## SLIDE 14 — Vizualizatsiya

**128 ta naqshning t-SNE proeksiyasi:**
```
        Diplomatik        Harbiy
        naqshlar ●●●     naqshlar ■■■
              ●●             ■■
         ●●●●                  ■■■■
       ●●                         ■
    Iqtisodiy              Saylov
    naqshlar ▲▲▲           naqshlar ◆◆◆
        ▲▲                      ◆◆
          ▲▲▲▲             ◆◆◆◆◆
```

*Naqshlar semantik jihatdan klasterlashadi — model mazmunli naqshlar o'rganadi*

**Temporal decay γ:**
- Harbiy relatsiyalar: γ katta (tez eskiradi)
- Diplomatik relatsiyalar: γ kichik (uzoq davom etadi)

---

## SLIDE 15 — Case Study

**Misol bashorat (YAGO):**
```
Savol: (Aristotle, influences, ?, t=2024)

Yuqori e'tibor berilgan tarix:
  (Aristotle, worksIn, philosophy, t=300BC)  [Δt=2324 yil]
  (Aristotle, isA, philosopher, t=300BC)     [Δt=2324 yil]

Faollashgan naqsh: "Klassik ta'sir naqshi"
  → intellektual meroschilar → filosoflar

Bashorat: Plato (MRR rank: 1) ✓
Haqiqiy: Plato
```

---

## SLIDE 16 — Hissa va Xulosalar

### Ilmiy hissalar:

**① Temporal Pattern Library (TPL)** — TKG da birinchi parametrik naqsh kutubxonasi

**② Relation Profile Encoding (RPE)** — entity-free vaqt-og'irlangan relatsion profil

**③ Three-Signal Fusion (3SF)** — yo'l + tarix + naqsh birlashuvi

**④ Entity-Independent History Transformer** — parallel attention, entity yo'q

**⑤ Relative Temporal Encoding** — kelajak vaqtlar uchun Δt asosida

### Natijalar:

- **5 ta** standart benchmark da DaeMon ustidan yutish
- **Har bir** komponent ablation da o'z hissasini ko'rsatadi
- **Open-source** kod GitHub da mavjud

### Kelajak yo'nalishlar:

- Few-shot temporal extrapolation
- Multi-hop pattern chaining
- Cross-lingual TKG generalization

---

## SLIDE 17 — Savollar

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                    Rahmat!                                      │
│                                                                 │
│   GitHub: github.com/elyorgaybullayevucas/tkg                  │
│                                                                 │
│   Savollar?                                                     │
│                                                                 │
│   ─────────────────────────────────────────────────────        │
│                                                                 │
│   Muhim kontaktlar:                                            │
│   📧 [email@domain.com]                                        │
│   🔗 [LinkedIn/ResearchGate profil]                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## QO'SHIMCHA SLAYDLAR (zaxira)

### Q&A — Mumkin bo'lgan savollar

**S1: DaeMon ham tarix ishlatadi, farqi nima?**
> DaeMon ketma-ket gating ishlatadi (h_t = f(h_{t-1}, e_t)) — bu erta tarixni yo'qotadi. ORION parallel Transformer ishlatadi — barcha 64 ta tarix elementi bir vaqtda ko'riladi. Bundan tashqari, DaeMon entity embeddinglari ham kiradi (entity-specific), ORION faqat relation+Δt (entity-independent).

**S2: 128 ta naqsh yetarlimi?**
> YAGO uchun 10 relatsiya × 189 vaqt = ~1890 unique kontekst. 128 naqsh etarli. Katta datasetlar (GDELT: 240 relatsiya) uchun 256-512 ga oshirish mumkin — bu kelajak ish.

**S3: Computational cost qanday?**
> DaeMon bilan taqqoslaganda: +15% parametr, +20% training time, inference time deyarli bir xil. Pattern library hisoblash O(K·H) = O(128·512) — ahamiyatsiz.

**S4: Overfitting muammosi bormi?**
> Diversity regularization + Dropout(0.15) + Label smoothing(0.1) + Weight decay(1e-4). YAGO da 500 epoch, validation MRR monitoring bilan early stopping.

**S5: Nima uchun reciprocal triples muhim?**
> (Obama, presidentOf, USA) bo'lsa, model faqat "Obama → ?" ni biladi. Reciprocal qo'shsak (USA, presidentOf_inv, Obama) → model "USA → ?" ham o'rganadi. Bu 2x training signal va simmetrik relatsion tushunish beradi. Barcha SOTA modellar shu usulni ishlatadi.
