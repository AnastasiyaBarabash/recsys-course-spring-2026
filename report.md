# Report:

## Abstract

I propose a **Reciprocal Rank Fusion (RRF)** ensemble that combines two complementary neural models already available in the botify service: the user-based **HSTU** ranker (captures long-term taste) and the item-based **SasRec-I2I** model (captures short-term session context). Each candidate track receives a fused score `Σ w/(k+rank)` over both sources; the highest-scoring unseen track is returned. An honest 50/50 A/B experiment (`RRF_VS_SASREC`) shows a statistically significant improvement in `mean_session_time` over the SasRec-I2I baseline.

---

## Details

HSTU knows *who* the user is; SasRec-I2I knows *what the session feels like right now*. Neither alone is optimal. RRF exploits the complementarity: a track ranked highly by **both** models is almost certainly relevant, while a track ranked highly by only one gets a proportionally smaller boost. This directly follows from the Condorcet Jury Theorem applied to ranked retrieval.

### Architecture

```
Request: POST /next/{user}
              │
              ▼
  ┌───────────────────────────────────────────────────────┐
  │                    RRFRecommender                     │
  │                                                       │
  │  listen history  ──>  top-5 anchor tracks             │
  │  (Redis DB 2)         by cumulative listen time       │
  │                              │                        │
  │                              ▼                        │
  │                    SasRec-I2I lists          score    │
  │                    (Redis DB 4, top-50) ──>  +=       │
  │                                           w/(60+rank) │
  │                                                       │
  │  HSTU user list  ───────────────────────>  score +=   │
  │  (Redis DB 5, top-100)                    1/(60+rank) │
  │                                                       │
  │  argmax(score) over unseen tracks                     │
  │  fallback → Random                                    │
  └───────────────────────────────────────────────────────┘
              │
              ▼
       recommendation
```

**Scoring formula** (`k = 60`, standard constant from Cormack et al. 2009):

```
score(track) = Σ_source  w_source / (60 + rank_source(track))
```

- **HSTU weight** `w = 1` (global user ranking, pre-computed by the model)
- **SasRec-I2I weight** `w = listen_time(anchor) / total_session_time` — anchors with higher listen fraction contribute more

Only tracks **not yet heard** in the current session are eligible.

### Experiment design

| Parameter | Value |
|-----------|-------|
| Experiment name | `RRF_VS_SASREC` |
| Split | 50/50 hash on `user_id` (MurmurHash3, deterministic) |
| Control | SasRec-I2I |
| Treatment | RRF (HSTU + SasRec-I2I) |
| Metric | `mean_session_time` (sum of per-track listen fractions per session) |
| Statistical test | One-sided Welch's t-test (H₁: treatment > control), α = 0.05 |
