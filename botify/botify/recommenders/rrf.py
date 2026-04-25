"""
Reciprocal Rank Fusion (RRF) Recommender
=========================================
Combines user-based HSTU recommendations with session-aware SasRec-I2I
recommendations using the Reciprocal Rank Fusion scoring formula:

    score(d) = Σ_r  1 / (k + rank_r(d))

This gives every candidate a score that rewards high placement in *any* of
the contributing ranked lists, which is provably better than picking from a
single source when the sources are complementary (HSTU captures long-term
user taste; SasRec captures short-term session context).

References
----------
Cormack, Clarke, and Buettcher (2009) "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods".
"""

import json
import pickle
from collections import defaultdict
from typing import List, Tuple

from .recommender import Recommender


class RRFRecommender(Recommender):
    """
    ML ensemble recommender that fuses two neural ranking models:

    1. **HSTU** (Hierarchical Sequential Transduction Units) – user-based
       pre-computed ranked lists.  Good at capturing *who* the user is.

    2. **SasRec-I2I** – item-to-item recommendations from a sequential
       attention model.  Good at capturing *what the session feels like*.

    Fusion is done with RRF (k=60 is the standard constant from the paper).
    The top-N unseen candidate with the highest fused score is returned.
    """

    def __init__(
        self,
        listen_history_redis,
        hstu_redis,
        i2i_redis,
        catalog,
        fallback,
        rrf_k: int = 60,
        hstu_top: int = 100,
        i2i_top: int = 50,
        max_anchors: int = 5,
    ):
        self.listen_history_redis = listen_history_redis
        self.hstu_redis = hstu_redis
        self.i2i_redis = i2i_redis
        self.catalog = catalog
        self.fallback = fallback
        self.rrf_k = rrf_k
        self.hstu_top = hstu_top
        self.i2i_top = i2i_top
        self.max_anchors = max_anchors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_history(user)
        seen_tracks: set = {track for track, _ in history}

        scores: dict = defaultdict(float)

        # ---- Source 1: HSTU user-based ranked list ----
        self._add_hstu_scores(user, seen_tracks, scores)

        # ---- Source 2: SasRec-I2I for best anchor tracks ----
        self._add_i2i_scores(history, seen_tracks, scores)

        if scores:
            best = max(scores, key=lambda t: scores[t])
            return best

        return self.fallback.recommend_next(user, prev_track, prev_track_time)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_hstu_scores(self, user: int, seen_tracks: set, scores: dict) -> None:
        """Add RRF scores from the HSTU user-based ranked list."""
        data = self.hstu_redis.get(user)
        if data is None:
            return
        hstu_tracks = list(self.catalog.from_bytes(data))
        rank = 0
        for track in hstu_tracks:
            if track in seen_tracks:
                continue
            if rank >= self.hstu_top:
                break
            scores[track] += 1.0 / (self.rrf_k + rank)
            rank += 1

    def _add_i2i_scores(
        self,
        history: List[Tuple[int, float]],
        seen_tracks: set,
        scores: dict,
    ) -> None:
        """Add RRF scores from SasRec-I2I for the most-listened anchor tracks."""
        if not history:
            return

        # Aggregate listen time per track
        track_time: dict = defaultdict(float)
        for track, t in history:
            track_time[track] += t
        total_time = sum(track_time.values()) or 1.0

        # Select top anchors by listen time
        anchors = sorted(track_time, key=lambda t: -track_time[t])[: self.max_anchors]

        for anchor in anchors:
            data = self.i2i_redis.get(anchor)
            if data is None:
                continue
            i2i_tracks = pickle.loads(data)
            anchor_weight = track_time[anchor] / total_time  # proportional contribution

            rank = 0
            for raw_track in i2i_tracks:
                candidate = int(raw_track)
                if candidate in seen_tracks:
                    continue
                if rank >= self.i2i_top:
                    break
                scores[candidate] += anchor_weight / (self.rrf_k + rank)
                rank += 1

    def _load_history(self, user: int) -> List[Tuple[int, float]]:
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history