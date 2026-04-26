"""
Reciprocal Rank Fusion (RRF) Recommender
"""

import json
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

from .recommender import Recommender


class RRFRecommender(Recommender):
    """
    Multi-source RRF recommender.

    Sources
    -------
    - SasRec-I2I: item-transition lists for recent anchor tracks
    - HSTU (optional): pre-computed per-user ranked list

    Anchor weighting
    ----------------
    anchor_weight[i] = recency_decay^i * listen_time[i]

    where i=0 is the MOST RECENT track in the session.
    This ensures that:
      * very recent tracks get a high recency bonus
      * tracks the user actually listened to (high %) contribute more
      * skipped tracks (low %) are down-weighted automatically

    RRF score
    ---------
    score(c) = Σ_anchor  anchor_weight / (k + rank_in_i2i_list(c))
             + hstu_weight / (k + rank_in_hstu_list(c))   [if available]
    """

    def __init__(
        self,
        listen_history_redis,
        i2i_redis,
        catalog,
        fallback,
        hstu_redis=None,
        rrf_k: int = 5,
        i2i_top: int = 40,
        hstu_top: int = 80,
        max_anchors: int = 5,
        recency_decay: float = 0.75,
        hstu_weight: float = 0.4,
        min_listen_time: float = 0.1,
    ):
        self.listen_history_redis = listen_history_redis
        self.i2i_redis = i2i_redis
        self.hstu_redis = hstu_redis
        self.catalog = catalog
        self.fallback = fallback

        self.rrf_k = rrf_k
        self.i2i_top = i2i_top
        self.hstu_top = hstu_top
        self.max_anchors = max_anchors
        self.recency_decay = recency_decay
        self.hstu_weight = hstu_weight
        self.min_listen_time = min_listen_time

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_history(user)
        seen: set = {track for track, _ in history}

        scores: Dict[int, float] = defaultdict(float)

        self._score_i2i(history, seen, scores)

        if self.hstu_redis is not None:
            self._score_hstu(user, seen, scores)

        if scores:
            return max(scores, key=lambda t: scores[t])

        # Fallback: let the standard I2I recommender handle it
        return self.fallback.recommend_next(user, prev_track, prev_track_time)

    # ------------------------------------------------------------------
    # Scoring sources
    # ------------------------------------------------------------------

    def _score_i2i(
        self,
        history: List[Tuple[int, float]],
        seen: set,
        scores: Dict[int, float],
    ) -> None:
        """
        For each anchor track (most recent first), fetch its I2I list and
        add RRF scores weighted by recency and listen time.

        history[0] = MOST RECENT (lpush → lrange order: newest first).
        """
        if not history:
            return

        # Deduplicate: keep only the first (most recent) occurrence of each track
        seen_anchors: set = set()
        anchors: List[Tuple[int, float]] = []
        for track, t in history:
            if track not in seen_anchors:
                anchors.append((track, t))
                seen_anchors.add(track)
            if len(anchors) >= self.max_anchors:
                break

        # Compute unnormalised anchor weights
        raw_weights = []
        for i, (track, listen_time) in enumerate(anchors):
            recency = self.recency_decay ** i
            w = recency * max(listen_time, self.min_listen_time)
            raw_weights.append(w)

        total_w = sum(raw_weights) or 1.0

        for (anchor_track, _), raw_w in zip(anchors, raw_weights):
            w = raw_w / total_w

            data = self.i2i_redis.get(anchor_track)
            if data is None:
                continue

            candidates = pickle.loads(data)
            rank = 0
            for raw_track in candidates:
                candidate = int(raw_track)
                if candidate in seen:
                    continue
                if rank >= self.i2i_top:
                    break
                scores[candidate] += w / (self.rrf_k + rank)
                rank += 1

    def _score_hstu(
        self,
        user: int,
        seen: set,
        scores: Dict[int, float],
    ) -> None:
        """Add RRF scores from the HSTU user-level ranked list."""
        data = self.hstu_redis.get(user)
        if data is None:
            return

        hstu_tracks = list(self.catalog.from_bytes(data))
        rank = 0
        for track in hstu_tracks:
            if track in seen:
                continue
            if rank >= self.hstu_top:
                break
            scores[track] += self.hstu_weight / (self.rrf_k + rank)
            rank += 1

    # ------------------------------------------------------------------
    # History loading
    # ------------------------------------------------------------------

    def _load_history(self, user: int) -> List[Tuple[int, float]]:
        """
        Load listen history from Redis.
        Returns list of (track_id, listen_time) ordered newest-first
        (lpush means index 0 = most recently pushed item).
        """
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)
        history = []
        for raw in raw_entries:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history
