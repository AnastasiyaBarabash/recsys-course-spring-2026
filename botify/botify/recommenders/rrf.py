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
from typing import List, Tuple, Dict

from .recommender import Recommender


class RRFRecommender(Recommender):
    """
    ML ensemble recommender that fuses two neural ranking models:

    1. Candidate generation from:
        - HSTU (user model)
        - SasRec-I2I (item transitions)

    2. Feature-based reranking:
        - i2i rank score
        - hstu rank score
        - recency
        - frequency

    Final selection: argmax over weighted sum of features.
    """

    def __init__(
        self,
        listen_history_redis,
        hstu_redis,
        i2i_redis,
        catalog,
        fallback,
        hstu_top: int = 100,
        i2i_top: int = 50,
        max_anchors: int = 3,
        w_i2i: float = 1.0,
        w_hstu: float = 0.5,
        w_recency: float = 0.3,
        w_freq: float = 0.2,
    ):
        self.listen_history_redis = listen_history_redis
        self.hstu_redis = hstu_redis
        self.i2i_redis = i2i_redis
        self.catalog = catalog
        self.fallback = fallback

        self.hstu_top = hstu_top
        self.i2i_top = i2i_top
        self.max_anchors = max_anchors

        self.w_i2i = w_i2i
        self.w_hstu = w_hstu
        self.w_recency = w_recency
        self.w_freq = w_freq

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        history = self._load_history(user)
        if not history:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        seen_tracks = {t for t, _ in history}

        features = defaultdict(lambda: {
            "i2i": 0.0,
            "hstu": 0.0,
            "recency": 0.0,
            "freq": 0.0,
        })

        self._add_i2i_features(history, seen_tracks, features)
        self._add_hstu_features(user, seen_tracks, features)
        self._add_behavior_features(history, features)

        if not features:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        best_track = None
        best_score = -1e9

        for track, f in features.items():
            score = (
                self.w_i2i * f["i2i"]
                + self.w_hstu * f["hstu"]
                + self.w_recency * f["recency"]
                + self.w_freq * f["freq"]
            )

            if score > best_score:
                best_score = score
                best_track = track

        return best_track if best_track is not None else \
            self.fallback.recommend_next(user, prev_track, prev_track_time)

    # ------------------------------------------------------------------
    # Feature builders
    # ------------------------------------------------------------------

    def _add_i2i_features(
        self,
        history: List[Tuple[int, float]],
        seen_tracks: set,
        features: Dict,
    ):
        # deterministic anchors: most recent + most listened
        track_time = defaultdict(float)
        for t, time in history:
            track_time[t] += time

        recent_tracks = [t for t, _ in history[-self.max_anchors:]]
        top_tracks = sorted(track_time, key=lambda x: -track_time[x])[:self.max_anchors]

        anchors = list(dict.fromkeys(recent_tracks + top_tracks))

        for anchor in anchors:
            data = self.i2i_redis.get(anchor)
            if data is None:
                continue

            recs = pickle.loads(data)

            for rank, raw_track in enumerate(recs[: self.i2i_top]):
                candidate = int(raw_track)
                if candidate in seen_tracks:
                    continue

                # normalized rank score (strong top emphasis)
                score = 1.0 / (1.0 + rank)

                # take max across anchors (important!)
                features[candidate]["i2i"] = max(
                    features[candidate]["i2i"],
                    score
                )


    def _add_hstu_features(self, user: int, seen_tracks: set, features: Dict):
        data = self.hstu_redis.get(user)
        if data is None:
            return

        tracks = list(self.catalog.from_bytes(data))

        for rank, track in enumerate(tracks[: self.hstu_top]):
            if track in seen_tracks:
                continue

            score = 1.0 / (1.0 + rank)

            features[track]["hstu"] = score


    def _add_behavior_features(
        self,
        history: List[Tuple[int, float]],
        features: Dict,
    ):
        track_time = defaultdict(float)
        last_position = {}

        for idx, (track, t) in enumerate(history):
            track_time[track] += t
            last_position[track] = idx

        n = len(history)

        for track in features.keys():
            if track in last_position:
                # recency: closer to end → higher
                features[track]["recency"] = 1.0 - (n - last_position[track]) / n

                # frequency: normalized listening time
                features[track]["freq"] = track_time[track] / sum(track_time.values())
            else:
                features[track]["recency"] = 0.0
                features[track]["freq"] = 0.0


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