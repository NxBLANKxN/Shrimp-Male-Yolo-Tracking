# modules/id_assigner.py

from __future__ import annotations

import math


class FixedIDAssigner:
    """Assign detections into a fixed ID pool, e.g. ID1..ID6."""

    def __init__(self, total_ids: int, match_distance: float) -> None:
        self.total_ids = max(1, total_ids)
        self.match_distance = match_distance
        self._next_id = 1
        self._tracks: dict[int, tuple[float, float]] = {}

    def assign(self, detections: list[dict]) -> list[dict]:
        if not detections:
            return detections

        for det in detections:
            det["include_in_stats"] = True

        pairs = []
        for det_idx, det in enumerate(detections):
            for shrimp_id, (cx, cy) in self._tracks.items():
                dist = math.hypot(det["cx"] - cx, det["cy"] - cy)
                pairs.append((dist, det_idx, shrimp_id))

        used_dets: set[int] = set()
        used_ids: set[int] = set()
        for dist, det_idx, shrimp_id in sorted(pairs):
            if det_idx in used_dets or shrimp_id in used_ids:
                continue
            status = "matched" if dist <= self.match_distance else "forced"
            self._tag(detections[det_idx], shrimp_id, status, dist)
            used_dets.add(det_idx)
            used_ids.add(shrimp_id)

        for det_idx, det in enumerate(detections):
            if det_idx in used_dets:
                continue
            if len(used_ids) >= self.total_ids:
                self._tag(det, "", "overflow", 0.0, include=False)
                continue
            shrimp_id, status, dist = self._new_or_nearest_id(det)
            while shrimp_id in used_ids and len(used_ids) < self.total_ids:
                shrimp_id = self._nearest_unused_id(det, used_ids)
                status = "forced"
                dist = self._distance_to_id(det, shrimp_id)
            self._tag(det, shrimp_id, status, dist)
            used_ids.add(shrimp_id)

        for det in detections:
            if det.get("include_in_stats", True):
                self._tracks[det["shrimp_id"]] = (det["cx"], det["cy"])

        return detections

    def _new_or_nearest_id(self, det: dict) -> tuple[int, str, float]:
        if self._next_id <= self.total_ids:
            shrimp_id = self._next_id
            self._next_id += 1
            return shrimp_id, "created", 0.0

        if not self._tracks:
            return 1, "forced", 0.0

        shrimp_id, dist = min(
            (
                (sid, math.hypot(det["cx"] - cx, det["cy"] - cy))
                for sid, (cx, cy) in self._tracks.items()
            ),
            key=lambda item: item[1],
        )
        return shrimp_id, "forced", dist

    def _nearest_unused_id(self, det: dict, used_ids: set[int]) -> int:
        candidates = [sid for sid in range(1, self.total_ids + 1) if sid not in used_ids]
        if not candidates:
            return 1
        return min(candidates, key=lambda sid: self._distance_to_id(det, sid))

    def _distance_to_id(self, det: dict, shrimp_id: int) -> float:
        if shrimp_id not in self._tracks:
            return 0.0
        cx, cy = self._tracks[shrimp_id]
        return math.hypot(det["cx"] - cx, det["cy"] - cy)

    @staticmethod
    def _tag(det: dict, shrimp_id, status: str, distance: float, include: bool = True) -> None:
        det["shrimp_id"] = shrimp_id
        det["id_status"] = status
        det["id_distance"] = round(float(distance), 2)
        det["include_in_stats"] = include
