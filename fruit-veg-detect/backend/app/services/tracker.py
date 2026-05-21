from __future__ import annotations

import logging
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock
from typing import Any

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

_AUTO_DEEPSORT_FALLBACK_NO_REUSE_FRAMES = 6
_AUTO_DEEPSORT_FALLBACK_MIN_TRACKS = 20

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
except Exception as exc:  # pragma: no cover - dependency optional at runtime
    DeepSort = None
    _deepsort_import_error = exc
else:
    _deepsort_import_error = None


def _should_use_gpu_embedder(device: str) -> bool:
    normalized = (device or "cpu").strip().lower()
    if normalized in {"", "cpu"}:
        return False
    if normalized == "auto":
        try:
            import torch
        except Exception:
            return False
        return bool(torch.cuda.is_available())
    if normalized in {"gpu", "cuda"}:
        normalized = "cuda:0"

    requested_indices: list[int] = [0]
    if normalized.startswith("cuda:"):
        suffix = normalized.split(":", 1)[1]
        if not suffix.isdigit():
            return False
        requested_indices = [int(suffix)]
    elif normalized.isdigit():
        requested_indices = [int(normalized)]
    elif "," in normalized and all(part.isdigit() for part in normalized.split(",")):
        requested_indices = [int(part) for part in normalized.split(",")]
    elif not normalized.startswith("cuda"):
        return False

    try:
        import torch
    except Exception:
        return False

    if not torch.cuda.is_available():
        return False

    device_count = torch.cuda.device_count()
    return all(0 <= index < device_count for index in requested_indices)


def _should_enable_deepsort(backend: str, device: str) -> bool:
    normalized_backend = (backend or "auto").strip().lower()
    if normalized_backend == "naive":
        return False
    if normalized_backend == "deepsort":
        return True
    return _should_use_gpu_embedder(device)


def get_deepsort_status() -> dict[str, Any]:
    return {
        "available": DeepSort is not None,
        "import_error": None if _deepsort_import_error is None else str(_deepsort_import_error),
    }


class _DeepSortAssociationFailure(RuntimeError):
    """Raised when DeepSort stays enabled but never links detections across frames."""


def _box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _box_scale(box: tuple[float, float, float, float]) -> float:
    return max(1.0, max(box[2] - box[0], box[3] - box[1]))


def _center_distance_ratio(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax, ay = _box_center(box_a)
    bx, by = _box_center(box_b)
    distance = math.hypot(ax - bx, ay - by)
    return float(distance / max(_box_scale(box_a), _box_scale(box_b), 1.0))


@dataclass
class _TrackStat:
    track_id: int
    cls_name: str
    cls_id: int
    first_frame: int
    last_frame: int
    frames_seen: int = 0
    path_length: float = 0.0
    last_center: tuple[float, float] | None = None


@dataclass
class _CameraSession:
    tracker: "VideoTracker"
    frame_index: int
    last_seen: datetime
    tracker_backend: str
    max_age: int
    n_init: int
    max_time_since_update: int
    max_iou_distance: float
    max_cosine_distance: float
    nn_budget: int


@dataclass(frozen=True)
class _CameraTrackerConfig:
    tracker_backend: str
    max_age: int
    n_init: int
    max_time_since_update: int
    max_iou_distance: float
    max_cosine_distance: float
    nn_budget: int


def _normalize_backend_mode(raw: str | None) -> str:
    normalized = (raw or "auto").strip().lower()
    if normalized in {"auto", "deepsort", "naive"}:
        return normalized
    return "auto"


class VideoTracker:
    def __init__(
        self,
        backend_override: str | None = None,
        n_init_override: int | None = None,
        max_age_override: int | None = None,
        max_time_since_update_override: int | None = None,
        max_cosine_distance_override: float | None = None,
        max_iou_distance_override: float | None = None,
        nn_budget_override: int | None = None,
        device_override: str | None = None,
        smoothing_enabled_override: bool | None = None,
        smoothing_alpha_override: float | None = None,
        smoothing_max_center_jump_override: float | None = None,
        debug: bool = False,
    ) -> None:
        self.tracker_name = "naive-iou"
        self.deepsort_enabled = False
        self._deep_sort: Any | None = None
        self.backend_mode = _normalize_backend_mode(backend_override or settings.TRACKER_BACKEND)
        self.device = device_override or settings.DEVICE
        self.max_age = max(
            1,
            int(max_age_override if max_age_override is not None else settings.TRACKER_MAX_AGE),
        )
        self.n_init = max(
            1,
            int(n_init_override if n_init_override is not None else settings.TRACKER_N_INIT),
        )
        self.max_cosine_distance = float(
            max_cosine_distance_override
            if max_cosine_distance_override is not None
            else settings.TRACKER_MAX_COSINE_DISTANCE
        )
        self.max_iou_distance = float(
            max_iou_distance_override
            if max_iou_distance_override is not None
            else settings.TRACKER_MAX_IOU_DISTANCE
        )
        nn_budget = int(nn_budget_override if nn_budget_override is not None else settings.TRACKER_NN_BUDGET)
        self.nn_budget: int | None = nn_budget if nn_budget > 0 else None
        self.max_time_since_update = max(
            0,
            int(
                max_time_since_update_override
                if max_time_since_update_override is not None
                else settings.TRACKER_MAX_TIME_SINCE_UPDATE
            ),
        )
        self.smoothing_enabled = bool(
            smoothing_enabled_override
            if smoothing_enabled_override is not None
            else settings.SMOOTHING_ENABLED
        )
        self.smoothing_alpha = min(
            1.0,
            max(
                0.0,
                float(
                    smoothing_alpha_override
                    if smoothing_alpha_override is not None
                    else settings.SMOOTHING_ALPHA
                ),
            ),
        )
        self.smoothing_max_center_jump = float(
            smoothing_max_center_jump_override
            if smoothing_max_center_jump_override is not None
            else settings.SMOOTHING_MAX_CENTER_JUMP
        )
        self.smoothing_apply_to_display_only = bool(settings.SMOOTHING_APPLY_TO_DISPLAY_ONLY)
        self.debug = bool(debug)

        enable_deepsort = _should_enable_deepsort(self.backend_mode, self.device)

        if DeepSort is None:
            logger.warning(
                "deep_sort_realtime is not available (%s). "
                "Video tracking will fallback to naive IoU tracker.",
                _deepsort_import_error,
            )
        elif not enable_deepsort:
            logger.info(
                "Tracker backend is set to %s under device %s. "
                "Using naive IoU tracker for lower latency.",
                self.backend_mode,
                self.device,
            )
        else:
            try:
                self._deep_sort = DeepSort(
                    max_age=max(self.max_age, self.max_time_since_update + 2),
                    n_init=self.n_init,
                    max_iou_distance=self.max_iou_distance,
                    max_cosine_distance=self.max_cosine_distance,
                    nn_budget=self.nn_budget,
                    nms_max_overlap=1.0,
                    embedder="mobilenet",
                    half=True,
                    bgr=True,
                    embedder_gpu=_should_use_gpu_embedder(self.device),
                )
                self.tracker_name = "deepsort"
                self.deepsort_enabled = True
            except Exception as exc:
                logger.warning("Failed to initialize DeepSort (%s). Falling back to naive IoU tracker.", exc)

        self._reset_tracking_state()

    def _reset_tracking_state(self) -> None:
        self._next_track_id = 1
        self._naive_tracks: dict[int, dict[str, Any]] = {}
        self._track_meta: dict[int, dict[str, Any]] = {}
        self._track_stats: dict[int, _TrackStat] = {}
        self._trajectories: dict[int, list[tuple[int, int] | None]] = defaultdict(list)
        self._smoothed_bboxes: dict[int, tuple[float, float, float, float]] = {}
        self._deepsort_no_reuse_frames = 0
        self._latest_frame_index = -1

    def _disable_deepsort(self, reason: Exception | str, *, reset_state: bool = False) -> None:
        logger.warning("DeepSort runtime failure, falling back to naive IoU tracker: %s", reason)
        self._deep_sort = None
        self.deepsort_enabled = False
        self.tracker_name = "naive-iou"
        self._deepsort_no_reuse_frames = 0
        if reset_state:
            self._reset_tracking_state()
            return

        current_max_track_id = max(self._track_stats.keys(), default=0)
        self._next_track_id = max(self._next_track_id, current_max_track_id + 1)

    def _match_detection(
        self,
        track_box: tuple[float, float, float, float],
        detections: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        best_det: dict[str, Any] | None = None
        best_iou = 0.0
        for det in detections:
            det_box = (float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"]))
            iou = _box_iou(track_box, det_box)
            if iou > best_iou:
                best_iou = iou
                best_det = det
        if best_iou < 0.01:
            return None
        return best_det

    def _smoothed_display_box(
        self,
        track_id: int,
        raw_box: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        if not self.smoothing_enabled:
            self._smoothed_bboxes[track_id] = raw_box
            return raw_box

        previous = self._smoothed_bboxes.get(track_id)
        if previous is None:
            smoothed = raw_box
        else:
            alpha = self.smoothing_alpha
            smoothed = tuple(
                (alpha * prev_value) + ((1.0 - alpha) * current_value)
                for prev_value, current_value in zip(previous, raw_box)
            )
        self._smoothed_bboxes[track_id] = smoothed
        return smoothed

    def _build_display_box(
        self,
        raw_box_data: dict[str, Any],
        raw_box: tuple[float, float, float, float],
    ) -> tuple[dict[str, Any], tuple[float, float, float, float]]:
        track_id = int(raw_box_data["track_id"])
        smoothed_box = self._smoothed_display_box(track_id, raw_box)
        display_box = dict(raw_box_data)
        display_box.update(
            {
                "x1": smoothed_box[0],
                "y1": smoothed_box[1],
                "x2": smoothed_box[2],
                "y2": smoothed_box[3],
                "raw_bbox": [float(value) for value in raw_box],
                "smoothed_bbox": [float(value) for value in smoothed_box],
                "smoothing_alpha": self.smoothing_alpha,
            }
        )
        return display_box, smoothed_box

    def _trim_trajectory(self, track_id: int) -> None:
        points = self._trajectories[track_id]
        trail_length = max(1, int(settings.TRACKER_TRAJECTORY_LEN))
        if len(points) > trail_length:
            del points[:-trail_length]

    def _prune_expired_tracks(self, frame_index: int) -> None:
        stale_ids = [
            track_id
            for track_id, stat in self._track_stats.items()
            if frame_index - stat.last_frame > self.max_age
        ]
        for track_id in stale_ids:
            self._naive_tracks.pop(track_id, None)
            self._track_meta.pop(track_id, None)
            self._track_stats.pop(track_id, None)
            self._trajectories.pop(track_id, None)
            self._smoothed_bboxes.pop(track_id, None)

    def _update_track_state(
        self,
        track_id: int,
        cls_name: str,
        cls_id: int,
        frame_index: int,
        box: tuple[float, float, float, float],
        *,
        time_since_update: int = 0,
    ) -> None:
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0

        stat = self._track_stats.get(track_id)
        if stat is None:
            stat = _TrackStat(
                track_id=track_id,
                cls_name=cls_name,
                cls_id=cls_id,
                first_frame=frame_index,
                last_frame=frame_index,
            )
            self._track_stats[track_id] = stat

        stat.cls_name = cls_name
        stat.cls_id = cls_id
        stat.last_frame = frame_index
        stat.frames_seen += 1

        large_jump = False
        if stat.last_center is not None:
            jump = float(math.hypot(cx - stat.last_center[0], cy - stat.last_center[1]))
            large_jump = (
                self.smoothing_enabled
                and self.smoothing_max_center_jump > 0
                and jump > self.smoothing_max_center_jump
            )
            if large_jump:
                if self.debug:
                    print(f"[Smoothing] large jump ignored: track_id={track_id}, jump={jump:.2f}", flush=True)
            else:
                stat.path_length += jump
        stat.last_center = (cx, cy)

        points = self._trajectories[track_id]
        if large_jump:
            if not points or points[-1] is not None:
                points.append(None)
            self._trim_trajectory(track_id)
            return

        point = (int(round(cx)), int(round(cy)))
        if not points or points[-1] != point:
            points.append(point)
        self._trim_trajectory(track_id)

    def _should_fallback_for_poor_deepsort_association(self) -> bool:
        if self._deepsort_no_reuse_frames < _AUTO_DEEPSORT_FALLBACK_NO_REUSE_FRAMES:
            return False
        if len(self._track_stats) < _AUTO_DEEPSORT_FALLBACK_MIN_TRACKS:
            return False

        multi_frame_tracks = sum(1 for stat in self._track_stats.values() if stat.frames_seen >= 2)
        return multi_frame_tracks == 0

    def _update_with_deepsort(
        self,
        detections: list[dict[str, Any]],
        frame_bgr: np.ndarray,
        frame_index: int,
    ) -> list[dict[str, Any]]:
        assert self._deep_sort is not None

        prepared: list[tuple[list[float], float, str]] = []
        for det in detections:
            x1 = float(det["x1"])
            y1 = float(det["y1"])
            x2 = float(det["x2"])
            y2 = float(det["y2"])
            prepared.append(([x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)], float(det["conf"]), str(det["cls_name"])))

        previous_track_ids = set(self._track_stats.keys())
        tracks = self._deep_sort.update_tracks(prepared, frame=frame_bgr)
        tracked_boxes: list[dict[str, Any]] = []
        reused_track_hits = 0

        for track in tracks:
            is_confirmed = bool(track.is_confirmed())
            if not is_confirmed:
                # When n_init is forced to 1 (e.g. sparse video sampling),
                # expose fresh tracks immediately to avoid "detections but no track ids".
                if self.n_init > 1:
                    continue
                if track.time_since_update > 0:
                    continue
            if track.time_since_update > self.max_time_since_update:
                continue

            ltrb = track.to_ltrb()
            track_id = int(track.track_id)
            time_since_update = int(getattr(track, "time_since_update", 0))
            track_box = (float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3]))
            if track_id in previous_track_ids:
                reused_track_hits += 1
            matched = self._match_detection(track_box, detections)

            if matched is not None:
                self._track_meta[track_id] = {
                    "cls_name": str(matched.get("cls_name", "obj")),
                    "cls_id": int(matched.get("cls_id", -1)),
                    "conf": float(matched.get("conf", 0.0)),
                }

            meta = self._track_meta.get(track_id, {})
            cls_name = str(meta.get("cls_name", getattr(track, "det_class", "obj") or "obj"))
            cls_id = int(meta.get("cls_id", -1))
            conf = float(meta.get("conf", 0.0))

            tracked_box = {
                "x1": track_box[0],
                "y1": track_box[1],
                "x2": track_box[2],
                "y2": track_box[3],
                "conf": conf,
                "cls_id": cls_id,
                "cls_name": cls_name,
                "track_id": track_id,
                "time_since_update": time_since_update,
            }
            display_box, smoothed_track_box = self._build_display_box(tracked_box, track_box)
            tracked_boxes.append(display_box)
            self._update_track_state(
                track_id,
                cls_name,
                cls_id,
                frame_index,
                smoothed_track_box,
                time_since_update=time_since_update,
            )

        if detections and tracked_boxes:
            if reused_track_hits == 0:
                self._deepsort_no_reuse_frames += 1
            else:
                self._deepsort_no_reuse_frames = 0

            if self._should_fallback_for_poor_deepsort_association():
                mode_label = "auto mode" if self.backend_mode == "auto" else f"{self.backend_mode} mode"
                self._disable_deepsort(
                    f"DeepSort could not associate detections across frames in {mode_label}.",
                    reset_state=True,
                )
                raise _DeepSortAssociationFailure("DeepSort produced only single-frame tracks.")
        elif reused_track_hits > 0:
            self._deepsort_no_reuse_frames = 0

        return tracked_boxes

    def _update_with_naive(
        self,
        detections: list[dict[str, Any]],
        frame_index: int,
    ) -> list[dict[str, Any]]:
        for track in self._naive_tracks.values():
            track["age"] = int(track["age"]) + 1

        assigned_tracks: set[int] = set()
        tracked_boxes: list[dict[str, Any]] = []

        for det in detections:
            det_box = (float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"]))
            best_track_id: int | None = None
            best_iou = 0.0
            best_dist_ratio = float("inf")
            best_score = float("-inf")
            det_cls_id = int(det.get("cls_id", -1))

            for track_id, state in self._naive_tracks.items():
                if track_id in assigned_tracks:
                    continue
                state_cls_id = int(state.get("cls_id", -1))
                if state_cls_id >= 0 and det_cls_id >= 0 and state_cls_id != det_cls_id:
                    continue

                iou = _box_iou(det_box, state["box"])
                dist_ratio = _center_distance_ratio(det_box, state["box"])
                if dist_ratio > settings.TRACKER_FALLBACK_MAX_CENTER_DISTANCE_RATIO:
                    continue

                score = iou - (dist_ratio * settings.TRACKER_FALLBACK_CENTER_DISTANCE_WEIGHT)
                if score > best_score:
                    best_score = score
                    best_iou = iou
                    best_dist_ratio = dist_ratio
                    best_track_id = track_id

            matches_existing_track = (
                best_track_id is not None
                and (
                    best_iou >= settings.TRACKER_FALLBACK_IOU
                    or best_dist_ratio <= settings.TRACKER_FALLBACK_RECONNECT_DISTANCE_RATIO
                )
            )

            if not matches_existing_track:
                best_track_id = self._next_track_id
                self._next_track_id += 1

            self._naive_tracks[best_track_id] = {
                "box": det_box,
                "age": 0,
                "cls_id": det_cls_id,
                "cls_name": str(det.get("cls_name", "obj")),
                "conf": float(det.get("conf", 0.0)),
            }
            assigned_tracks.add(best_track_id)

            tracked_box = {
                **det,
                "track_id": best_track_id,
                "time_since_update": 0,
            }
            display_box, smoothed_track_box = self._build_display_box(tracked_box, det_box)
            tracked_boxes.append(display_box)
            self._update_track_state(
                best_track_id,
                str(det.get("cls_name", "obj")),
                int(det.get("cls_id", -1)),
                frame_index,
                smoothed_track_box,
                time_since_update=0,
            )

        stale_ids = [
            track_id
            for track_id, state in self._naive_tracks.items()
            if int(state.get("age", 0)) > self.max_time_since_update
        ]
        for track_id in stale_ids:
            self._naive_tracks.pop(track_id, None)

        return tracked_boxes

    def update(
        self,
        detections: list[dict[str, Any]],
        frame_bgr: np.ndarray,
        frame_index: int,
    ) -> list[dict[str, Any]]:
        self._latest_frame_index = max(self._latest_frame_index, frame_index)
        tracked_boxes: list[dict[str, Any]]
        if self.deepsort_enabled and self._deep_sort is not None:
            try:
                tracked_boxes = self._update_with_deepsort(detections, frame_bgr, frame_index)
                self._prune_expired_tracks(frame_index)
                return tracked_boxes
            except _DeepSortAssociationFailure:
                tracked_boxes = self._update_with_naive(detections, frame_index)
                self._prune_expired_tracks(frame_index)
                return tracked_boxes
            except Exception as exc:
                logger.exception("DeepSort tracking failed")
                self._disable_deepsort(exc)
        tracked_boxes = self._update_with_naive(detections, frame_index)
        self._prune_expired_tracks(frame_index)
        return tracked_boxes

    def get_trajectories(self, active_within_frames: int | None = None) -> dict[int, list[tuple[int, int] | None]]:
        visible: dict[int, list[tuple[int, int] | None]] = {}
        for track_id, points in self._trajectories.items():
            if active_within_frames is not None:
                stat = self._track_stats.get(track_id)
                if stat is None:
                    continue
                if self._latest_frame_index - stat.last_frame > active_within_frames:
                    continue
            visible[track_id] = points[:]
        return visible

    def get_track_summaries(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for track_id in sorted(self._track_stats.keys()):
            stat = self._track_stats[track_id]
            items.append(
                {
                    "track_id": stat.track_id,
                    "cls_name": stat.cls_name,
                    "cls_id": stat.cls_id,
                    "first_frame": stat.first_frame,
                    "last_frame": stat.last_frame,
                    "frames_seen": stat.frames_seen,
                    "path_length": round(stat.path_length, 2),
                }
            )
        return items

    def get_serialized_trajectories(self, active_within_frames: int | None = None) -> dict[str, list[dict[str, int | bool]]]:
        serialized: dict[str, list[dict[str, int | bool]]] = {}
        for track_id, points in self.get_trajectories(active_within_frames=active_within_frames).items():
            items: list[dict[str, int | bool]] = []
            for point in points:
                if point is None:
                    items.append({"break": True})
                    continue
                x, y = point
                items.append({"x": int(x), "y": int(y)})
            serialized[str(track_id)] = items
        return serialized


class CameraTrackerManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[str, _CameraSession] = {}

    def _cleanup_locked(self, now: datetime) -> None:
        ttl = timedelta(seconds=max(1, settings.CAMERA_TRACKER_SESSION_TTL_SEC))
        stale = [sid for sid, sess in self._sessions.items() if now - sess.last_seen > ttl]
        for sid in stale:
            self._sessions.pop(sid, None)

    def _build_tracker(
        self,
        config: _CameraTrackerConfig,
    ) -> VideoTracker:
        return VideoTracker(
            backend_override=config.tracker_backend,
            n_init_override=config.n_init,
            max_age_override=config.max_age,
            max_time_since_update_override=config.max_time_since_update,
            max_cosine_distance_override=config.max_cosine_distance,
            max_iou_distance_override=config.max_iou_distance,
            nn_budget_override=config.nn_budget,
            device_override=settings.CAMERA_DEVICE,
        )

    def _resolve_tracker_config(
        self,
        tracker_backend: str | None,
        max_time_since_update_override: int | None,
        max_age_override: int | None = None,
        n_init_override: int | None = None,
        max_cosine_distance_override: float | None = None,
        max_iou_distance_override: float | None = None,
        nn_budget_override: int | None = None,
    ) -> _CameraTrackerConfig:
        max_age = max(
            1,
            int(max_age_override if max_age_override is not None else settings.CAMERA_TRACKER_MAX_AGE),
        )
        return _CameraTrackerConfig(
            tracker_backend=_normalize_backend_mode(tracker_backend or settings.TRACKER_BACKEND),
            max_age=max_age,
            n_init=max(
                1,
                int(n_init_override if n_init_override is not None else settings.CAMERA_TRACKER_N_INIT),
            ),
            max_time_since_update=max(
                1,
                int(
                    max_time_since_update_override
                    if max_time_since_update_override is not None
                    else max_age
                ),
            ),
            max_iou_distance=float(
                max_iou_distance_override
                if max_iou_distance_override is not None
                else settings.TRACKER_MAX_IOU_DISTANCE
            ),
            max_cosine_distance=float(
                max_cosine_distance_override
                if max_cosine_distance_override is not None
                else settings.CAMERA_TRACKER_MAX_COSINE_DISTANCE
            ),
            nn_budget=max(
                0,
                int(nn_budget_override if nn_budget_override is not None else settings.CAMERA_TRACKER_NN_BUDGET),
            ),
        )

    def create_or_get_session(self, session_id: str | None = None) -> str:
        now = datetime.now()
        with self._lock:
            self._cleanup_locked(now)
            sid = session_id or uuid.uuid4().hex
            if sid not in self._sessions:
                config = self._resolve_tracker_config(None, None)
                self._sessions[sid] = _CameraSession(
                    tracker=self._build_tracker(config),
                    frame_index=0,
                    last_seen=now,
                    tracker_backend=config.tracker_backend,
                    max_age=config.max_age,
                    n_init=config.n_init,
                    max_time_since_update=config.max_time_since_update,
                    max_iou_distance=config.max_iou_distance,
                    max_cosine_distance=config.max_cosine_distance,
                    nn_budget=config.nn_budget,
                )
            return sid

    def reset_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def update_session(
        self,
        session_id: str | None,
        detections: list[dict[str, Any]],
        frame_bgr: np.ndarray,
        tracker_backend: str | None = None,
        max_time_since_update_override: int | None = None,
        max_age_override: int | None = None,
        n_init_override: int | None = None,
        max_cosine_distance_override: float | None = None,
        max_iou_distance_override: float | None = None,
        nn_budget_override: int | None = None,
        debug: bool = False,
    ) -> dict[str, Any]:
        now = datetime.now()
        with self._lock:
            self._cleanup_locked(now)
            sid = session_id or uuid.uuid4().hex
            session = self._sessions.get(sid)
            requested_config = self._resolve_tracker_config(
                tracker_backend,
                max_time_since_update_override,
                max_age_override=max_age_override,
                n_init_override=n_init_override,
                max_cosine_distance_override=max_cosine_distance_override,
                max_iou_distance_override=max_iou_distance_override,
                nn_budget_override=nn_budget_override,
            )
            if session is None:
                session = _CameraSession(
                    tracker=self._build_tracker(requested_config),
                    frame_index=0,
                    last_seen=now,
                    tracker_backend=requested_config.tracker_backend,
                    max_age=requested_config.max_age,
                    n_init=requested_config.n_init,
                    max_time_since_update=requested_config.max_time_since_update,
                    max_iou_distance=requested_config.max_iou_distance,
                    max_cosine_distance=requested_config.max_cosine_distance,
                    nn_budget=requested_config.nn_budget,
                )
                self._sessions[sid] = session
            elif (
                session.tracker_backend != requested_config.tracker_backend
                or session.max_age != requested_config.max_age
                or session.n_init != requested_config.n_init
                or session.max_time_since_update != requested_config.max_time_since_update
                or session.max_iou_distance != requested_config.max_iou_distance
                or session.max_cosine_distance != requested_config.max_cosine_distance
                or session.nn_budget != requested_config.nn_budget
            ):
                session = _CameraSession(
                    tracker=self._build_tracker(requested_config),
                    frame_index=0,
                    last_seen=now,
                    tracker_backend=requested_config.tracker_backend,
                    max_age=requested_config.max_age,
                    n_init=requested_config.n_init,
                    max_time_since_update=requested_config.max_time_since_update,
                    max_iou_distance=requested_config.max_iou_distance,
                    max_cosine_distance=requested_config.max_cosine_distance,
                    nn_budget=requested_config.nn_budget,
                )
                self._sessions[sid] = session

            session.tracker.debug = bool(debug)
            frame_index = session.frame_index
            tracked_boxes = session.tracker.update(detections, frame_bgr, frame_index)
            session.frame_index += 1
            session.last_seen = now

            return {
                "session_id": sid,
                "frame_index": frame_index,
                "boxes": tracked_boxes,
                "trajectories": session.tracker.get_serialized_trajectories(
                    active_within_frames=session.max_time_since_update,
                ),
                "track_summaries": session.tracker.get_track_summaries(),
                "tracker": session.tracker.tracker_name,
                "deepsort_enabled": session.tracker.deepsort_enabled,
                "tracker_config": {
                    "max_age": session.max_age,
                    "n_init": session.n_init,
                    "max_time_since_update": session.max_time_since_update,
                    "max_iou_distance": session.max_iou_distance,
                    "max_cosine_distance": session.max_cosine_distance,
                    "nn_budget": session.nn_budget,
                    "trail_length": settings.TRACKER_TRAJECTORY_LEN,
                    "smoothing_enabled": session.tracker.smoothing_enabled,
                    "smoothing_alpha": session.tracker.smoothing_alpha,
                    "max_center_jump": session.tracker.smoothing_max_center_jump,
                },
            }

    def session_count(self) -> int:
        now = datetime.now()
        with self._lock:
            self._cleanup_locked(now)
            return len(self._sessions)


camera_tracker_manager = CameraTrackerManager()
