#!/usr/bin/env python3
"""
SecurePi — Raspberry Pi AI Camera (IMX500) Security Monitor.

Runs object detection *on the IMX500 sensor* to find people and bags
(backpack, handbag, suitcase), tracks each bag across frames, and raises an
alert when a bag is left unattended (no person nearby) for longer than a
configured time.

Detections are read from the neural-network output tensors carried in the
Picamera2 frame metadata via the IMX500 helper (`get_outputs` +
`convert_inference_coords`) — matching the official Raspberry Pi IMX500 object
detection demo. Runs with a live preview window by default, or fully headless
(`--headless`) for SSH / systemd use. Alert snapshots are saved to disk.

Settings come from preset files in the presets/ folder (required):
presets/common.args is applied automatically as the shared base, then the
named preset's overrides, then any command-line flags.

Examples
--------
    python securePi.py @lobby.args
    python securePi.py @kitchen.args --headless --unattended-time 60
    python securePi.py @lobby.args --model /usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import concurrent.futures

import cv2

try:
    from picamera2 import Picamera2
    from picamera2.devices import IMX500
    from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)
    _IMX500_AVAILABLE = True
except ImportError:  # let --help / imports work off-device
    Picamera2 = IMX500 = NetworkIntrinsics = postprocess_nanodet_detection = None  # type: ignore[assignment]
    _IMX500_AVAILABLE = False


LOGGER = logging.getLogger("securepi")

# Default network shipped by `sudo apt install imx500-all` (COCO SSD MobileNetV2).
DEFAULT_MODEL = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"

# COCO categories we treat as "bags". Labels come back as lower-case strings.
DEFAULT_BAG_LABELS = {"backpack", "handbag", "suitcase"}
DEFAULT_PERSON_LABELS = {"person"}

# Global executor for non-blocking snapshot saving
SNAPSHOT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# BGR colours (OpenCV order).
COLOR_PERSON = (0, 255, 0)
COLOR_ATTENDED = (255, 255, 0)
COLOR_WARNING = (0, 165, 255)
COLOR_ALERT = (0, 0, 255)
COLOR_HUD = (0, 255, 0)


@dataclass
class Config:
    """Tunable parameters for the monitor."""

    unattended_time_sec: float = 120.0   # alert after this long unattended
    stationary_radius: float = 120.0     # association radius: px a bag may move between
                                         # detections and still match the same track
    person_proximity_px: float = 150.0   # px within which the owner attends a bag
    owner_claim_sec: float = 3.0         # after a bag appears, a person near it within this
                                         # window is adopted as its OWNER; only the owner
                                         # attending resets the timer (others are ignored)
    person_timeout_sec: float = 2.0      # drop a person track unseen for this long
    person_match_radius: float = 150.0   # px to associate a person to the same track
    track_timeout_sec: float = 10.0      # "coast" window: keep a track alive (and its
                                         # unattended timer running) through detection
                                         # dropouts; only drop it after this long unseen
    draw_grace_sec: float = 1.5          # stop drawing a coasting track's box once it has
                                         # been unseen this long (avoids lingering "ghost"
                                         # boxes); the track itself lives until the timeout
    min_confidence: float = 0.5          # ignore detections below this score
    box_smoothing: float = 0.6           # weight of the newest detection when smoothing a
                                         # track's drawn box (1.0 = no smoothing); damps
                                         # frame-to-frame detector jitter on static objects
    frame_size: tuple[int, int] = (640, 480)
    headless: bool = False               # run without a preview window
    alert_cooldown_sec: float = 30.0     # seconds between repeat alerts per bag
    snapshot_dir: Path = Path("alerts")  # where alert snapshots are written
    max_snapshots: int = 500             # keep at most this many snapshots; oldest are
                                         # deleted so long runs can't fill the SD card
    # IMX500 detector settings
    model_path: str = DEFAULT_MODEL
    labels_path: Optional[str] = None    # override the model's built-in labels
    iou: float = 0.65                    # NMS IoU threshold (nanodet models)
    max_detections: int = 10             # max detections (nanodet models)
    bag_labels: set[str] = field(default_factory=lambda: set(DEFAULT_BAG_LABELS))
    person_labels: set[str] = field(default_factory=lambda: set(DEFAULT_PERSON_LABELS))


@dataclass
class Detection:
    """A single object detection for the current frame (box in image pixels)."""

    label: str
    score: float
    box: tuple[int, int, int, int]  # x, y, w, h in pixels

    @property
    def centroid(self) -> tuple[float, float]:
        return box_centroid(self.box)


@dataclass
class TrackedBag:
    """State for a bag followed across frames."""

    bag_id: int
    centroid: tuple[float, float]
    box: tuple[int, int, int, int]
    last_seen: float
    created_at: float                       # when the track was first registered
    owner_id: Optional[int] = None          # person id adopted as the bag's owner
    unattended_start: Optional[float] = None
    alerted: bool = False
    last_alert_time: float = 0.0


@dataclass
class PersonTrack:
    """A person followed across frames, giving them a stable id for owner matching."""

    person_id: int
    centroid: tuple[float, float]
    box: tuple[int, int, int, int]
    last_seen: float


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def box_centroid(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def calculate_iou(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    return interArea / float(boxAArea + boxBArea - interArea)


def smooth_box(old: tuple[int, int, int, int], new: tuple[int, int, int, int],
               alpha: float) -> tuple[int, int, int, int]:
    """Blend consecutive boxes to damp detector jitter on the drawn box.

    ``alpha`` is the weight of the NEW detection (1.0 disables smoothing). If
    the new box barely overlaps the old one the object genuinely moved, so snap
    to the detection instead of dragging a laggy box across the scene.
    """
    if alpha >= 1.0 or calculate_iou(old, new) < 0.2:
        return new
    x, y, w, h = (round(o + alpha * (n - o)) for o, n in zip(old, new))
    return (x, y, w, h)


# Detections/tracks overlapping this much are the same physical object.
DEDUP_IOU = 0.45


def dedup_detections(detections: list[Detection]) -> list[Detection]:
    """Cross-label NMS: collapse detections of the same physical object.

    The sensor's SSD post-processing suppresses duplicates per class only, so
    one bag can come back as both "backpack" and "handbag" at nearly the same
    box in the same frame — which would spawn one track (and one alert box)
    per label. All bag labels mean the same thing here, so keep only the
    highest-confidence detection of any overlapping cluster.
    """
    keep: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.score, reverse=True):
        if all(calculate_iou(det.box, k.box) < DEDUP_IOU for k in keep):
            keep.append(det)
    return keep


def match_detections(detections: list[Detection], tracks: list,
                     iou_gate: float, dist_gate: float) -> dict[int, Any]:
    """Best-first global matching of this frame's detections to tracks.

    Considers every detection/track pair at once: IoU overlaps (>= iou_gate)
    always outrank centroid-distance fallbacks (<= dist_gate), and within each
    tier the strongest pair is assigned first. Unlike per-detection greedy
    matching this is independent of detection order, so one detection can't
    steal the track that another detection overlaps better.

    Returns {detection_index: track}.
    """
    pairs: list[tuple[int, float, int, int]] = []
    for di, det in enumerate(detections):
        for ti, track in enumerate(tracks):
            iou = calculate_iou(det.box, track.box)
            if iou >= iou_gate:
                pairs.append((1, iou, di, ti))
            else:
                d = distance(det.centroid, track.centroid)
                if d <= dist_gate:
                    pairs.append((0, -d, di, ti))
    pairs.sort(key=lambda p: (p[0], p[1]), reverse=True)

    matches: dict[int, Any] = {}
    used_tracks: set[int] = set()
    for _, _, di, ti in pairs:
        if di in matches or ti in used_tracks:
            continue
        matches[di] = tracks[ti]
        used_tracks.add(ti)
    return matches


class IMX500Detector:
    """Loads a network onto the IMX500 and turns frame metadata into Detections.

    Mirrors the parsing contract of the official Raspberry Pi
    `imx500_object_detection_demo.py` (output tensors -> boxes/scores/classes ->
    `convert_inference_coords`), so it works with the standard COCO models.
    """

    def __init__(self, config: Config) -> None:
        if not _IMX500_AVAILABLE:
            raise RuntimeError(
                "IMX500 support is unavailable. Install the Pi camera stack: "
                "`sudo apt install -y python3-picamera2 imx500-all`."
            )
        self.threshold = config.min_confidence
        self.iou = config.iou
        self.max_detections = config.max_detections

        self.imx500 = IMX500(config.model_path)
        intrinsics = self.imx500.network_intrinsics
        if not intrinsics:
            intrinsics = NetworkIntrinsics()
            intrinsics.task = "object detection"
        elif intrinsics.task != "object detection":
            raise RuntimeError(f"Model '{config.model_path}' is not an object-detection network.")

        if config.labels_path:
            with open(config.labels_path, encoding="utf-8") as fh:
                intrinsics.labels = fh.read().splitlines()
        intrinsics.update_with_defaults()

        self.intrinsics = intrinsics
        self._labels = self._build_labels()

    @property
    def camera_num(self) -> int:
        return self.imx500.camera_num

    @property
    def inference_rate(self) -> Optional[float]:
        return self.intrinsics.inference_rate

    def show_progress(self) -> None:
        """Display the firmware-upload progress bar (first run can take ~30s)."""
        self.imx500.show_network_fw_progress_bar()

    def _build_labels(self) -> list[str]:
        labels = self.intrinsics.labels or []
        if self.intrinsics.ignore_dash_labels:
            labels = [lbl for lbl in labels if lbl and lbl != "-"]
        return labels

    def _label_for(self, idx: int) -> str:
        if 0 <= idx < len(self._labels):
            return self._labels[idx].lower()
        return str(idx)

    def detect(self, metadata: Any, picam2: Any) -> list[Detection]:
        """Parse the IMX500 output tensors for this frame into Detections."""
        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return []  # firmware still uploading or no inference yet this frame

        intr = self.intrinsics
        input_w, input_h = self.imx500.get_input_size()

        if intr.postprocess == "nanodet":
            from picamera2.devices.imx500.postprocess import scale_boxes
            boxes, scores, classes = postprocess_nanodet_detection(
                outputs=np_outputs[0], conf=self.threshold,
                iou_thres=self.iou, max_out_dets=self.max_detections,
            )[0]
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
        else:
            boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            if intr.bbox_normalization:
                boxes = boxes / input_h
            if intr.bbox_order == "xy":
                boxes = boxes[:, [1, 0, 3, 2]]

        detections: list[Detection] = []
        for box, score, category in zip(boxes, scores, classes):
            if score <= self.threshold:
                continue
            # convert_inference_coords maps normalized box -> (x, y, w, h) pixels
            # in the main-stream image space, accounting for the ScalerCrop.
            x, y, w, h = self.imx500.convert_inference_coords(box, metadata, picam2)
            label = self._label_for(int(category))
            # round() rather than int(): truncation shifts every box up-left.
            detections.append(Detection(label, float(score),
                                        (round(x), round(y), round(w), round(h))))
        return detections


class PersonTracker:
    """Lightweight IoU/centroid tracker that assigns persons stable ids.

    The ids let BagTracker tell a bag's owner apart from anyone else who happens
    to walk near it.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.tracks: dict[int, PersonTrack] = {}
        self._next_id = 0

    def update(self, detections: list[Detection], now: float) -> None:
        detections = dedup_detections(detections)
        matches = match_detections(detections, list(self.tracks.values()),
                                   iou_gate=0.2,
                                   dist_gate=self.config.person_match_radius)
        for di, det in enumerate(detections):
            track = matches.get(di)
            if track is None:
                track = PersonTrack(self._next_id, det.centroid, det.box, now)
                self.tracks[track.person_id] = track
                self._next_id += 1
            else:
                track.box = smooth_box(track.box, det.box, self.config.box_smoothing)
                track.centroid = box_centroid(track.box)
                track.last_seen = now

    def prune(self, now: float) -> None:
        expired = [pid for pid, t in self.tracks.items()
                   if now - t.last_seen > self.config.person_timeout_sec]
        for pid in expired:
            del self.tracks[pid]


class BagTracker:
    """Greedy nearest-centroid tracker with an unattended timer per bag."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.bags: dict[int, TrackedBag] = {}
        self._next_id = 0

    def update(self, bag_detections: list[Detection],
               persons: list[PersonTrack], now: float) -> None:
        """Match this frame's bag detections to tracks and refresh attention state."""
        bag_detections = dedup_detections(bag_detections)
        matches = match_detections(bag_detections, list(self.bags.values()),
                                   iou_gate=0.3,
                                   dist_gate=self.config.stationary_radius)
        for di, det in enumerate(bag_detections):
            bag = matches.get(di)
            if bag is None:
                bag = self._register(det, now)
            else:
                bag.box = smooth_box(bag.box, det.box, self.config.box_smoothing)
                bag.centroid = box_centroid(bag.box)
                bag.last_seen = now
            self._update_attention(bag, persons, now)
        self._merge_duplicate_tracks()

    def _merge_duplicate_tracks(self) -> None:
        """Collapse tracks stacked on the same physical bag.

        Detection flicker can strand a coasting track that a fresh track then
        piles onto — both would alert (and both would stamp a red box on the
        snapshot). The oldest track wins: it keeps its id, owner, and
        unattended timer, and adopts the duplicate's fresher sighting.
        """
        ids = sorted(self.bags)  # ascending id = oldest first
        dropped: set[int] = set()
        for i, keep_id in enumerate(ids):
            if keep_id in dropped:
                continue
            keeper = self.bags[keep_id]
            for dup_id in ids[i + 1:]:
                if dup_id in dropped:
                    continue
                dup = self.bags[dup_id]
                if calculate_iou(keeper.box, dup.box) < DEDUP_IOU:
                    continue
                if dup.last_seen > keeper.last_seen:
                    keeper.box = dup.box
                    keeper.centroid = dup.centroid
                    keeper.last_seen = dup.last_seen
                if keeper.owner_id is None:
                    keeper.owner_id = dup.owner_id
                dropped.add(dup_id)
                LOGGER.debug("Merged duplicate bag #%d into bag #%d", dup_id, keep_id)
        for bid in dropped:
            del self.bags[bid]

    def prune(self, now: float) -> list[int]:
        """Drop tracks not seen within the timeout. Returns removed ids."""
        expired = [
            bid for bid, b in self.bags.items()
            if now - b.last_seen > self.config.track_timeout_sec
        ]
        for bid in expired:
            LOGGER.debug("Dropping bag #%d (unseen %.1fs > timeout %.1fs)",
                         bid, now - self.bags[bid].last_seen, self.config.track_timeout_sec)
            del self.bags[bid]
        return expired

    def _register(self, det: Detection, now: float) -> TrackedBag:
        bag = TrackedBag(self._next_id, det.centroid, det.box, now, now)
        self.bags[bag.bag_id] = bag
        self._next_id += 1
        LOGGER.debug("Registered new bag #%d (%s) at %s", bag.bag_id, det.label, det.box)
        return bag

    def _update_attention(self, bag: TrackedBag,
                          persons: list[PersonTrack], now: float) -> None:
        """Owner-locked attention: only the bag's owner being near resets the timer.

        The owner is the person adopted while the bag is new (within owner_claim_sec).
        Once a bag has no owner — e.g. it entered the scene already abandoned — no
        bystander can claim it, so a passer-by or someone standing nearby never
        resets the unattended timer.
        """
        prox = self.config.person_proximity_px

        # Is the current owner (if any) near the bag right now?
        owner_present = False
        if bag.owner_id is not None:
            owner = next((p for p in persons if p.person_id == bag.owner_id), None)
            owner_present = owner is not None and distance(bag.centroid, owner.centroid) <= prox

        # While the bag is still new, adopt the nearest in-range person as its owner.
        if bag.owner_id is None and (now - bag.created_at) <= self.config.owner_claim_sec:
            in_range = [(distance(bag.centroid, p.centroid), p) for p in persons]
            nearest = min((pair for pair in in_range if pair[0] <= prox),
                          key=lambda pair: pair[0], default=(None, None))[1]
            if nearest is not None:
                bag.owner_id = nearest.person_id
                owner_present = True
                LOGGER.debug("Bag #%d adopted owner = person #%d", bag.bag_id, nearest.person_id)

        if owner_present:
            bag.unattended_start = None
            bag.alerted = False
        elif bag.unattended_start is None:
            bag.unattended_start = now


def save_snapshot_worker(frame_copy, path: Path, directory: Path, keep: int) -> None:
    if cv2.imwrite(str(path), frame_copy):
        LOGGER.info("Saved alert snapshot: %s", path)
    else:
        LOGGER.error("Failed to save alert snapshot: %s", path)
    try:
        # Cap the directory so an unattended deployment can't fill the SD card
        # (a full card takes the whole Pi down, not just the snapshots).
        snaps = sorted(directory.glob("alert_*.jpg"), key=lambda f: f.stat().st_mtime)
        for old in snaps[:max(0, len(snaps) - keep)]:
            old.unlink()
            LOGGER.debug("Pruned old snapshot: %s", old)
    except OSError as exc:
        LOGGER.warning("Snapshot pruning failed: %s", exc)


def save_snapshot(frame, bag: TrackedBag, config: Config) -> Path:
    config.snapshot_dir.mkdir(parents=True, exist_ok=True)
    # Wall clock for the filename only — tracking runs on the monotonic clock.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = config.snapshot_dir / f"alert_bag{bag.bag_id}_{stamp}.jpg"
    SNAPSHOT_EXECUTOR.submit(save_snapshot_worker, frame.copy(), path,
                             config.snapshot_dir, config.max_snapshots)
    return path


# Every alert is appended here (inside snapshot_dir) as a CSV row, giving a
# machine-readable record of incidents for later analysis or reporting.
EVENT_LOG_NAME = "events.csv"


def append_event_worker(csv_path: Path, row: list) -> None:
    try:
        header_needed = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if header_needed:
                writer.writerow(["time", "bag_id", "unattended_sec", "snapshot"])
            writer.writerow(row)
    except OSError as exc:
        LOGGER.warning("Could not append to event log %s: %s", csv_path, exc)


def _alert_due(bag: TrackedBag, config: Config, now: float) -> bool:
    """True if this bag should log + snapshot this frame (first alert or cooldown up)."""
    if bag.unattended_start is None:
        return False
    if now - bag.unattended_start < config.unattended_time_sec:
        return False
    return not bag.alerted or now - bag.last_alert_time >= config.alert_cooldown_sec


def _fire_alert(frame, bag: TrackedBag, config: Config, now: float,
                renderer: "Renderer") -> None:
    bag.alerted = True
    bag.last_alert_time = now
    duration = int(now - bag.unattended_start)
    LOGGER.warning("UNATTENDED BAG ALERT - bag #%d unattended for %ds",
                   bag.bag_id, duration)
    # Stamp the alert box here rather than relying on the render pass: the
    # renderer skips tracks unseen past draw_grace_sec, so a bag the detector
    # lost at alert time would otherwise save a snapshot with nothing marking
    # it. Redrawing over an already-drawn box is a no-op.
    renderer.draw_box(frame, bag.box, COLOR_ALERT,
                      f"ALERT! Bag #{bag.bag_id} unattended {duration}s",
                      thickness=3)
    snapshot_path = save_snapshot(frame, bag, config)
    SNAPSHOT_EXECUTOR.submit(append_event_worker, config.snapshot_dir / EVENT_LOG_NAME,
                             [time.strftime("%Y-%m-%d %H:%M:%S"), bag.bag_id,
                              duration, snapshot_path.name])


class Renderer:
    """Handles UI rendering to decouple it from tracking logic."""

    def __init__(self, config: Config):
        self.config = config

    def draw_box(self, frame, box: tuple[int, int, int, int], color, label: Optional[str] = None, thickness: int = 2) -> None:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        if label:
            cv2.putText(frame, label, (x, max(y - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_countdown(self, frame, box: tuple[int, int, int, int],
                       fraction: float, color) -> None:
        """Progress bar under the box filling toward the alert threshold."""
        x, y, w, h = box
        bar_h = 6
        y0 = y + h + 4
        if y0 + bar_h >= frame.shape[0]:   # would fall off-frame: draw above instead
            y0 = max(0, y - bar_h - 4)
        x0 = max(0, x)
        x1 = min(frame.shape[1] - 1, x + w)
        if x1 <= x0:
            return
        cv2.rectangle(frame, (x0, y0), (x1, y0 + bar_h), color, 1)
        fill = x0 + int((x1 - x0) * min(1.0, max(0.0, fraction)))
        if fill > x0:
            cv2.rectangle(frame, (x0, y0), (fill, y0 + bar_h), color, -1)

    def draw_hud(self, frame, person_count: int, bag_count: int, fps: float) -> None:
        cv2.rectangle(frame, (10, 10), (330, 72), (0, 0, 0), -1)
        cv2.putText(frame, f"People: {person_count}  Bags: {bag_count}", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HUD, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HUD, 2)

    def handle_bag(self, frame, bag: TrackedBag, now: float) -> None:
        """Draw a bag in the right state."""
        unseen = now - bag.last_seen
        # Don't draw a long-coasting track: avoids a "ghost" box lingering at the
        # old location after the bag has moved (or genuinely left). The track stays
        # alive until track_timeout_sec so its unattended timer keeps running.
        if unseen > self.config.draw_grace_sec:
            return
        # Mark tracks we're coasting on (no detection matched this frame).
        suffix = " (searching...)" if unseen > 0.5 else ""

        if bag.unattended_start is None:
            self.draw_box(frame, bag.box, COLOR_ATTENDED, f"Bag #{bag.bag_id} (Attended){suffix}")
            return

        duration = now - bag.unattended_start
        if duration >= self.config.unattended_time_sec:
            self.draw_box(frame, bag.box, COLOR_ALERT,
                     f"ALERT! Bag #{bag.bag_id} unattended {int(duration)}s", thickness=3)
            self.draw_countdown(frame, bag.box, 1.0, COLOR_ALERT)
        else:
            self.draw_box(frame, bag.box, COLOR_WARNING,
                     f"Bag #{bag.bag_id} unattended {int(duration)}s{suffix}")
            # Countdown bar fills toward the alert threshold so the state is
            # readable from across the room.
            self.draw_countdown(frame, bag.box,
                                duration / self.config.unattended_time_sec,
                                COLOR_WARNING)


def _sigterm_exit(signum, frame) -> None:
    """Route SIGTERM (systemctl stop) through the normal cleanup path."""
    raise SystemExit(0)


def run(config: Config) -> None:
    detector = IMX500Detector(config)

    picam2 = Picamera2(detector.camera_num)
    controls = {}
    if detector.inference_rate:
        controls["FrameRate"] = detector.inference_rate
    # RGB888 yields a 3-channel array in BGR order — exactly what OpenCV expects,
    # so the preview and saved snapshots have correct colours.
    # buffer_count: enough to ride out processing hiccups without queueing
    # stale frames (each buffer is a full frame of CMA memory, and a deep
    # queue means alerting on the past if the loop ever falls behind).
    cam_config = picam2.create_preview_configuration(
        main={"size": config.frame_size, "format": "RGB888"},
        controls=controls,
        buffer_count=6,
    )
    picam2.configure(cam_config)
    detector.show_progress()  # firmware upload progress bar on first run
    picam2.start()

    signal.signal(signal.SIGTERM, _sigterm_exit)  # systemd stop → clean shutdown

    tracker = BagTracker(config)
    person_tracker = PersonTracker(config)
    renderer = Renderer(config)
    LOGGER.info("Security monitor started (%s mode). Press 'q' in the window to quit.",
                "headless" if config.headless else "preview")

    fps = 0.0
    prev = time.monotonic()

    try:
        while True:
            # capture_request() keeps the frame and its detection metadata in
            # sync. The request is held while tracking runs so the pixel copy
            # (~1 MB per frame) can be skipped entirely on headless frames
            # where no alert snapshot is due.
            request = picam2.capture_request()
            try:
                metadata = request.get_metadata()

                # Monotonic clock for all track/alert timing: the Pi has no
                # RTC, so time.time() can jump hours when NTP syncs — which
                # would instantly "expire" every unattended timer.
                now = time.monotonic()
                detections = detector.detect(metadata, picam2)
                person_dets = [d for d in detections if d.label in config.person_labels]
                bag_dets = [d for d in detections if d.label in config.bag_labels]

                # Track people first (for stable ids), then bags (which need those ids
                # to recognise their owner). Pass all live person tracks — including ones
                # coasting through a brief miss — so the owner isn't lost to a blip.
                person_tracker.update(person_dets, now)
                person_tracker.prune(now)
                persons = list(person_tracker.tracks.values())

                tracker.update(bag_dets, persons, now)
                tracker.prune(now)

                alerts_due = [b for b in tracker.bags.values()
                              if _alert_due(b, config, now)]
                frame = (request.make_array("main")
                         if not config.headless or alerts_due else None)
            finally:
                request.release()

            dt = now - prev
            prev = now
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst

            # Drawing happens BEFORE the alerts fire so snapshots carry the
            # full annotations. frame is None only on headless frames with no
            # alert due, which need no drawing at all.
            if frame is not None:
                owner_ids = {b.owner_id for b in tracker.bags.values()
                             if b.owner_id is not None}
                visible_persons = 0
                for person in persons:
                    if now - person.last_seen > config.draw_grace_sec:
                        continue  # don't draw a coasting person's stale box
                    visible_persons += 1
                    tag = " (owner)" if person.person_id in owner_ids else ""
                    renderer.draw_box(frame, person.box, COLOR_PERSON,
                                      f"Person #{person.person_id}{tag}")
                for bag in tracker.bags.values():
                    renderer.handle_bag(frame, bag, now)
                renderer.draw_hud(frame, visible_persons, len(tracker.bags), fps)

            for bag in alerts_due:
                _fire_alert(frame, bag, config, now, renderer)

            if not config.headless:
                cv2.imshow("SecurePi - AI Security Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        LOGGER.info("Security monitor stopped.")


class _ArgFileParser(argparse.ArgumentParser):
    """ArgumentParser that also reads flags from a file: `securePi.py @site.args`.

    Inside the file, blank lines are skipped, everything after a `#` is a
    comment, and a flag and its value may share a line (`--unattended-time 60`).
    """

    def convert_arg_line_to_args(self, arg_line: str) -> list[str]:
        line = arg_line.split("#", 1)[0].strip()
        return line.split()


# Location presets live here; common.args in the same folder is the shared
# base applied automatically before whichever preset the user names.
PRESETS_DIR = Path(__file__).resolve().parent / "presets"


def _require_args_file(parser: argparse.ArgumentParser, raw_args: list[str]) -> None:
    """Refuse to run without an @<preset>.args file, listing the ones available."""
    if any(str(a).startswith("@") for a in raw_args):
        return
    presets = sorted(f.name for f in PRESETS_DIR.glob("*.args")
                     if f.name != "common.args")
    if presets:
        hint = "available presets: " + ", ".join(f"@{name}" for name in presets)
    else:
        hint = f"create one in {PRESETS_DIR} (see README)"
    parser.error(f"a settings file is required, e.g. `python securePi.py @lobby.args` — {hint}")


def _resolve_preset_refs(raw_args: list[str]) -> list[str]:
    """Let `@lobby.args` find presets/lobby.args regardless of working directory.

    An @path that exists as given (relative to the CWD, or absolute) is kept;
    otherwise it is looked up by name in PRESETS_DIR.
    """
    resolved = []
    for arg in raw_args:
        if isinstance(arg, str) and arg.startswith("@") and not Path(arg[1:]).exists():
            candidate = PRESETS_DIR / Path(arg[1:]).name
            if candidate.exists():
                arg = f"@{candidate}"
        resolved.append(arg)
    return resolved


def parse_args(argv=None) -> argparse.Namespace:
    p = _ArgFileParser(
        description="SecurePi — IMX500 unattended-bag security monitor.",
        fromfile_prefix_chars="@",
        epilog="A @<preset>.args settings file is required, e.g. `python securePi.py "
               "@lobby.args`. Presets live in the presets/ folder next to this script; "
               "presets/common.args is applied automatically as the base. Flags given "
               "after the preset override it.",
    )
    d = Config()  # single source of truth for defaults
    p.add_argument("--model", default=d.model_path,
                   help="Path to the IMX500 .rpk network (default: COCO SSD MobileNetV2).")
    p.add_argument("--labels", default=d.labels_path,
                   help="Optional path to a labels file overriding the model's built-in labels.")
    p.add_argument("--unattended-time", type=float, default=d.unattended_time_sec,
                   help="Seconds before an unattended bag triggers an alert "
                        "(default: %(default)s).")
    p.add_argument("--proximity", type=float, default=d.person_proximity_px,
                   help="Max pixel distance for the owner to attend a bag "
                        "(default: %(default)s).")
    p.add_argument("--owner-claim-time", type=float, default=d.owner_claim_sec,
                   help="Seconds after a bag first appears during which a nearby person is "
                        "adopted as its owner (default: %(default)s). Only the owner "
                        "attending resets the unattended timer; passers-by and bystanders "
                        "are ignored.")
    p.add_argument("--stationary-radius", type=float, default=d.stationary_radius,
                   help="Association radius in px: how far a bag may move between "
                        "detections and still match the same track (default: %(default)s). "
                        "Raise it if moving a bag spawns a second box; lower it if "
                        "two nearby bags get merged.")
    p.add_argument("--timeout", type=float, default=d.track_timeout_sec,
                   help="Coast window: seconds a bag track survives detection dropouts "
                        "before being dropped (default: %(default)s). Raise it if static "
                        "bags vanish; the unattended timer keeps running while coasting.")
    p.add_argument("--min-confidence", type=float, default=d.min_confidence,
                   help="Minimum detection confidence 0..1 (default: %(default)s).")
    p.add_argument("--box-smoothing", type=float, default=d.box_smoothing,
                   help="Weight of the newest detection when smoothing drawn boxes, "
                        "0..1 (default: %(default)s). Lower = steadier boxes on static "
                        "objects; 1.0 disables smoothing.")
    p.add_argument("--iou", type=float, default=d.iou,
                   help="NMS IoU threshold for nanodet models (default: %(default)s).")
    p.add_argument("--max-detections", type=int, default=d.max_detections,
                   help="Max detections for nanodet models (default: %(default)s).")
    p.add_argument("--bag-labels", nargs="+", default=sorted(d.bag_labels),
                   help="List of COCO labels to treat as bags (default: %(default)s).")
    p.add_argument("--person-labels", nargs="+", default=sorted(d.person_labels),
                   help="List of COCO labels to treat as people (default: %(default)s).")
    p.add_argument("--headless", action="store_true",
                   help="Run without a preview window; only save alert snapshots.")
    p.add_argument("--snapshot-dir", type=Path, default=d.snapshot_dir,
                   help="Directory for alert snapshots (default: %(default)s).")
    p.add_argument("--max-snapshots", type=int, default=d.max_snapshots,
                   help="Keep at most this many snapshots; oldest are deleted so long "
                        "runs can't fill the SD card (default: %(default)s).")
    p.add_argument("--alert-cooldown", type=float, default=d.alert_cooldown_sec,
                   help="Seconds between repeat alerts for the same bag "
                        "(default: %(default)s).")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    raw = sys.argv[1:] if argv is None else list(argv)
    expanded = _resolve_preset_refs(raw)
    base = PRESETS_DIR / "common.args"
    if base.exists():
        # Shared base first, so the user's preset and CLI flags override it.
        expanded = [f"@{base}"] + expanded
    args = p.parse_args(expanded)   # parse first so -h/--help still works
    _require_args_file(p, raw)
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    config = Config(
        unattended_time_sec=args.unattended_time,
        stationary_radius=args.stationary_radius,
        person_proximity_px=args.proximity,
        owner_claim_sec=args.owner_claim_time,
        track_timeout_sec=args.timeout,
        min_confidence=args.min_confidence,
        box_smoothing=args.box_smoothing,
        headless=args.headless,
        alert_cooldown_sec=args.alert_cooldown,
        snapshot_dir=args.snapshot_dir,
        max_snapshots=args.max_snapshots,
        model_path=args.model,
        labels_path=args.labels,
        iou=args.iou,
        max_detections=args.max_detections,
        bag_labels=set(args.bag_labels),
        person_labels=set(args.person_labels),
    )
    run(config)


if __name__ == "__main__":
    main()
