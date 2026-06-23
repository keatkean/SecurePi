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

Examples
--------
    python securePi.py
    python securePi.py --headless --unattended-time 60
    python securePi.py --model /usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk
"""

from __future__ import annotations

import argparse
import logging
import math
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
    frame_size: tuple[int, int] = (640, 480)
    headless: bool = False               # run without a preview window
    alert_cooldown_sec: float = 30.0     # seconds between repeat alerts per bag
    snapshot_dir: Path = Path("alerts")  # where alert snapshots are written
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
        x, y, w, h = self.box
        return (x + w / 2.0, y + h / 2.0)


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
            detections.append(Detection(label, float(score), (int(x), int(y), int(w), int(h))))
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
        matched: set[int] = set()
        for det in detections:
            track = self._match(det.box, det.centroid, matched)
            if track is None:
                track = PersonTrack(self._next_id, det.centroid, det.box, now)
                self.tracks[track.person_id] = track
                self._next_id += 1
            else:
                track.centroid = det.centroid
                track.box = det.box
                track.last_seen = now
            matched.add(track.person_id)

    def prune(self, now: float) -> None:
        expired = [pid for pid, t in self.tracks.items()
                   if now - t.last_seen > self.config.person_timeout_sec]
        for pid in expired:
            del self.tracks[pid]

    def _match(self, box, centroid, matched: set[int]) -> Optional[PersonTrack]:
        best: Optional[PersonTrack] = None
        best_iou = -1.0
        best_dist = self.config.person_match_radius
        for track in self.tracks.values():
            if track.person_id in matched:
                continue
            iou = calculate_iou(box, track.box)
            d = distance(centroid, track.centroid)
            if iou > 0.2:
                if iou > best_iou:
                    best_iou = iou
                    best = track
            elif best_iou == -1.0 and d < best_dist:
                best_dist = d
                best = track
        return best


class BagTracker:
    """Greedy nearest-centroid tracker with an unattended timer per bag."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.bags: dict[int, TrackedBag] = {}
        self._next_id = 0

    def update(self, bag_detections: list[Detection],
               persons: list[PersonTrack], now: float) -> None:
        """Match this frame's bag detections to tracks and refresh attention state."""
        matched: set[int] = set()
        for det in bag_detections:
            bag = self._match(det.centroid, det.box, matched)
            if bag is None:
                bag = self._register(det, now)
            else:
                bag.centroid = det.centroid
                bag.box = det.box
                bag.last_seen = now
            matched.add(bag.bag_id)
            self._update_attention(bag, persons, now)

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

    def _match(self, centroid: tuple[float, float], box: tuple[int, int, int, int], matched: set[int]) -> Optional[TrackedBag]:
        best: Optional[TrackedBag] = None
        best_score = -1.0 # higher is better (IoU)
        best_dist = self.config.stationary_radius
        for bag in self.bags.values():
            if bag.bag_id in matched:
                continue
            iou = calculate_iou(box, bag.box)
            d = distance(centroid, bag.centroid)
            # Prefer IoU if they overlap significantly
            if iou > 0.3:
                if iou > best_score:
                    best_score = iou
                    best = bag
            # Fallback to distance if no good IoU
            elif best_score == -1.0 and d < best_dist:
                best_dist = d
                best = bag
        return best

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
            nearest = min((p for p in persons if distance(bag.centroid, p.centroid) <= prox),
                          key=lambda p: distance(bag.centroid, p.centroid), default=None)
            if nearest is not None:
                bag.owner_id = nearest.person_id
                owner_present = True
                LOGGER.debug("Bag #%d adopted owner = person #%d", bag.bag_id, nearest.person_id)

        if owner_present:
            bag.unattended_start = None
            bag.alerted = False
        elif bag.unattended_start is None:
            bag.unattended_start = now


def save_snapshot_worker(frame_copy, path: Path) -> None:
    if cv2.imwrite(str(path), frame_copy):
        LOGGER.info("Saved alert snapshot: %s", path)
    else:
        LOGGER.error("Failed to save alert snapshot: %s", path)


def save_snapshot(frame, bag: TrackedBag, config: Config, now: float) -> None:
    config.snapshot_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    path = config.snapshot_dir / f"alert_bag{bag.bag_id}_{stamp}.jpg"
    frame_copy = frame.copy()
    SNAPSHOT_EXECUTOR.submit(save_snapshot_worker, frame_copy, path)


def _maybe_alert(frame, bag: TrackedBag, config: Config, now: float, duration: float) -> None:
    """Log + snapshot on the first alert, then at most once per cooldown."""
    first = not bag.alerted
    repeat_due = now - bag.last_alert_time >= config.alert_cooldown_sec
    if not (first or repeat_due):
        return
    bag.alerted = True
    bag.last_alert_time = now
    LOGGER.warning("UNATTENDED BAG ALERT - bag #%d unattended for %ds",
                   bag.bag_id, int(duration))
    save_snapshot(frame, bag, config, now)


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
        else:
            self.draw_box(frame, bag.box, COLOR_WARNING,
                     f"Bag #{bag.bag_id} unattended {int(duration)}s{suffix}")


def run(config: Config) -> None:
    detector = IMX500Detector(config)

    picam2 = Picamera2(detector.camera_num)
    controls = {}
    if detector.inference_rate:
        controls["FrameRate"] = detector.inference_rate
    # RGB888 yields a 3-channel array in BGR order — exactly what OpenCV expects,
    # so the preview and saved snapshots have correct colours.
    cam_config = picam2.create_preview_configuration(
        main={"size": config.frame_size, "format": "RGB888"},
        controls=controls,
        buffer_count=12,
    )
    picam2.configure(cam_config)
    detector.show_progress()  # firmware upload progress bar on first run
    picam2.start()

    tracker = BagTracker(config)
    person_tracker = PersonTracker(config)
    renderer = Renderer(config)
    LOGGER.info("Security monitor started (%s mode). Press 'q' in the window to quit.",
                "headless" if config.headless else "preview")

    fps = 0.0
    prev = time.monotonic()

    try:
        while True:
            # capture_request() keeps the frame and its detection metadata in sync.
            request = picam2.capture_request()
            try:
                frame = request.make_array("main")
                metadata = request.get_metadata()
            finally:
                request.release()

            now = time.time()
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

            owner_ids = {b.owner_id for b in tracker.bags.values() if b.owner_id is not None}
            for person in persons:
                if now - person.last_seen > config.draw_grace_sec:
                    continue  # don't draw a coasting person's stale box
                tag = " (owner)" if person.person_id in owner_ids else ""
                renderer.draw_box(frame, person.box, COLOR_PERSON,
                                  f"Person #{person.person_id}{tag}")
            for bag in tracker.bags.values():
                renderer.handle_bag(frame, bag, now)

                # Check alert independently of drawing
                if bag.unattended_start is not None:
                    duration = now - bag.unattended_start
                    if duration >= config.unattended_time_sec:
                        _maybe_alert(frame, bag, config, now, duration)

            t = time.monotonic()
            dt = t - prev
            prev = t
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst

            renderer.draw_hud(frame, len(person_dets), len(tracker.bags), fps)

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


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SecurePi — IMX500 unattended-bag security monitor.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Path to the IMX500 .rpk network (default: COCO SSD MobileNetV2).")
    p.add_argument("--labels", default=None,
                   help="Optional path to a labels file overriding the model's built-in labels.")
    p.add_argument("--unattended-time", type=float, default=120.0,
                   help="Seconds before an unattended bag triggers an alert (default: 120).")
    p.add_argument("--proximity", type=float, default=150.0,
                   help="Max pixel distance for the owner to attend a bag (default: 150).")
    p.add_argument("--owner-claim-time", type=float, default=3.0,
                   help="Seconds after a bag first appears during which a nearby person is "
                        "adopted as its owner (default: 3). Only the owner attending resets "
                        "the unattended timer; passers-by and bystanders are ignored.")
    p.add_argument("--stationary-radius", type=float, default=120.0,
                   help="Association radius in px: how far a bag may move between "
                        "detections and still match the same track (default: 120). "
                        "Raise it if moving a bag spawns a second box; lower it if "
                        "two nearby bags get merged.")
    p.add_argument("--timeout", type=float, default=10.0,
                   help="Coast window: seconds a bag track survives detection dropouts "
                        "before being dropped (default: 10). Raise it if static bags "
                        "vanish; the unattended timer keeps running while coasting.")
    p.add_argument("--min-confidence", type=float, default=0.5,
                   help="Minimum detection confidence 0..1 (default: 0.5).")
    p.add_argument("--iou", type=float, default=0.65,
                   help="NMS IoU threshold for nanodet models (default: 0.65).")
    p.add_argument("--max-detections", type=int, default=10,
                   help="Max detections for nanodet models (default: 10).")
    p.add_argument("--bag-labels", nargs="+", default=list(DEFAULT_BAG_LABELS),
                   help="List of COCO labels to treat as bags.")
    p.add_argument("--person-labels", nargs="+", default=list(DEFAULT_PERSON_LABELS),
                   help="List of COCO labels to treat as people.")
    p.add_argument("--headless", action="store_true",
                   help="Run without a preview window; only save alert snapshots.")
    p.add_argument("--snapshot-dir", type=Path, default=Path("alerts"),
                   help="Directory for alert snapshots (default: ./alerts).")
    p.add_argument("--alert-cooldown", type=float, default=30.0,
                   help="Seconds between repeat alerts for the same bag (default: 30).")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return p.parse_args(argv)


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
        headless=args.headless,
        alert_cooldown_sec=args.alert_cooldown,
        snapshot_dir=args.snapshot_dir,
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
