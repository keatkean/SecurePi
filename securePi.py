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
    stationary_radius: float = 50.0      # px a bag may shift and stay the same track
    person_proximity_px: float = 150.0   # px within which a bag counts as attended
    track_timeout_sec: float = 3.0       # drop a track unseen for this long
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
    unattended_start: Optional[float] = None
    alerted: bool = False
    last_alert_time: float = 0.0


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


class BagTracker:
    """Greedy nearest-centroid tracker with an unattended timer per bag."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.bags: dict[int, TrackedBag] = {}
        self._next_id = 0

    def update(self, bag_detections: list[Detection],
               person_centroids: list[tuple[float, float]], now: float) -> None:
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
            self._update_attention(bag, person_centroids, now)

    def prune(self, now: float) -> list[int]:
        """Drop tracks not seen within the timeout. Returns removed ids."""
        expired = [
            bid for bid, b in self.bags.items()
            if now - b.last_seen > self.config.track_timeout_sec
        ]
        for bid in expired:
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
        bag = TrackedBag(self._next_id, det.centroid, det.box, now)
        self.bags[bag.bag_id] = bag
        self._next_id += 1
        LOGGER.debug("Registered new bag #%d (%s) at %s", bag.bag_id, det.label, det.box)
        return bag

    def _update_attention(self, bag: TrackedBag,
                          person_centroids: list[tuple[float, float]], now: float) -> None:
        nearest = min(
            (distance(bag.centroid, p) for p in person_centroids),
            default=float("inf"),
        )
        if nearest > self.config.person_proximity_px:
            # No one nearby: start the unattended timer if it isn't running.
            if bag.unattended_start is None:
                bag.unattended_start = now
        else:
            # A person is close: reset the timer and clear any alert latch.
            bag.unattended_start = None
            bag.alerted = False


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
        if bag.unattended_start is None:
            self.draw_box(frame, bag.box, COLOR_ATTENDED, f"Bag #{bag.bag_id} (Attended)")
            return

        duration = now - bag.unattended_start
        if duration >= self.config.unattended_time_sec:
            self.draw_box(frame, bag.box, COLOR_ALERT,
                     f"ALERT! Bag #{bag.bag_id} unattended {int(duration)}s", thickness=3)
        else:
            self.draw_box(frame, bag.box, COLOR_WARNING,
                     f"Bag #{bag.bag_id} unattended {int(duration)}s")


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
            persons = [d for d in detections if d.label in config.person_labels]
            bags = [d for d in detections if d.label in config.bag_labels]
            person_centroids = [p.centroid for p in persons]

            tracker.update(bags, person_centroids, now)
            tracker.prune(now)

            for person in persons:
                renderer.draw_box(frame, person.box, COLOR_PERSON, f"Person {person.score:.2f}")
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

            renderer.draw_hud(frame, len(persons), len(tracker.bags), fps)

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
                   help="Max pixel distance for a bag to count as attended (default: 150).")
    p.add_argument("--stationary-radius", type=float, default=50.0,
                   help="Pixel radius for matching a bag to the same track (default: 50).")
    p.add_argument("--timeout", type=float, default=3.0,
                   help="Seconds before an unseen bag is dropped (default: 3).")
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
