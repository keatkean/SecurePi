"""Off-device tests for securePi tracking, matching, alerting, and CLI parsing.

Runs on any machine — cv2 and picamera2 are not required (cv2 is stubbed
before import). Usage:

    python tests/test_securepi.py
"""
import contextlib
import io
import sys
import tempfile
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Stub cv2 so the module imports without OpenCV. imwrite writes a placeholder
# file so the snapshot-pruning test can observe real files on disk.
_cv2 = types.SimpleNamespace(
    imwrite=lambda path, img: Path(path).write_bytes(b"jpg") > 0,
)
sys.modules.setdefault("cv2", _cv2)

from securePi import (Config, Detection, PersonTracker, BagTracker,  # noqa: E402
                      match_detections, smooth_box, parse_args, _alert_due,
                      save_snapshot_worker, append_event_worker)

CFG = Config()


def test_smooth_box():
    old, new = (100, 100, 50, 50), (110, 110, 50, 50)
    assert smooth_box(old, new, 0.6) == (106, 106, 50, 50)
    far = (400, 400, 50, 50)
    assert smooth_box(old, far, 0.6) == far          # jump: snap, don't drag
    assert smooth_box(old, new, 1.0) == new          # 1.0 disables smoothing


def test_match_detections_order_independent():
    class T:
        def __init__(self, box):
            self.box = box
            self.centroid = (box[0] + box[2] / 2, box[1] + box[3] / 2)

    tA, tB = T((100, 100, 40, 40)), T((160, 100, 40, 40))
    d1 = Detection("bag", 0.9, (158, 100, 40, 40))   # clearly B, but first in list
    d2 = Detection("bag", 0.9, (102, 100, 40, 40))   # clearly A
    m = match_detections([d1, d2], [tA, tB], iou_gate=0.3, dist_gate=120)
    assert m[0] is tB and m[1] is tA


def test_person_tracker_stable_ids():
    pt = PersonTracker(CFG)
    pt.update([Detection("person", 0.9, (100, 100, 60, 120))], now=0.0)
    pt.update([Detection("person", 0.9, (104, 98, 62, 118))], now=0.1)
    assert len(pt.tracks) == 1
    track = next(iter(pt.tracks.values()))
    assert track.person_id == 0
    assert 100 <= track.box[0] <= 104                # smoothed between detections


def test_bag_tracker_no_swap():
    bt = BagTracker(CFG)
    bt.update([Detection("suitcase", 0.9, (100, 100, 40, 40)),
               Detection("suitcase", 0.9, (200, 100, 40, 40))], [], now=0.0)
    bt.update([Detection("suitcase", 0.9, (201, 101, 40, 40)),   # reversed order
               Detection("suitcase", 0.9, (99, 99, 40, 40))], [], now=0.1)
    assert len(bt.bags) == 2
    assert abs(bt.bags[0].centroid[0] - 120) < 5     # bag 0 stayed left
    assert abs(bt.bags[1].centroid[0] - 220) < 5     # bag 1 stayed right


def test_alert_due():
    bt = BagTracker(CFG)
    bt.update([Detection("suitcase", 0.9, (0, 0, 10, 10))], [], now=0.0)
    bag = bt.bags[0]
    bag.unattended_start = 0.0
    assert not _alert_due(bag, CFG, now=10.0)
    assert _alert_due(bag, CFG, now=CFG.unattended_time_sec + 1)
    bag.alerted = True
    bag.last_alert_time = CFG.unattended_time_sec + 1
    assert not _alert_due(bag, CFG, now=CFG.unattended_time_sec + 5)
    assert _alert_due(bag, CFG,
                      now=CFG.unattended_time_sec + 1 + CFG.alert_cooldown_sec)


def test_snapshot_pruning():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for i in range(5):
            (directory / f"alert_bag0_2026010{i}-000000.jpg").write_bytes(b"x")
        save_snapshot_worker(None, directory / "alert_bag1_new.jpg",
                             directory, keep=3)
        remaining = sorted(f.name for f in directory.glob("alert_*.jpg"))
        assert len(remaining) == 3, remaining
        assert "alert_bag1_new.jpg" in remaining     # newest survives


def test_event_log_append():
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "events.csv"
        append_event_worker(log, ["2026-07-07 10:00:00", 0, 12, "alert_bag0_x.jpg"])
        append_event_worker(log, ["2026-07-07 10:01:00", 1, 45, "alert_bag1_y.jpg"])
        lines = log.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == "time,bag_id,unattended_sec,snapshot"  # header once
        assert len(lines) == 3
        assert lines[2].startswith("2026-07-07 10:01:00,1,45,")


def test_demo_preset():
    args = parse_args(["@demo.args"])
    assert args.unattended_time == 10.0              # demo override
    assert args.owner_claim_time == 5.0              # demo override
    assert args.alert_cooldown == 15.0               # demo override
    assert args.proximity == 150.0                   # inherited from common.args
    assert str(args.snapshot_dir).replace("\\", "/") == "alerts/demo"


def test_presets_resolution_and_layering():
    args = parse_args(["@lobby.args"])               # short name, any CWD
    assert args.unattended_time == 60.0              # lobby override
    assert args.proximity == 120.0                   # lobby override
    assert args.alert_cooldown == 30.0               # inherited from common.args
    args = parse_args(["@kitchen.args"])
    assert args.unattended_time == 300.0
    assert args.proximity == 150.0                   # inherited
    args = parse_args(["@lobby.args", "--unattended-time", "15"])
    assert args.unattended_time == 15.0              # CLI beats preset


def test_preset_required():
    for argv in ([], ["--headless"]):
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                parse_args(argv)
            raise AssertionError(f"should have exited for argv={argv}")
        except SystemExit as e:
            assert e.code == 2
        assert "settings file is required" in err.getvalue()
        assert "@lobby.args" in err.getvalue()
    # -h still prints help without a preset
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            parse_args(["-h"])
    except SystemExit as e:
        assert e.code == 0
    assert "--unattended-time" in out.getvalue()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"{test.__name__} OK")
    print(f"ALL {len(tests)} TESTS PASSED")
