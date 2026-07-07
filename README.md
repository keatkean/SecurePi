# SecurePi

An unattended-bag security monitor for the **Raspberry Pi AI Camera (IMX500)**.

SecurePi runs object detection *on the IMX500 sensor* to spot people and bags
(backpack, handbag, suitcase), tracks each bag across frames, and raises an
alert when a bag has been left **unattended** — i.e. no person nearby — for
longer than a configurable time. Alerts are logged and saved as timestamped
snapshots, and the live view annotates each bag's state.

![status: backpack tracked, timer running, alert on timeout]

---

## How it works

1. **Detect** — the IMX500 runs the neural network on-sensor; SecurePi reads the
   output tensors from the frame metadata (`get_outputs` +
   `convert_inference_coords`), exactly like the official Raspberry Pi IMX500
   object-detection demo. No inference runs on the Pi's CPU.
2. **Track** — bags *and* people are matched to existing tracks by scoring
   every detection/track pair at once and assigning best-first: Intersection
   over Union (IoU) overlaps always win, with nearest centroid as the fallback
   (bags within `--stationary-radius` px). This global matching is independent
   of detection order, so two nearby objects can't swap tracks. Matched boxes
   are smoothed across frames (`--box-smoothing`) to damp detector jitter, and
   snap instantly when an object genuinely moves. People get stable ids so a
   bag can recognise its owner. Tracks unseen for `--timeout` seconds (bags)
   are dropped.
3. **Attend (owner-locked)** — when a bag first appears, the person near it
   (within `--proximity` px) during the first `--owner-claim-time` seconds is
   adopted as its **owner**. Only that owner being near resets the *unattended*
   timer. Anyone else — a passer-by, or someone standing nearby on their phone —
   is ignored, so they can neither reset the timer nor "claim" a bag that arrived
   with no owner. If the owner leaves, the timer runs.
4. **Alert** — when a bag stays unattended for `--unattended-time` seconds it
   enters the **ALERT** state: a warning is logged and a fully annotated JPEG
   snapshot (boxes, labels, HUD) is written to the snapshot directory. Each
   alert fires once, then repeats at most once per `--alert-cooldown` seconds
   until a person returns.

### On-screen states

| State | Box colour | Meaning |
|-------|-----------|---------|
| Attended | cyan | The bag's **owner** is within `--proximity` |
| Unattended (counting) | orange | Owner away (or none); timer running, below threshold |
| ALERT | red | Unattended ≥ `--unattended-time` seconds |
| Person | green | A tracked person (labelled `(owner)` if they own a bag) |

---

## Requirements

- Raspberry Pi (Bookworm) with the **Raspberry Pi AI Camera (IMX500)** attached.
- The Pi camera + AI stack and the IMX500 firmware/models:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv imx500-all
```

`imx500-all` installs the sensor firmware and the model files under
`/usr/share/imx500-models/`, including the default network
`imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk` (COCO classes).

> Installing via `apt` is recommended over `pip` so `picamera2`/`opencv` match
> the system libcamera/Qt builds. See [requirements.txt](requirements.txt) for
> reference if you use a non-apt environment.

---

## Usage

Every run takes a `@<preset>.args` settings file (enforced — running without
one prints the available presets). Extra flags after the preset override it:

```bash
# Live preview window (press 'q' to quit)
python securePi.py @lobby.args

# Headless — no window; alert snapshots only (good for SSH / systemd).
# Skips all frame annotation except when a snapshot is saved, so it's the
# cheapest way to run.
python securePi.py @lobby.args --headless

# Tune detection and alerting for one run without editing the preset
python securePi.py @lobby.args --unattended-time 60 --proximity 200 --min-confidence 0.6 -v

# Use a different IMX500 network and/or labels file (see [MODELS.md](MODELS.md) for options)
python securePi.py @lobby.args --model /usr/share/imx500-models/<network>.rpk --labels labels.txt

# Customize which objects to monitor and who can attend to them (see [LABELS.md](LABELS.md) for all 80 options)
python securePi.py @lobby.args --bag-labels laptop --person-labels person dog
```

> **First run** uploads the network firmware to the sensor — this can take ~30s
> and shows a progress bar before the camera starts.

### Settings files — no flags to remember

All settings live in preset files; the script refuses to start without one, so
a camera can never accidentally run with unintended defaults. Pass a preset
with `@`:

```bash
python securePi.py @lobby.args      # busy lobby camera
python securePi.py @kitchen.args    # staff kitchen camera
```

Presets live in the [presets/](presets/) folder and are layered:

- [presets/common.args](presets/common.args) — the shared base: every setting,
  commented. It is **applied automatically** before whichever preset you name.
- [presets/lobby.args](presets/lobby.args),
  [presets/kitchen.args](presets/kitchen.args) — per-location files containing
  only the overrides for that spot. Copy one to add a new location:

```text
# SecurePi — LOBBY preset:  python securePi.py @lobby.args
--unattended-time 60
--proximity 120
--snapshot-dir alerts/lobby
```

You don't need to type the folder: `@lobby.args` is looked up in `presets/`
automatically (an explicit path like `@presets/lobby.args` or an absolute path
also works, from any directory).

The file format: one flag per line (the value goes on the same line), blank
lines are skipped, and anything after `#` is a comment. Later flags win, so a
location preset overrides `common.args`, and the command line overrides both:

```bash
python securePi.py @lobby.args --headless --unattended-time 30
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `…/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk` | IMX500 `.rpk` network |
| `--labels` | *(model built-in)* | Override the model's label file |
| `--unattended-time` | `120` | Seconds unattended before an alert |
| `--proximity` | `150` | Max px between bag and its owner to count as attended |
| `--owner-claim-time` | `3` | Seconds after a bag appears during which a nearby person is adopted as its owner |
| `--stationary-radius` | `120` | Association radius: px a bag may move between detections and still match the same track |
| `--timeout` | `10` | Coast window: seconds a track survives detection dropouts before being dropped (timer keeps running while coasting) |
| `--min-confidence` | `0.5` | Minimum detection confidence (0–1) |
| `--box-smoothing` | `0.6` | Weight of the newest detection when smoothing drawn boxes (0–1). Lower = steadier boxes on static objects; `1.0` disables smoothing |
| `--iou` | `0.65` | NMS IoU threshold (nanodet models) |
| `--max-detections` | `10` | Max detections (nanodet models) |
| `--bag-labels` | `backpack handbag suitcase` | List of COCO labels to treat as bags |
| `--person-labels` | `person` | List of COCO labels to treat as people |
| `--headless` | off | Run without a preview window |
| `--snapshot-dir` | `./alerts` | Where alert snapshots are written |
| `--alert-cooldown` | `30` | Seconds between repeat alerts per bag |
| `-v`, `--verbose` | off | Debug logging |

All thresholds are in **pixels at the configured frame size** (640×480 by
default). If you change the resolution, scale `--proximity` and
`--stationary-radius` accordingly.

---

## Running as a service (headless)

To run automatically on boot, create a systemd unit, e.g.
`/etc/systemd/system/securepi.service`:

```ini
[Unit]
Description=SecurePi unattended-bag monitor
After=multi-user.target

[Service]
User=pi
WorkingDirectory=/home/pi/SecurePi
ExecStart=/usr/bin/python3 /home/pi/SecurePi/securePi.py @lobby.args --headless
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now securepi.service
journalctl -u securepi.service -f      # follow logs / alerts
```

---

## Project layout

```
SecurePi/
├── securePi.py        # the monitor (detector + tracker + alerting)
├── presets/           # settings files — run with `python securePi.py @<name>.args`
│   ├── common.args    # shared base (every flag, commented) — applied automatically
│   ├── lobby.args     # location preset: overrides for the lobby camera
│   ├── kitchen.args   # location preset: overrides for the kitchen camera
│   └── carpark.args   # location preset: overrides for the carpark camera
├── requirements.txt   # dependency notes
├── README.md          # this file
├── LABELS.md          # list of supported tracking objects
├── MODELS.md          # explanation of available .rpk networks
└── alerts/            # alert snapshots (created on first alert)
```

## Notes & limitations

- Tracking uses best-first global matching on Intersection over Union (IoU)
  with a nearest-centroid fallback — robust for a few bags in a fairly static
  scene, not a full multi-object tracker (no motion prediction or appearance
  re-identification). Heavy occlusion or objects crossing while undetected can
  still swap or drop IDs.
- Ownership is matched by position, not by face/appearance re-identification. If
  the owner walks **out of frame** and returns, they're treated as a new person,
  so the bag keeps counting as unattended (errs toward alerting — the safe side
  for security). The owner is recognised across small moves and brief misses.
- The default model uses COCO classes. By default, only `person`, `backpack`, 
  `handbag`, and `suitcase` are tracked, but this is fully configurable via CLI.
