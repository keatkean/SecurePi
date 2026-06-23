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
2. **Track** — each detected bag is matched to an existing track using Intersection 
   over Union (IoU) or nearest centroid (within `--stationary-radius` px). 
   Tracks unseen for `--timeout` seconds are dropped.
3. **Attend** — for every tracked bag, SecurePi finds the nearest person. If the
   nearest person is farther than `--proximity` px (or there is no person), the
   bag's *unattended* timer starts; if a person comes close, the timer resets.
4. **Alert** — when a bag stays unattended for `--unattended-time` seconds it
   enters the **ALERT** state: a warning is logged and a JPEG snapshot is written
   to the snapshot directory. Each alert fires once, then repeats at most once
   per `--alert-cooldown` seconds until a person returns.

### On-screen states

| State | Box colour | Meaning |
|-------|-----------|---------|
| Attended | cyan | A person is within `--proximity` of the bag |
| Unattended (counting) | orange | No person nearby; timer running, below threshold |
| ALERT | red | Unattended ≥ `--unattended-time` seconds |
| Person | green | A detected person |

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

```bash
# Live preview window (press 'q' to quit)
python securePi.py

# Headless — no window; alert snapshots only (good for SSH / systemd)
python securePi.py --headless

# Tune detection and alerting
python securePi.py --unattended-time 60 --proximity 200 --min-confidence 0.6 -v

# Use a different IMX500 network and/or labels file (see [MODELS.md](MODELS.md) for options)
python securePi.py --model /usr/share/imx500-models/<network>.rpk --labels labels.txt

# Customize which objects to monitor and who can attend to them (see [LABELS.md](LABELS.md) for all 80 options)
python securePi.py --bag-labels laptop --person-labels person dog
```

> **First run** uploads the network firmware to the sensor — this can take ~30s
> and shows a progress bar before the camera starts.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `…/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk` | IMX500 `.rpk` network |
| `--labels` | *(model built-in)* | Override the model's label file |
| `--unattended-time` | `120` | Seconds unattended before an alert |
| `--proximity` | `150` | Max px between bag and person to count as attended |
| `--stationary-radius` | `50` | Px a bag may move and stay the same track |
| `--timeout` | `10` | Coast window: seconds a track survives detection dropouts before being dropped (timer keeps running while coasting) |
| `--min-confidence` | `0.5` | Minimum detection confidence (0–1) |
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
ExecStart=/usr/bin/python3 /home/pi/SecurePi/securePi.py --headless --snapshot-dir /home/pi/SecurePi/alerts
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
├── requirements.txt   # dependency notes
├── README.md          # this file
├── LABELS.md          # list of supported tracking objects
├── MODELS.md          # explanation of available .rpk networks
└── alerts/            # alert snapshots (created on first alert)
```

## Notes & limitations

- Tracking uses Intersection over Union (IoU) and a nearest-centroid fallback — robust for a few
  bags in a fairly static scene, not a full multi-object tracker. Crossing paths
  or heavy occlusion can swap or drop IDs.
- "Attended" is purely proximity-based; it does not verify *whose* bag it is.
- The default model uses COCO classes. By default, only `person`, `backpack`, 
  `handbag`, and `suitcase` are tracked, but this is fully configurable via CLI.
