# 🚦 Traffic Violation Detection System

> **Dual-model, parallel-inference** system for comprehensive traffic violation detection with number plate OCR.  
> Built for the **Flipkart Grid Hackathon R2**.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   FastAPI Server / CLI                        │
│            Accepts: image / video / live stream               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   Input Frame ───► ThreadPoolExecutor (parallel)             │
│                     │                    │                    │
│                     ▼                    ▼                    │
│              ┌─────────────┐    ┌──────────────┐             │
│              │   MODEL 1   │    │   MODEL 2    │             │
│              │  Violation   │    │  Plate OCR   │             │
│              │  Detection   │    │  Detection   │             │
│              │  (YOLOv8)   │    │  + EasyOCR   │             │
│              └──────┬──────┘    └──────┬───────┘             │
│                     │                  │                      │
│                     └────────┬─────────┘                     │
│                              ▼                               │
│                    ┌──────────────────┐                      │
│                    │  Result Merger   │                      │
│                    │  (spatial match) │                      │
│                    └──────────────────┘                      │
│                              │                               │
│                    ┌─────────┼──────────┐                   │
│                    ▼         ▼          ▼                    │
│              Violation  Plate No.  Confidence                │
│              Report     + OCR      Scores                    │
└──────────────────────────────────────────────────────────────┘
```

## Violations Detected

| # | Violation | Applies To | Detection Method |
|---|-----------|------------|------------------|
| 1 | **No Helmet** | 2-wheelers, 3-wheelers | YOLO `without_helmet` + vehicle association |
| 2 | **Triple Riding** | 2-wheelers | Count riders overlapping with motorcycle bbox |
| 3 | **Red Light Jumping** | All vehicles | Red light + vehicle past stop line |
| 4 | **Illegal Parking** | All vehicles | Stationary vehicle in no-parking zone |
| 5 | **Stop Line Violation** | All vehicles | Vehicle past stop line on non-green |
| 6 | **Modified Vehicle** | All vehicles | YOLO `modified` class detection |

Each violation includes a **composite confidence score** (0.0–1.0) and severity level (HIGH/MEDIUM/LOW).

---

## Quick Start

### 1. Install Dependencies

```bash
cd code
pip install -r requirements.txt
```

### 2. Train Models on Kaggle

**Model 1 — Violation Detection:**
1. Upload `datasets/combined-full/` to Kaggle as a dataset
2. Create a new notebook with **GPU** enabled
3. Copy-paste `training/train_violation_model.py` into a cell → Run
4. Download `violation_model.pt` from the Output tab
5. Place in `code/models/violation_model.pt`

**Model 2 — Plate Detection:**
1. Find a license plate dataset on Kaggle (search "license plate detection YOLO")
2. Create a notebook with **GPU** enabled
3. Copy-paste `training/train_plate_ocr_model.py` into a cell → Run
4. Download `plate_model.pt` from the Output tab
5. Place in `code/models/plate_model.pt`

> **Shortcut**: If you already have `best.pt` in the parent directory, Model 1 will auto-detect it as a fallback.

### 3. Run Detection

```bash
# Single image
python run.py --source path/to/image.jpg

# Video
python run.py --source path/to/video.mp4 --stride 3

# Webcam
python run.py --source 0

# Start API server
python run.py --server

# Analyze dataset quality
python run.py --analyze-dataset ../datasets/combined-full --type violation
```

---

## API Server

### Start Server
```bash
python run.py --server
# or
python -m server.app
```

Server runs at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Server health + model status |
| `POST` | `/api/analyze/image` | Upload image → violations + plates |
| `POST` | `/api/analyze/video` | Upload video → per-frame violations |
| `WS` | `/api/analyze/stream` | Live stream via WebSocket |
| `POST` | `/api/analyze/dataset` | Validate dataset quality |

### Example — Analyze Image (curl)
```bash
curl -X POST http://localhost:8000/api/analyze/image \
  -F "file=@traffic_image.jpg" \
  -F "return_annotated=true"
```

### Example — Analyze Image (Python)
```python
import requests

with open("traffic_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/analyze/image",
        files={"file": f},
        params={"return_annotated": True},
    )

result = response.json()
for v in result["report"]["violations"]:
    print(f"🚨 {v['type']} — Confidence: {v['confidence']:.0%} [{v['severity']}]")
    if v.get("plate"):
        print(f"   Plate: {v['plate']['text']}")
```

### Example — Live Stream (WebSocket)
```python
import asyncio
import websockets
import json

async def stream():
    async with websockets.connect("ws://localhost:8000/api/analyze/stream") as ws:
        await ws.send(json.dumps({"stream_url": "rtsp://camera-ip/stream", "stride": 5}))
        while True:
            data = json.loads(await ws.recv())
            if data.get("violations"):
                print(f"Frame {data['frame_id']}: {len(data['violations'])} violations")

asyncio.run(stream())
```

---

## Confidence Scoring

Each violation produces a composite confidence:

```
composite = w1 × detection_conf + w2 × spatial_conf + w3 × temporal_conf
```

| Level | Range | Meaning |
|-------|-------|---------|
| 🔴 **HIGH** | ≥ 0.80 | Strong evidence, auto-flag |
| 🟡 **MEDIUM** | 0.50–0.79 | Likely violation, review recommended |
| 🟢 **LOW** | < 0.50 | Possible, low certainty |

---

## Project Structure

```
code/
├── run.py                          # CLI entry point
├── config.py                       # Central configuration
├── pipeline.py                     # Dual-model parallel pipeline
├── visualizer.py                   # Frame annotation
├── requirements.txt
│
├── models/                         # ← Drop trained .pt files here
│   ├── violation_model.pt          # Model 1
│   └── plate_model.pt             # Model 2
│
├── engine/                         # Core inference
│   ├── violation_detector.py       # Model 1 wrapper
│   ├── plate_reader.py            # Model 2 + EasyOCR
│   ├── tracker.py                  # Object tracking
│   └── spatial_utils.py           # Geometry utilities
│
├── violations/                     # Rule-based analyzers
│   ├── helmet.py                   # No helmet
│   ├── tripling.py                # Triple riding
│   ├── red_light.py               # Red light jumping
│   ├── illegal_parking.py        # Illegal parking
│   ├── stop_line.py               # Stop line crossing
│   └── vehicle_mods.py           # Modified vehicles
│
├── server/                         # FastAPI server
│   ├── app.py                     # Application setup
│   ├── routes.py                  # API endpoints
│   └── schemas.py                 # Pydantic models
│
├── training/                       # Kaggle training scripts
│   ├── train_violation_model.py   # Model 1 training
│   └── train_plate_ocr_model.py   # Model 2 training
│
└── analysis/                       # Dataset validation
    ├── analyze_violation_dataset.py
    └── analyze_plate_dataset.py
```

---

## Configuration

Edit `config.py` to customize:

- **Model paths**: Auto-detected from `models/` directory
- **Confidence thresholds**: Per-violation thresholds
- **No-parking zones**: Define polygon zones for parking violation detection
- **Inference settings**: Image size, device, max detections
- **Server settings**: Host, port, upload size limits

---

## 21 Detection Classes (Model 1)

| ID | Class | Category |
|----|-------|----------|
| 0-9 | Bus, Car, Motorcycle, Truck, Three Wheeler, Tractor, Van, Vikram, Two Wheeler, Bike | Vehicles |
| 10-13 | With Helmet, Without Helmet, Rider With Helmet, Rider Without Helmet | Helmet |
| 14-17 | Red Light, Green Light, Yellow Light, Traffic Light | Signals |
| 18-20 | Stop Line, Fixed Obstacle, Modified | Infrastructure |
