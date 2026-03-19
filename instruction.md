# AI Vision — 50-Stream Person Detection System
## Server Deployment Manual

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Ubuntu | 22.04 LTS |
| Docker Engine | 24.0+ |
| Docker Compose | v2.20+ (plugin) |
| NVIDIA Driver | 570.x+ |
| NVIDIA Container Toolkit | 1.14+ |
| CUDA (driver-level) | 12.x |
| Python 3 | 3.8+ (for ONNX export only) |

Verify GPU access from Docker before proceeding:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## Step 1: Folder Preparation

All persistent data lives under `/data/AI/AI_Vision/`. The root drive is capacity-constrained — never write outside this path.

```bash
# Create the full directory tree
sudo mkdir -p /data/AI/AI_Vision/{triton_models/yolov8/1,sample_videos,kafka_data,zookeeper_data,zookeeper_log}

# Set permissions so Docker containers (running as root or uid 1000) can write freely
sudo chmod -R 777 /data/AI/AI_Vision/kafka_data
sudo chmod -R 777 /data/AI/AI_Vision/zookeeper_data
sudo chmod -R 777 /data/AI/AI_Vision/zookeeper_log
```

Transfer the project files to the server:
```bash
# From your local machine:
scp -r ./AI_Vision/* user@server:/data/AI/AI_Vision/
```

Expected structure on the server:
```
/data/AI/AI_Vision/
├── docker-compose.yml
├── deepstream_app.py
├── nvinferserver_config.pbtxt
├── tracker_config.yml
├── mediamtx.yml
├── streams.txt
├── deepstream/
│   └── Dockerfile
├── frontend/
│   ├── Dockerfile
│   ├── index.html          # Vite entry point (must be in frontend root)
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       └── index.jsx
└── triton_models/
    └── yolov8/
        ├── config.pbtxt
        └── 1/
            └── (model.plan — built in Step 3)
```

---

## Step 2: Model Export (YOLOv8m ONNX)

The ONNX file must be exported dynamically using the Ultralytics CLI. Do not attempt to wget it from GitHub — the release assets are `.pt` weights, not ONNX.

```bash
cd /data/AI/AI_Vision

# Install the Ultralytics package
pip install ultralytics

# Export YOLOv8m to ONNX with dynamic batch and simplification
yolo export model=yolov8m.pt format=onnx dynamic=True simplify=True imgsz=640
```

This downloads `yolov8m.pt` automatically, then produces `yolov8m.onnx` in the current directory.

Verify the export:
```bash
ls -lh /data/AI/AI_Vision/yolov8m.onnx
# Expected: ~50 MB ONNX file
```

---

## Step 3: TensorRT Engine Compilation

Build the FP16 TensorRT engine inside the Triton container to guarantee version alignment. The engine interfaces use FP32 I/O tensors while computing internally at FP16 precision.

```bash
docker run --rm --gpus all \
  -v /data/AI/AI_Vision:/workspace \
  nvcr.io/nvidia/tritonserver:24.12-py3 \
  /usr/src/tensorrt/bin/trtexec \
    --onnx=/workspace/yolov8m.onnx \
    --saveEngine=/workspace/triton_models/yolov8/1/model.plan \
    --fp16 \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:50x3x640x640 \
    --maxShapes=images:50x3x640x640 \
    --verbose
```

**Notes:**
- The `--workspace` flag has been removed — it is deprecated in TensorRT 10.x and causes errors.
- The explicit path `/usr/src/tensorrt/bin/trtexec` is required as `trtexec` is not on the default `$PATH` inside this container.
- Engine compilation takes 5-15 minutes on the L40S.

Verify the engine:
```bash
ls -lh /data/AI/AI_Vision/triton_models/yolov8/1/model.plan
# Expected: ~50-100 MB .plan file
```

---

## Step 4: Build and Launch

```bash
cd /data/AI/AI_Vision

# Build the custom containers (DeepStream, Frontend)
docker compose up -d --build
```

On first launch, Docker will pull ~20 GB of base images (DeepStream alone is ~15 GB). Monitor progress with:
```bash
docker compose logs -f
```

---

## Step 5: Verification

### 5a. Check all containers are running
```bash
docker compose ps
```
Expected: all 6 containers (`triton`, `deepstream`, `kafka`, `zookeeper`, `mediamtx`, `frontend`) showing `Up` or `healthy`.

### 5b. Verify Triton model is loaded
```bash
docker exec triton curl -s http://localhost:8000/v2/models/yolov8 | python3 -m json.tool
```
Look for `"state": "READY"` in the output.

### 5c. Verify DeepStream pipeline
```bash
docker compose logs deepstream | tail -20
```
Look for: `[INFO] Pipeline PLAYING — processing streams`

### 5d. Verify Kafka is receiving detections
```bash
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic detections \
  --from-beginning \
  --max-messages 3
```

### 5e. Verify MediaMTX streams
```bash
docker exec mediamtx wget -qO- http://localhost:9997/v3/paths/list
```

### 5f. Open the Web Dashboard
Open a browser and navigate to:
```
http://<server-ip>:5258
```
You should see the stream grid with live annotated video feeds. Click any tile to expand to full view.

---

## Stopping the System

```bash
cd /data/AI/AI_Vision
docker compose down
```

To stop and wipe all persistent data:
```bash
docker compose down
rm -rf /data/AI/AI_Vision/kafka_data/* /data/AI/AI_Vision/zookeeper_data/* /data/AI/AI_Vision/zookeeper_log/*
```
