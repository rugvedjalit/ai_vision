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

Verify GPU access from Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## Step 1: Create Directory Structure

All persistent data lives under `/data/AI/AI_Vision/`. The root drive is capacity-constrained — never write outside this path.

```bash
sudo mkdir -p /data/AI/AI_Vision/{triton_models/yolov8/1,sample_videos,kafka_data,zookeeper_data,zookeeper_log}
sudo chown -R $USER:$USER /data/AI/AI_Vision
```

---

## Step 2: Transfer Project Files

Copy the entire project directory to the server:

```bash
# From your local machine:
scp -r ./AI_Vision/* user@server:/data/AI/AI_Vision/
```

The server should have this structure:
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
│   ├── package.json
│   ├── vite.config.js
│   ├── public/
│   │   └── index.html
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

## Step 3: Prepare the YOLOv8 TensorRT Engine

### 3a. Download YOLOv8m ONNX Model

```bash
cd /data/AI/AI_Vision

# Option A: Export from Ultralytics (if Python is available on host)
pip install ultralytics
yolo export model=yolov8m.pt format=onnx dynamic=True simplify=True imgsz=640

# Option B: Download pre-exported ONNX (if available from your model registry)
# wget <your-model-registry-url>/yolov8m.onnx
```

### 3b. Build TensorRT Engine (FP16, Dynamic Batch)

**CRITICAL:** Build the engine using the same Triton container image to guarantee TensorRT version alignment.

```bash
docker run --rm --gpus all \
  -v /data/AI/AI_Vision:/workspace \
  nvcr.io/nvidia/tritonserver:24.12-py3 \
  trtexec \
    --onnx=/workspace/yolov8m.onnx \
    --saveEngine=/workspace/triton_models/yolov8/1/model.plan \
    --fp16 \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:50x3x640x640 \
    --maxShapes=images:50x3x640x640 \
    --workspace=4096 \
    --verbose
```

**Expected output:** Engine build takes 5-15 minutes on the L40S. Look for:
```
[INFO] Engine built successfully.
[INFO] Saved engine to /workspace/triton_models/yolov8/1/model.plan
```

### 3c. Verify Engine File

```bash
ls -lh /data/AI/AI_Vision/triton_models/yolov8/1/model.plan
# Should be ~50-100 MB
```

---

## Step 4: Add Demo Videos (Optional)

For testing without live RTSP cameras, place MP4 files in the sample_videos directory:

```bash
# Example: download a public pedestrian video
cd /data/AI/AI_Vision/sample_videos
wget -O demo1.mp4 "https://your-video-source/pedestrians.mp4"

# Copy the same file as multiple demos for testing
for i in $(seq 2 5); do cp demo1.mp4 demo$i.mp4; done
```

---

## Step 5: Configure Stream Sources

Edit `/data/AI/AI_Vision/streams.txt` to list your actual sources:

```bash
nano /data/AI/AI_Vision/streams.txt
```

- Uncomment RTSP lines and replace with real camera URLs
- Keep `file://` lines for demo sources
- Total lines (non-comment) = number of streams to process (max 50)

---

## Step 6: Pre-pull Large Images

The DeepStream base image is ~15 GB. Pull it before `docker compose up` to avoid timeout issues:

```bash
docker pull nvcr.io/nvidia/deepstream:7.1-triton-multiarch
docker pull nvcr.io/nvidia/tritonserver:24.12-py3
docker pull confluentinc/cp-kafka:7.6.0
docker pull confluentinc/cp-zookeeper:7.6.0
docker pull bluenviron/mediamtx:latest
```

---

## Step 7: Build and Launch

```bash
cd /data/AI/AI_Vision
docker compose build
docker compose up -d
```

---

## Step 8: Verify Deployment

### 8a. Check all containers are running
```bash
docker compose ps
```

Expected: all 6 containers (triton, deepstream, kafka, zookeeper, mediamtx, frontend) showing `Up` or `healthy`.

### 8b. Verify Triton is serving the model
```bash
curl http://localhost:8000/v2/health/ready
# Expected: 200 OK

curl http://localhost:8000/v2/models/yolov8 | jq .
# Expected: JSON with "state": "READY"
```

Note: Triton ports (8000/8001/8002) are internal to the Docker network only — use `docker exec` if curl is not available on the host:
```bash
docker exec triton curl -s http://localhost:8000/v2/models/yolov8
```

### 8c. Verify DeepStream pipeline
```bash
docker compose logs -f deepstream
# Look for: "[INFO] Pipeline PLAYING — processing streams"
```

### 8d. Verify Kafka is receiving detections
```bash
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic detections \
  --from-beginning \
  --max-messages 5
```

### 8e. Verify MediaMTX streams
```bash
# Check API for active streams
docker exec mediamtx wget -qO- http://localhost:9997/v3/paths/list | head -50
```

### 8f. Open the Web Dashboard
Open a browser and navigate to:
```
http://<server-ip>:5258
```

You should see the stream grid with live annotated video feeds.

---

## Troubleshooting

### TensorRT version mismatch
**Symptom:** Triton logs show `model.plan was not built with this version of TensorRT`.
**Fix:** Rebuild the engine inside the exact Triton container (Step 3b). Never build with a different `trtexec` binary.

### NVENC session limit
**Symptom:** `nvv4l2h264enc` fails with "resource busy" for streams beyond ~3.
**Fix:** The L40S with data-center driver should have unlimited NVENC. Verify:
```bash
docker exec deepstream nvidia-smi -q | grep -i encoder
```
If limited, check that the server is using the data-center driver (not consumer GeForce driver).

### DeepStream fails to connect to Triton
**Symptom:** `nvinferserver` error: "gRPC connection refused".
**Fix:** Ensure `triton` container is `healthy` before DeepStream starts. The `depends_on` condition handles this. Check:
```bash
docker compose logs triton | grep -i ready
```

### Kafka data directory permissions
**Symptom:** Kafka crashes with permission denied on `/var/lib/kafka/data`.
**Fix:**
```bash
sudo chown -R 1000:1000 /data/AI/AI_Vision/kafka_data
```

### MediaMTX not receiving streams
**Symptom:** No streams appear in MediaMTX API.
**Fix:** Check that `rtspclientsink` in DeepStream can reach `mediamtx:8554`:
```bash
docker exec deepstream curl -s rtsp://mediamtx:8554
```

### Out of GPU memory
**Symptom:** CUDA OOM errors in DeepStream or Triton logs.
**Fix:** Reduce the number of streams in `streams.txt`, lower `MUXER_WIDTH`/`MUXER_HEIGHT`, or reduce `ENCODE_BITRATE`.

---

## Monitoring

### GPU utilization
```bash
watch -n 1 nvidia-smi
```

### Container resource usage
```bash
docker stats --no-stream
```

### Log rotation verification
```bash
# Check log sizes aren't exceeding limits
du -sh /var/lib/docker/containers/*/\*-json.log
```

---

## Stopping the System

```bash
cd /data/AI/AI_Vision
docker compose down
```

To stop and remove all data volumes:
```bash
docker compose down -v
rm -rf /data/AI/AI_Vision/kafka_data/* /data/AI/AI_Vision/zookeeper_data/*
```
