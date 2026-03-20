#!/usr/bin/env python3
"""
50-Stream Person Detection Pipeline (L40S Optimized)
=====================================================
DeepStream 7.x + Triton Inference Server + Kafka + MediaMTX

Pipeline:
  nvurisrcbin (x50) → nvstreammux → nvinferserver (gRPC→Triton)
    → [Python YOLOv8 tensor parser probe]
      → nvtracker (NvDCF) → nvdsosd → tee
        ├─ Branch A: nvmsgconv → nvmsgbroker → Kafka
        └─ Branch B: nvstreamdemux → 50x (encode → rtspclientsink → MediaMTX)

Parser Strategy:
  nvinferserver is configured with `postprocess { other {} }` which passes
  the raw output tensor through without built-in parsing. A Python pad probe
  between nvinferserver and nvtracker decodes the YOLOv8 output0 tensor
  (84x8400), applies NMS, and injects NvDsObjectMeta for person detections.
  This eliminates the need for a compiled C++ custom parser .so library.
"""

import sys
import os
import json
import datetime
import ctypes
import numpy as np
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import pyds

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STREAMS_FILE = os.environ.get("STREAMS_FILE", "/app/streams.txt")
NVINFERSERVER_CONFIG = os.environ.get("NVINFERSERVER_CONFIG", "/app/nvinferserver_config.pbtxt")
TRACKER_CONFIG = os.environ.get("TRACKER_CONFIG", "/app/tracker_config.yml")
TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"

KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "kafka:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "detections")
KAFKA_PROTO_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so"

MEDIAMTX_URL = os.environ.get("MEDIAMTX_URL", "rtsp://mediamtx:8554")

MUXER_WIDTH = int(os.environ.get("MUXER_WIDTH", "1920"))
MUXER_HEIGHT = int(os.environ.get("MUXER_HEIGHT", "1080"))
MUXER_BATCH_TIMEOUT = 40000  # microseconds (~25 fps target)

ENCODE_BITRATE = int(os.environ.get("ENCODE_BITRATE", "2000000"))

# YOLOv8 parsing constants
PERSON_CLASS_ID = 0       # COCO class 0 = person
CONF_THRESHOLD = 0.4
NMS_IOU_THRESHOLD = 0.45
NMS_TOPK = 300
NUM_CLASSES = 80
# YOLOv8 output0 shape: [84, 8400] where 84 = 4 (cx,cy,w,h) + 80 (class scores)
YOLO_NUM_BOXES = 8400
YOLO_BOX_DIM = 4


# ---------------------------------------------------------------------------
# YOLOv8 Output Tensor Parser (Python — replaces C++ custom_parse_bbox_func)
# ---------------------------------------------------------------------------
def yolov8_parse_tensor(output_tensor, net_w=640, net_h=640, img_w=1920, img_h=1080):
    """
    Parse YOLOv8 output0 tensor [84, 8400] into person detections.

    Returns list of (left, top, width, height, confidence) in image coordinates.
    """
    # output_tensor shape: (84, 8400)
    # Rows 0-3: cx, cy, w, h (normalized to network input 640x640)
    # Rows 4-83: class confidence scores for 80 COCO classes
    cx = output_tensor[0]   # (8400,)
    cy = output_tensor[1]
    w = output_tensor[2]
    h = output_tensor[3]
    class_scores = output_tensor[4:]  # (80, 8400)

    # Extract person class scores (class 0)
    person_scores = class_scores[PERSON_CLASS_ID]  # (8400,)

    # Filter by confidence threshold
    mask = person_scores > CONF_THRESHOLD
    if not np.any(mask):
        return []

    filtered_cx = cx[mask]
    filtered_cy = cy[mask]
    filtered_w = w[mask]
    filtered_h = h[mask]
    filtered_conf = person_scores[mask]

    # Convert from center-x,y,w,h to left,top,right,bottom (in network coords)
    x1 = filtered_cx - filtered_w / 2.0
    y1 = filtered_cy - filtered_h / 2.0
    x2 = filtered_cx + filtered_w / 2.0
    y2 = filtered_cy + filtered_h / 2.0

    # Apply NMS
    indices = _nms(x1, y1, x2, y2, filtered_conf, NMS_IOU_THRESHOLD, NMS_TOPK)

    # Scale from network coordinates (640x640) to image coordinates
    scale_x = img_w / net_w
    scale_y = img_h / net_h

    detections = []
    for i in indices:
        left = float(x1[i] * scale_x)
        top = float(y1[i] * scale_y)
        width = float((x2[i] - x1[i]) * scale_x)
        height = float((y2[i] - y1[i]) * scale_y)
        conf = float(filtered_conf[i])

        # Clamp to image bounds
        left = max(0.0, left)
        top = max(0.0, top)
        width = min(width, img_w - left)
        height = min(height, img_h - top)

        if width > 0 and height > 0:
            detections.append((left, top, width, height, conf))

    return detections


def _nms(x1, y1, x2, y2, scores, iou_threshold, topk):
    """Greedy NMS on numpy arrays. Returns list of kept indices."""
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    if topk > 0:
        order = order[:topk]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep


# ---------------------------------------------------------------------------
# Probe: Parse raw tensor output from nvinferserver → inject NvDsObjectMeta
# ---------------------------------------------------------------------------
def nvinferserver_src_pad_probe(pad, info, user_data):
    """
    Probe on the nvinferserver src pad. For each frame in the batch:
    - Extract the raw output tensor from NVDS_TENSOR_OUTPUT_META
    - Parse YOLOv8 detections using Python
    - Inject NvDsObjectMeta for each person detection
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Find the tensor output metadata attached by nvinferserver
        l_user = frame_meta.frame_user_meta_list
        tensor_meta = None
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_TENSOR_OUTPUT_META:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                break

            try:
                l_user = l_user.next
            except StopIteration:
                break

        if tensor_meta is not None:
            # Get output layer 0 (output0: shape [84, 8400])
            layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)

            if layer.dataType == 0:  # FP32
                ptr = ctypes.cast(
                    pyds.get_ptr(layer.buffer),
                    ctypes.POINTER(ctypes.c_float)
                )
                tensor = np.ctypeslib.as_array(ptr, shape=(84, YOLO_NUM_BOXES)).copy()
            elif layer.dataType == 1:  # FP16 — convert to FP32
                ptr = ctypes.cast(
                    pyds.get_ptr(layer.buffer),
                    ctypes.POINTER(ctypes.c_uint16)
                )
                raw = np.ctypeslib.as_array(ptr, shape=(84, YOLO_NUM_BOXES)).copy()
                tensor = raw.view(np.float16).astype(np.float32)
            else:
                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break
                continue

            # Parse detections
            detections = yolov8_parse_tensor(
                tensor,
                net_w=640, net_h=640,
                img_w=frame_meta.source_frame_width or MUXER_WIDTH,
                img_h=frame_meta.source_frame_height or MUXER_HEIGHT,
            )

            # Inject NvDsObjectMeta for each person detection
            for left, top, width, height, confidence in detections:
                obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
                obj_meta.unique_component_id = 1
                obj_meta.class_id = PERSON_CLASS_ID
                obj_meta.confidence = confidence
                obj_meta.object_id = pyds.UNTRACKED_OBJECT_ID

                rect = obj_meta.rect_params
                rect.left = left
                rect.top = top
                rect.width = width
                rect.height = height

                # Bounding box display styling
                rect.border_width = 2
                rect.border_color.set(0.0, 1.0, 0.0, 1.0)  # green
                rect.has_bg_color = False

                # Text label
                txt = obj_meta.text_params
                txt.display_text = f"person {confidence:.2f}"
                txt.x_offset = int(left)
                txt.y_offset = max(0, int(top) - 12)
                txt.font_params.font_name = "Serif"
                txt.font_params.font_size = 12
                txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                txt.set_bg_clr = True
                txt.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)

                pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


# ---------------------------------------------------------------------------
# Stream Utilities
# ---------------------------------------------------------------------------
def read_streams(path):
    """Read stream URIs from file, skip comments and blanks."""
    streams = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                streams.append(line)
    if not streams:
        sys.stderr.write(f"ERROR: No streams found in {path}\n")
        sys.exit(1)
    print(f"[INFO] Loaded {len(streams)} stream(s) from {path}")
    return streams


def create_source_bin(index, uri):
    """
    Create a GstBin wrapping nvurisrcbin for a single source.
    Supports both rtsp:// and file:// URIs.
    """
    bin_name = f"source-bin-{index:02d}"
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        raise RuntimeError(f"Failed to create source bin '{bin_name}'")

    uri_src = Gst.ElementFactory.make("nvurisrcbin", f"uri-src-{index:02d}")
    if not uri_src:
        raise RuntimeError("Failed to create nvurisrcbin")

    uri_src.set_property("uri", uri)
    uri_src.set_property("gpu-id", 0)
    uri_src.set_property("cudadec-memtype", 0)  # NVBUF_MEM_DEFAULT

    # Enable infinite looping for file sources
    if uri.startswith("file://"):
        uri_src.set_property("file-loop", True)

    # RTSP-specific: enable reconnection
    if uri.startswith("rtsp://"):
        uri_src.set_property("rtsp-reconnect-interval", 5)

    nbin.add(uri_src)

    # Create ghost pad — nvurisrcbin has a dynamic src pad
    ghost_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
    nbin.add_pad(ghost_pad)

    def on_pad_added(src, new_pad, ghost):
        sinkpad_name = new_pad.get_name()
        if "src" in sinkpad_name:
            ghost.set_target(new_pad)

    uri_src.connect("pad-added", on_pad_added, ghost_pad)

    return nbin


# ---------------------------------------------------------------------------
# Kafka Metadata Probe (on OSD sink pad)
# ---------------------------------------------------------------------------
def generate_event_msg(frame_number, stream_id, source_uri, obj_meta):
    """Build a JSON detection event dict from object metadata."""
    rect = obj_meta.rect_params
    return {
        "track_id": obj_meta.object_id,
        "class_id": obj_meta.class_id,
        "label": "person",
        "confidence": round(obj_meta.confidence, 4),
        "bbox": {
            "left": round(rect.left, 1),
            "top": round(rect.top, 1),
            "width": round(rect.width, 1),
            "height": round(rect.height, 1),
        },
    }


def osd_sink_pad_buffer_probe(pad, info, user_data):
    """
    Probe on the OSD sink pad. For every frame in the batch:
    - Iterate object metadata
    - Filter for person detections (class_id == 0)
    - Attach NVDS_EVENT_MSG_META for Kafka output
    """
    streams = user_data

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        stream_id = frame_meta.source_id
        frame_number = frame_meta.frame_num
        source_uri = streams[stream_id] if stream_id < len(streams) else "unknown"

        detections = []
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if obj_meta.class_id == PERSON_CLASS_ID:
                detections.append(
                    generate_event_msg(frame_number, stream_id, source_uri, obj_meta)
                )

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Attach event message metadata for nvmsgconv/nvmsgbroker
        if detections:
            msg_meta = pyds.alloc_nvds_event_msg_meta()
            msg_meta.bbox.top = 0
            msg_meta.bbox.left = 0
            msg_meta.bbox.width = 0
            msg_meta.bbox.height = 0
            msg_meta.frameId = frame_number
            msg_meta.trackingId = 0
            msg_meta.confidence = 0
            msg_meta.sensorId = stream_id
            msg_meta.ts = str(datetime.datetime.utcnow().isoformat() + "Z")
            msg_meta.objectId = "person"

            user_meta = pyds.nvds_acquire_user_meta_from_pool(batch_meta)
            if user_meta:
                user_meta.user_meta_data = msg_meta
                user_meta.base_meta.meta_type = pyds.NvDsMetaType.NVDS_EVENT_MSG_META
                pyds.nvds_add_user_meta_to_frame(frame_meta, user_meta)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


# ---------------------------------------------------------------------------
# Bus Message Handler
# ---------------------------------------------------------------------------
def bus_call(bus, message, loop):
    """Handle pipeline bus messages."""
    t = message.type
    if t == Gst.MessageType.EOS:
        print("[INFO] End-of-Stream reached")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"[WARN] {err}: {debug}")
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"[ERROR] {err}: {debug}")
        loop.quit()
    elif t == Gst.MessageType.STATE_CHANGED:
        if message.src.get_name() == "pipeline":
            old, new, pending = message.parse_state_changed()
            print(f"[INFO] Pipeline state: {old.value_nick} → {new.value_nick}")
    return True


def make_element(factory, name):
    """Create a GStreamer element or raise with a clear error."""
    elem = Gst.ElementFactory.make(factory, name)
    if not elem:
        raise RuntimeError(f"Failed to create element: {factory} (name={name})")
    return elem


# ---------------------------------------------------------------------------
# Main Pipeline Construction
# ---------------------------------------------------------------------------
def main():
    Gst.init(None)

    streams = read_streams(STREAMS_FILE)
    num_sources = len(streams)
    print(f"[INFO] Building pipeline for {num_sources} stream(s)")

    # -----------------------------------------------------------------------
    # Create pipeline and elements
    # -----------------------------------------------------------------------
    pipeline = Gst.Pipeline.new("pipeline")
    if not pipeline:
        raise RuntimeError("Failed to create pipeline")

    # --- Streammux (L40S optimized) ---
    streammux = make_element("nvstreammux", "streammux")
    streammux.set_property("batch-size", num_sources)
    streammux.set_property("width", MUXER_WIDTH)
    streammux.set_property("height", MUXER_HEIGHT)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT)
    streammux.set_property("gpu-id", 0)
    streammux.set_property("live-source", 1)
    streammux.set_property("nvbuf-memory-type", 0)
    streammux.set_property("enable-padding", False)  # L40S: skip padding to save VRAM
    pipeline.add(streammux)

    # --- Create and link source bins ---
    for i, uri in enumerate(streams):
        print(f"[INFO] Adding source {i}: {uri}")
        source_bin = create_source_bin(i, uri)
        pipeline.add(source_bin)

        sinkpad = streammux.request_pad_simple(f"sink_{i}")
        srcpad = source_bin.get_static_pad("src")
        if not sinkpad or not srcpad:
            raise RuntimeError(f"Failed to get pads for source {i}")
        srcpad.link(sinkpad)

    # --- Inference (Triton via gRPC) ---
    # postprocess is set to { other {} } — raw tensor output, parsed by Python probe
    nvinferserver = make_element("nvinferserver", "nvinferserver")
    nvinferserver.set_property("config-file-path", NVINFERSERVER_CONFIG)
    nvinferserver.set_property("input-tensor-meta", False)
    pipeline.add(nvinferserver)

    # --- Tracker (NvDCF) ---
    tracker = make_element("nvtracker", "tracker")
    tracker.set_property("tracker-width", 640)
    tracker.set_property("tracker-height", 384)
    tracker.set_property("gpu-id", 0)
    tracker.set_property("ll-lib-file", TRACKER_LIB)
    tracker.set_property("ll-config-file", TRACKER_CONFIG)
    tracker.set_property("enable-batch-process", True)
    pipeline.add(tracker)

    # --- On-Screen Display ---
    osd = make_element("nvdsosd", "osd")
    osd.set_property("process-mode", 1)  # GPU mode
    osd.set_property("display-text", True)
    pipeline.add(osd)

    # --- Tee (split into Kafka branch + Video branch) ---
    tee = make_element("tee", "tee")
    pipeline.add(tee)

    # ===================================================================
    # Branch A: Kafka Metadata Output
    # ===================================================================
    queue_kafka = make_element("queue", "queue-kafka")
    queue_kafka.set_property("max-size-buffers", 200)
    queue_kafka.set_property("leaky", 2)  # downstream — drop oldest if full
    pipeline.add(queue_kafka)

    msgconv = make_element("nvmsgconv", "msgconv")
    msgconv.set_property("config", "")
    msgconv.set_property("payload-type", 1)  # minimal payload
    msgconv.set_property("comp-id", 1)
    pipeline.add(msgconv)

    msgbroker = make_element("nvmsgbroker", "msgbroker")
    msgbroker.set_property("proto-lib", KAFKA_PROTO_LIB)
    msgbroker.set_property("conn-str", f"{KAFKA_BROKER};{KAFKA_TOPIC}")
    msgbroker.set_property("topic", KAFKA_TOPIC)
    msgbroker.set_property("sync", False)  # Non-blocking: pipeline never stalls on Kafka
    pipeline.add(msgbroker)

    # ===================================================================
    # Branch B: Per-stream Video Output → MediaMTX
    # ===================================================================
    queue_video = make_element("queue", "queue-video")
    queue_video.set_property("max-size-buffers", 200)
    queue_video.set_property("leaky", 2)  # downstream — drop oldest if full
    pipeline.add(queue_video)

    demux = make_element("nvstreamdemux", "demux")
    pipeline.add(demux)

    # Per-stream encode pipelines
    for i in range(num_sources):
        # Video convert (GPU)
        vidconv = make_element("nvvideoconvert", f"vidconv-{i}")
        vidconv.set_property("gpu-id", 0)
        pipeline.add(vidconv)

        # Caps filter to set output format for encoder
        capsfilter = make_element("capsfilter", f"caps-{i}")
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
        capsfilter.set_property("caps", caps)
        pipeline.add(capsfilter)

        # Hardware H.264 encoder (L40S optimized)
        encoder = make_element("nvv4l2h264enc", f"encoder-{i}")
        encoder.set_property("gpu-id", 0)          # Explicit GPU pin for NVENC
        encoder.set_property("bitrate", ENCODE_BITRATE)
        encoder.set_property("control-rate", 1)     # CBR — predictable bitrate across 50 streams
        encoder.set_property("preset-level", 1)     # ultrafast
        encoder.set_property("profile", 4)          # high
        encoder.set_property("iframeinterval", 30)
        encoder.set_property("maxperf-enable", True)
        pipeline.add(encoder)

        # H.264 parse
        h264parse = make_element("h264parse", f"h264parse-{i}")
        pipeline.add(h264parse)

        # RTSP client sink → push to MediaMTX
        rtspsink = make_element("rtspclientsink", f"rtspsink-{i}")
        rtspsink.set_property("location", f"{MEDIAMTX_URL}/stream{i}")
        rtspsink.set_property("protocols", "tcp")
        rtspsink.set_property("latency", 100)
        pipeline.add(rtspsink)

        # Link demux src pad → vidconv → caps → encoder → parse → rtspsink
        demux_srcpad = demux.request_pad_simple(f"src_{i}")
        vidconv_sinkpad = vidconv.get_static_pad("sink")
        demux_srcpad.link(vidconv_sinkpad)

        vidconv.link(capsfilter)
        capsfilter.link(encoder)
        encoder.link(h264parse)
        h264parse.link(rtspsink)

    # -----------------------------------------------------------------------
    # Link the main pipeline chain
    # -----------------------------------------------------------------------
    # streammux → nvinferserver → tracker → osd → tee
    streammux.link(nvinferserver)
    nvinferserver.link(tracker)
    tracker.link(osd)
    osd.link(tee)

    # tee → Branch A (Kafka)
    tee_kafka_pad = tee.request_pad_simple("src_%u")
    queue_kafka_sinkpad = queue_kafka.get_static_pad("sink")
    tee_kafka_pad.link(queue_kafka_sinkpad)
    queue_kafka.link(msgconv)
    msgconv.link(msgbroker)

    # tee → Branch B (Video → demux)
    tee_video_pad = tee.request_pad_simple("src_%u")
    queue_video_sinkpad = queue_video.get_static_pad("sink")
    tee_video_pad.link(queue_video_sinkpad)
    queue_video.link(demux)

    # -----------------------------------------------------------------------
    # Attach probes
    # -----------------------------------------------------------------------
    # Probe 1: Parse raw YOLOv8 tensor → inject NvDsObjectMeta (between infer and tracker)
    nvinfer_srcpad = nvinferserver.get_static_pad("src")
    nvinfer_srcpad.add_probe(
        Gst.PadProbeType.BUFFER, nvinferserver_src_pad_probe, None
    )

    # Probe 2: Build Kafka event metadata from parsed detections (on OSD sink)
    osd_sinkpad = osd.get_static_pad("sink")
    osd_sinkpad.add_probe(
        Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, streams
    )

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    print("[INFO] Setting pipeline to PLAYING...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[ERROR] Failed to set pipeline to PLAYING")
        sys.exit(1)

    print("[INFO] Pipeline PLAYING — processing streams")
    try:
        loop.run()
    except KeyboardInterrupt:
        print("[INFO] Interrupted — shutting down")
    finally:
        pipeline.set_state(Gst.State.NULL)
        print("[INFO] Pipeline stopped")


if __name__ == "__main__":
    main()
