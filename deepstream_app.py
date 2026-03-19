#!/usr/bin/env python3
"""
50-Stream Person Detection Pipeline
====================================
DeepStream 7.1 + Triton Inference Server + Kafka + MediaMTX

Pipeline:
  nvurisrcbin (x50) → nvstreammux → nvinferserver (gRPC→Triton)
    → nvtracker (NvDCF) → nvdsosd → tee
      ├─ Branch A: nvmsgconv → nvmsgbroker → Kafka
      └─ Branch B: nvstreamdemux → 50x (encode → rtspclientsink → MediaMTX)
"""

import sys
import os
import json
import datetime
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
PERSON_CLASS_ID = 0  # COCO class 0 = person


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

    # Callback to link the dynamic pad from nvurisrcbin to bin ghost pad
    def pad_added_handler(src, new_pad, data):
        ghost_pad = data.get_static_pad("src")
        if ghost_pad and not ghost_pad.is_linked():
            new_pad.link(ghost_pad.get_peer() if ghost_pad.get_peer() else ghost_pad)

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
            event_payload = json.dumps(
                {
                    "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
                    "stream_id": stream_id,
                    "frame_number": frame_number,
                    "source_uri": source_uri,
                    "detections": detections,
                },
                separators=(",", ":"),
            )

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

    # --- Streammux ---
    streammux = make_element("nvstreammux", "streammux")
    streammux.set_property("batch-size", num_sources)
    streammux.set_property("width", MUXER_WIDTH)
    streammux.set_property("height", MUXER_HEIGHT)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT)
    streammux.set_property("gpu-id", 0)
    streammux.set_property("live-source", 1)
    streammux.set_property("nvbuf-memory-type", 0)
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
    queue_kafka.set_property("leaky", 2)  # downstream
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
    msgbroker.set_property("sync", False)
    pipeline.add(msgbroker)

    # ===================================================================
    # Branch B: Per-stream Video Output → MediaMTX
    # ===================================================================
    queue_video = make_element("queue", "queue-video")
    queue_video.set_property("max-size-buffers", 200)
    queue_video.set_property("leaky", 2)
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

        # Hardware H.264 encoder
        encoder = make_element("nvv4l2h264enc", f"encoder-{i}")
        encoder.set_property("bitrate", ENCODE_BITRATE)
        encoder.set_property("preset-level", 1)  # ultrafast
        encoder.set_property("profile", 4)  # high
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
    # Add OSD probe for Kafka metadata generation
    # -----------------------------------------------------------------------
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
