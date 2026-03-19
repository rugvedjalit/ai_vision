FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

# Install Python dependencies for Kafka metadata handling
RUN pip3 install --no-cache-dir kafka-python-ng

# Application working directory
WORKDIR /app

# Copy pipeline application and configs
COPY deepstream_app.py .
COPY nvinferserver_config.pbtxt .
COPY tracker_config.yml .
COPY streams.txt .

# Healthcheck: verify GStreamer is functional
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python3 -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; Gst.init(None); print('ok')" || exit 1

ENTRYPOINT ["python3", "/app/deepstream_app.py"]
