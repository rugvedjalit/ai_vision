import React, { useState, useEffect, useRef, useCallback } from "react";

const TOTAL_STREAMS = 50;
const MEDIAMTX_HOST = window.location.hostname;
const WHEP_PORT = 5257;

/**
 * WHEP (WebRTC-HTTP Egress Protocol) client.
 * Negotiates a WebRTC session with MediaMTX for a given stream path.
 */
async function startWhepSession(streamPath, videoElement) {
  const pc = new RTCPeerConnection({
    iceServers: [],
    bundlePolicy: "max-bundle",
  });

  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addTransceiver("audio", { direction: "recvonly" });

  pc.ontrack = (event) => {
    if (event.streams && event.streams[0]) {
      videoElement.srcObject = event.streams[0];
    }
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  // Wait for ICE gathering to complete (or timeout after 2s)
  await new Promise((resolve) => {
    if (pc.iceGatheringState === "complete") {
      resolve();
      return;
    }
    const timeout = setTimeout(resolve, 2000);
    pc.onicegatheringstatechange = () => {
      if (pc.iceGatheringState === "complete") {
        clearTimeout(timeout);
        resolve();
      }
    };
  });

  const whepUrl = `http://${MEDIAMTX_HOST}:${WHEP_PORT}/${streamPath}/whep`;

  const response = await fetch(whepUrl, {
    method: "POST",
    headers: { "Content-Type": "application/sdp" },
    body: pc.localDescription.sdp,
  });

  if (!response.ok) {
    throw new Error(`WHEP ${response.status}: ${response.statusText}`);
  }

  const answerSdp = await response.text();
  await pc.setRemoteDescription(
    new RTCSessionDescription({ type: "answer", sdp: answerSdp })
  );

  return pc;
}

/** Single stream video tile */
function StreamTile({ streamIndex, isSelected, onClick }) {
  const videoRef = useRef(null);
  const pcRef = useRef(null);
  const [status, setStatus] = useState("connecting");

  useEffect(() => {
    let cancelled = false;
    const connect = async () => {
      try {
        setStatus("connecting");
        if (pcRef.current) {
          pcRef.current.close();
          pcRef.current = null;
        }
        const pc = await startWhepSession(
          `stream${streamIndex}`,
          videoRef.current
        );
        if (cancelled) {
          pc.close();
          return;
        }
        pcRef.current = pc;
        setStatus("live");

        pc.onconnectionstatechange = () => {
          if (pc.connectionState === "failed" || pc.connectionState === "disconnected") {
            setStatus("error");
          }
        };
      } catch (err) {
        if (!cancelled) {
          console.error(`Stream ${streamIndex}:`, err);
          setStatus("error");
        }
      }
    };

    connect();

    return () => {
      cancelled = true;
      if (pcRef.current) {
        pcRef.current.close();
        pcRef.current = null;
      }
    };
  }, [streamIndex]);

  const statusColor =
    status === "live" ? "#00ff88" : status === "connecting" ? "#ffaa00" : "#ff4444";

  return (
    <div
      onClick={onClick}
      style={{
        position: "relative",
        background: "#111118",
        borderRadius: 6,
        overflow: "hidden",
        cursor: "pointer",
        border: isSelected ? "2px solid #00aaff" : "2px solid #222",
        transition: "border-color 0.2s",
      }}
    >
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
      />
      <div
        style={{
          position: "absolute",
          top: 6,
          left: 8,
          display: "flex",
          alignItems: "center",
          gap: 6,
          background: "rgba(0,0,0,0.7)",
          padding: "2px 8px",
          borderRadius: 4,
          fontSize: 12,
        }}
      >
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: statusColor,
            display: "inline-block",
          }}
        />
        <span>Stream {streamIndex}</span>
      </div>
    </div>
  );
}

/** Main application */
export default function App() {
  const [gridSize, setGridSize] = useState(9); // 3x3 default
  const [startIndex, setStartIndex] = useState(0);
  const [selectedStream, setSelectedStream] = useState(null);

  const gridOptions = [
    { label: "2x2", value: 4, cols: 2 },
    { label: "3x3", value: 9, cols: 3 },
    { label: "4x4", value: 16, cols: 4 },
    { label: "5x5", value: 25, cols: 5 },
  ];

  const currentGrid = gridOptions.find((g) => g.value === gridSize) || gridOptions[1];
  const visibleStreams = Array.from(
    { length: Math.min(currentGrid.value, TOTAL_STREAMS - startIndex) },
    (_, i) => startIndex + i
  );

  const maxStart = Math.max(0, TOTAL_STREAMS - currentGrid.value);

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Header */}
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "12px 24px",
          background: "#12121a",
          borderBottom: "1px solid #222",
        }}
      >
        <h1 style={{ fontSize: 18, fontWeight: 600, color: "#fff" }}>
          AI Vision — Person Detection
        </h1>

        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          {/* Grid size selector */}
          <div style={{ display: "flex", gap: 4 }}>
            {gridOptions.map((opt) => (
              <button
                key={opt.value}
                onClick={() => {
                  setGridSize(opt.value);
                  setStartIndex(0);
                }}
                style={{
                  padding: "4px 10px",
                  borderRadius: 4,
                  border: "none",
                  cursor: "pointer",
                  fontSize: 13,
                  background: gridSize === opt.value ? "#00aaff" : "#2a2a35",
                  color: gridSize === opt.value ? "#000" : "#ccc",
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>

          {/* Page navigation */}
          <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <button
              disabled={startIndex === 0}
              onClick={() => setStartIndex(Math.max(0, startIndex - currentGrid.value))}
              style={{
                padding: "4px 10px",
                borderRadius: 4,
                border: "none",
                cursor: startIndex === 0 ? "not-allowed" : "pointer",
                background: "#2a2a35",
                color: startIndex === 0 ? "#555" : "#ccc",
                fontSize: 13,
              }}
            >
              Prev
            </button>
            <span style={{ fontSize: 13, color: "#888", minWidth: 80, textAlign: "center" }}>
              {startIndex}–{Math.min(startIndex + currentGrid.value - 1, TOTAL_STREAMS - 1)} / {TOTAL_STREAMS}
            </span>
            <button
              disabled={startIndex >= maxStart}
              onClick={() =>
                setStartIndex(Math.min(maxStart, startIndex + currentGrid.value))
              }
              style={{
                padding: "4px 10px",
                borderRadius: 4,
                border: "none",
                cursor: startIndex >= maxStart ? "not-allowed" : "pointer",
                background: "#2a2a35",
                color: startIndex >= maxStart ? "#555" : "#ccc",
                fontSize: 13,
              }}
            >
              Next
            </button>
          </div>
        </div>
      </header>

      {/* Selected stream full view */}
      {selectedStream !== null && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 100,
            background: "rgba(0,0,0,0.92)",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            padding: 24,
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              width: "100%",
              maxWidth: 1280,
              marginBottom: 12,
            }}
          >
            <span style={{ fontSize: 16, fontWeight: 600 }}>
              Stream {selectedStream} — Full View
            </span>
            <button
              onClick={() => setSelectedStream(null)}
              style={{
                padding: "4px 16px",
                borderRadius: 4,
                border: "none",
                cursor: "pointer",
                background: "#ff4444",
                color: "#fff",
                fontSize: 13,
              }}
            >
              Close
            </button>
          </div>
          <div style={{ width: "100%", maxWidth: 1280, aspectRatio: "16/9" }}>
            <StreamTile
              streamIndex={selectedStream}
              isSelected={false}
              onClick={() => {}}
            />
          </div>
        </div>
      )}

      {/* Stream grid */}
      <main
        style={{
          flex: 1,
          padding: 16,
          display: "grid",
          gridTemplateColumns: `repeat(${currentGrid.cols}, 1fr)`,
          gap: 8,
          gridAutoRows: "1fr",
        }}
      >
        {visibleStreams.map((idx) => (
          <StreamTile
            key={idx}
            streamIndex={idx}
            isSelected={selectedStream === idx}
            onClick={() => setSelectedStream(idx)}
          />
        ))}
      </main>

      {/* Footer */}
      <footer
        style={{
          padding: "8px 24px",
          background: "#12121a",
          borderTop: "1px solid #222",
          fontSize: 12,
          color: "#555",
          textAlign: "center",
        }}
      >
        AI Vision — 50-Stream Person Detection System | DeepStream + Triton + YOLOv8
      </footer>
    </div>
  );
}
