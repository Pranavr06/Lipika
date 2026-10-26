"""
Interactive Web Application for Lipika Kannada Character Recognizer.
Provides an HTML5 drawing canvas and file upload UI with real-time sub-millisecond inference.
Runs using Python's built-in http.server (no external server framework required).
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import json
import base64
import io
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))
from inference.predictor import KannadaPredictor

# Initialize ONNX inference engine
predictor = KannadaPredictor(backend="onnx")

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lipika - Offline Kannada Handwritten Character Recognizer</title>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Kannada:wght@400;700&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-card: #1e293b;
            --accent: #38bdf8;
            --accent-hover: #0ea5e9;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
            --success: #22c55e;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 24px;
        }
        header {
            text-align: center;
            margin-bottom: 28px;
        }
        header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        header p { color: var(--text-muted); font-size: 1rem; }
        .badge {
            display: inline-block;
            background: #0369a1;
            color: #e0f2fe;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-top: 6px;
        }
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            max-width: 900px;
            width: 100%;
        }
        @media(max-width: 768px) {
            .container { grid-template-columns: 1fr; }
        }
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
        }
        .canvas-wrapper {
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;
        }
        canvas {
            background: #000000;
            border: 2px solid var(--accent);
            border-radius: 12px;
            cursor: crosshair;
            touch-action: none;
            width: 280px;
            height: 280px;
        }
        .controls {
            display: flex;
            gap: 12px;
            margin-top: 16px;
            width: 280px;
        }
        button {
            flex: 1;
            padding: 10px 16px;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .btn-clear {
            background: #475569;
            color: white;
        }
        .btn-clear:hover { background: #64748b; }
        .btn-recognize {
            background: var(--accent);
            color: #0f172a;
        }
        .btn-recognize:hover { background: var(--accent-hover); }
        .result-box {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            text-align: center;
        }
        .big-glyph {
            font-family: 'Noto Sans Kannada', sans-serif;
            font-size: 5.5rem;
            color: var(--accent);
            line-height: 1.1;
            margin: 12px 0;
            min-height: 90px;
        }
        .phonetic {
            font-size: 1.3rem;
            font-weight: 600;
            color: var(--text-main);
        }
        .group-tag {
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 4px;
        }
        .confidence-meter {
            width: 100%;
            margin-top: 20px;
        }
        .candidate-row {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            gap: 10px;
            font-size: 0.95rem;
        }
        .candidate-char {
            font-family: 'Noto Sans Kannada', sans-serif;
            font-size: 1.2rem;
            width: 32px;
            color: var(--accent);
        }
        .bar-container {
            flex: 1;
            background: #334155;
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
        }
        .bar-fill {
            background: var(--accent);
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        .stat-latency {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 14px;
        }
    </style>
</head>
<body>
    <header>
        <h1>Lipika (ಲಿಪಿಕಾ)</h1>
        <p>Offline Kannada Handwritten Character Recognizer using Fast-KannadaNet</p>
        <span class="badge">ONNX CPU Runtime &lt; 1 ms</span>
    </header>

    <div class="container">
        <!-- Canvas Card -->
        <div class="card canvas-wrapper">
            <h3 style="margin-bottom: 12px;">Draw Kannada Character</h3>
            <canvas id="charCanvas" width="280" height="280"></canvas>
            <div class="controls">
                <button class="btn-clear" onclick="clearCanvas()">Clear</button>
                <button class="btn-recognize" onclick="recognizeCanvas()">Recognize</button>
            </div>
            <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 12px;">
                Tip: Draw in the center with bold strokes.
            </p>
        </div>

        <!-- Result Card -->
        <div class="card result-box">
            <h3 style="color: var(--text-muted);">Top Recognition</h3>
            <div class="big-glyph" id="charDisplay">-</div>
            <div class="phonetic" id="phoneticDisplay">Draw character on canvas</div>
            <div class="group-tag" id="groupDisplay">Category: -</div>

            <div class="confidence-meter" id="candidatesList"></div>
            <div class="stat-latency" id="latencyDisplay">Inference Latency: 0.0 ms</div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('charCanvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        let hasDrawn = false;

        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 18;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        function startPosition(e) {
            isDrawing = true;
            hasDrawn = true;
            draw(e);
        }

        function endPosition() {
            if (isDrawing) {
                isDrawing = false;
                ctx.beginPath();
                recognizeCanvas(); // Auto-recognize on stroke finish
            }
        }

        function getPos(e) {
            const rect = canvas.getBoundingClientRect();
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            return {
                x: clientX - rect.left,
                y: clientY - rect.top
            };
        }

        function draw(e) {
            if (!isDrawing) return;
            e.preventDefault();
            const pos = getPos(e);
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(pos.x, pos.y);
        }

        canvas.addEventListener('mousedown', startPosition);
        canvas.addEventListener('mouseup', endPosition);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseleave', endPosition);

        canvas.addEventListener('touchstart', startPosition, { passive: false });
        canvas.addEventListener('touchend', endPosition);
        canvas.addEventListener('touchmove', draw, { passive: false });

        function clearCanvas() {
            ctx.fillStyle = '#000000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.beginPath();
            hasDrawn = false;
            document.getElementById('charDisplay').innerText = '-';
            document.getElementById('phoneticDisplay').innerText = 'Draw character on canvas';
            document.getElementById('groupDisplay').innerText = 'Category: -';
            document.getElementById('candidatesList').innerHTML = '';
            document.getElementById('latencyDisplay').innerText = 'Inference Latency: 0.0 ms';
        }

        async function recognizeCanvas() {
            if (!hasDrawn) return;
            const dataUrl = canvas.toDataURL('image/png');
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: dataUrl })
                });
                const data = await response.json();
                renderResults(data);
            } catch (err) {
                console.error('Inference error:', err);
            }
        }

        function renderResults(res) {
            if (!res || !res.top_predictions || res.top_predictions.length === 0) return;
            const best = res.top_predictions[0];
            document.getElementById('charDisplay').innerText = best.char;
            document.getElementById('phoneticDisplay').innerText = best.phonetic.toUpperCase() + ' (' + best.unicode + ')';
            document.getElementById('groupDisplay').innerText = 'Category: ' + best.group_name;
            document.getElementById('latencyDisplay').innerText = 'Inference Latency: ' + res.latency_ms.toFixed(2) + ' ms';

            let listHtml = '';
            res.top_predictions.forEach(cand => {
                const pct = (cand.confidence * 100).toFixed(1);
                listHtml += `
                    <div class="candidate-row">
                        <span class="candidate-char">${cand.char}</span>
                        <span style="width: 50px; font-size: 0.85rem; color: #94a3b8;">${cand.phonetic}</span>
                        <div class="bar-container">
                            <div class="bar-fill" style="width: ${pct}%;"></div>
                        </div>
                        <span style="font-size: 0.85rem; width: 45px; text-align: right;">${pct}%</span>
                    </div>
                `;
            });
            document.getElementById('candidatesList').innerHTML = listHtml;
        }
    </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
            
            # Extract base64 image
            image_data = data.get("image", "")
            if "," in image_data:
                image_data = image_data.split(",")[1]
                
            image_bytes = base64.b64decode(image_data)
            pil_img = Image.open(io.BytesIO(image_bytes))
            
            # Predict
            result = predictor.predict(pil_img, top_k=3)
            # Remove numpy array before serializing
            result.pop("processed_image", None)
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port: int = 8000):
    server = HTTPServer(("0.0.0.0", port), RequestHandler)
    print(f"Lipika Web Server running at http://localhost:{port}")
    server.serve_forever()

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
