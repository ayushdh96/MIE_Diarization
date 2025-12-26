import os
import uuid
import subprocess
import json
import time
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
KEEP_UPLOAD_ARTIFACTS = os.getenv("KEEP_UPLOAD_ARTIFACTS", "false").lower() == "true"
ARTIFACT_EXTS = [".webm", ".txt", ".json", "_summary.txt"]

# Delete upload files older than this many hours (default: 24h)
UPLOAD_RETENTION_HOURS = int(os.getenv("UPLOAD_RETENTION_HOURS", "24"))

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "https://voice.opensource.mieweb.org"]}})
LATEST_POINTER = UPLOAD_DIR / "latest.txt"

def cleanup_old_uploads(max_age_seconds: int) -> None:
    """Best-effort cleanup of upload files based on age (mtime)."""
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        now = time.time()

        for p in UPLOAD_DIR.iterdir():
            if not p.is_file():
                continue
            # Don't delete the pointer file (even though we may not rely on it)
            if p.name == LATEST_POINTER.name:
                continue

            try:
                age = now - p.stat().st_mtime
                if age > max_age_seconds:
                    p.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception:
        pass

def get_latest_stem() -> str:
    try:
        if LATEST_POINTER.exists():
            return LATEST_POINTER.read_text().strip()
    except Exception:
        pass
    return ""

def set_latest_stem(stem: str) -> None:
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        LATEST_POINTER.write_text(stem)
    except Exception:
        pass

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"message": "Backend is running!"})

@app.route('/api/diarize', methods=['POST'])
def diarize_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Generate a unique filename and save
    filename = f"{uuid.uuid4().hex}.webm"
    save_path = UPLOAD_DIR / filename
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    audio_file.save(save_path)

    # Cleanup old files (older than retention window)
    cleanup_old_uploads(UPLOAD_RETENTION_HOURS * 3600)

    mode = request.form.get("mode", "full")
    print("Received processing mode:", mode)

    try:
        result = subprocess.run(
            ['python3', 'diarize.py', '-a', str(save_path), '--mode', mode],
            capture_output=True,
            text=True,
            check=True
        )

        stem = os.path.splitext(filename)[0]

        # Path to transcript file
        transcript_file = UPLOAD_DIR / (stem + ".txt")
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript_text = f.read().lstrip('\ufeff')

        # Load diarization JSON produced by diarize.py (if available)
        diarization_json = None
        diar_json_path = UPLOAD_DIR / (stem + ".json")
        if diar_json_path.exists():
            try:
                with open(diar_json_path, "r", encoding="utf-8") as jf:
                    diarization_json = json.load(jf)
                # Optional: print a small portion to backend console for debugging
                # print("[DEBUG] Diarization JSON (truncated):", json.dumps(diarization_json, indent=2)[:2000])
            except Exception as e:
                print("Error reading diarization JSON:", str(e))
        else:
            print(f"[WARN] Diarization JSON not found at {diar_json_path}")

        # Keep all recent artifacts for debugging; retention cleanup handles disk usage.
        if not KEEP_UPLOAD_ARTIFACTS:
            set_latest_stem(stem)

        return jsonify({
            "message": "Diarization done",
            "filename": filename,
            "transcript": transcript_text,
            "diarization_json": diarization_json
        }), 200
    except subprocess.CalledProcessError as e:
        return jsonify({
            "error": "Diarization failed",
            "details": e.stderr
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)