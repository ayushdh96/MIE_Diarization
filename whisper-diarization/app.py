import os
import uuid
import subprocess
import json
import requests
import time
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()
OZWELL_API_KEY = os.getenv("OZWELL_API_KEY")

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

    interaction_type = request.form.get("interaction_type", "medical")
    print("Received interactionType:", interaction_type)

    # New: processing mode ("full" = diarization + transcription, "asr" = transcription only)
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

        # Prepare Ozwell summarization request
        ozwell_url = "https://ai.bluehive.com/api/v1/completion"
        headers = {
                    "Authorization": f"Bearer {OZWELL_API_KEY}",
                    "Content-Type": "application/json"
                    }

        if interaction_type.lower() == "general":
            system_message = "You are a helpful, general-purpose assistant. Summarize this conversation clearly and concisely for any reader. Do not assume any medical context. Avoid disclaimers."
        else:
            system_message = """You are a helpful medical assistant. Given a doctor-patient conversation transcript, generate a clear, concise summary understandable to both doctor and patient.

Your summary must include as many of the following as possible, based only on what is actually mentioned in the conversation:
- Patient's main complaint or condition.
- Any relevant history or self-treatment the patient attempted.
- Possible causes or contributing factors discussed.
- Diagnostic steps or assessments performed by the doctor.
- The doctor’s explanation of the condition (if any).
- Clearly state the treatment plan or advice.
- Any follow-up instructions or prognosis.
- Any advice or lifestyle recommendations.
- Be easy for patient and doctor to understand.
- Be complete but not unnecessarily verbose.

Do not guess or hallucinate missing information. Use simple language. Organize as short paragraphs or bullet points. Be complete but not verbose.

Start with: **Patient Summary:**

If any item is not covered, skip it politely.
"""

        payload = {
            "prompt": f"Provide a clear, concise summary of the following conversation:\n\n{transcript_text}",
            "systemMessage": system_message,
            "temperature": 0.0,
            "maxTokens": 500
        }

        ozwell_response = requests.post(ozwell_url, headers=headers, json=payload)
        ozwell_summary = "Could not get summary."
        if ozwell_response.ok:
            try:
                ozwell_summary = ozwell_response.json()["choices"][0]["message"]["content"]
                # Save the summary to a new file
                summary_file = UPLOAD_DIR / (stem + "_summary.txt")
                with open(summary_file, "w", encoding="utf-8") as f:
                    f.write(ozwell_summary)
            except Exception as e:
                print("Error parsing Ozwell summary:", str(e))
        
        # Keep all recent artifacts for debugging; retention cleanup handles disk usage.
        if not KEEP_UPLOAD_ARTIFACTS:
            set_latest_stem(stem)

        return jsonify({
            "message": "Diarization and summarization done",
            "filename": filename,
            "transcript": transcript_text,
            "summary": ozwell_summary,
            "diarization_json": diarization_json
        }), 200
    except subprocess.CalledProcessError as e:
        return jsonify({
            "error": "Diarization failed",
            "details": e.stderr
        }), 500



@app.route('/api/test_ozwell', methods=['GET'])
def test_ozwell():
    try:
        headers = {
            "Authorization": f"Bearer {OZWELL_API_KEY}",
            "Content-Type": "application/json"
        }

        # This URL is for testing credentials only
        response = requests.post(
            "https://ai.bluehive.com/api/v1/test-credentials",
            headers=headers
        )

        print("Ozwell Response:", response.text)
        return jsonify({"ozwell_reply": response.json()}), response.status_code

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/api/ozwell_chat', methods=['POST'])
def ozwell_chat():
    try:
        user_input = request.json.get("message")
        if not user_input:
            return jsonify({"error": "No message provided"}), 400

        headers = {
            "Authorization": f"Bearer {OZWELL_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "prompt": user_input,
            "systemMessage": "You are a helpful assistant."
        }

        response = requests.post(
            "https://ai.bluehive.com/api/v1/completion",
            headers=headers,
            json=payload
        )

        print("Ozwell Chat Status Code:", response.status_code)
        print("Ozwell Chat Raw Response:", response.text)

        return jsonify(response.json()), response.status_code

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)