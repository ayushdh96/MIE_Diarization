import argparse
import logging
import os
import re
import sys
import shutil

import faster_whisper
import torch
import torchaudio
import json
import time
import pickle
from typing import Optional
import whisper
from dataclasses import dataclass
from whisperx.vads.pyannote import Pyannote

from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from nemo.collections.asr.models import EncDecSpeakerLabelModel

from helpers import (
    cleanup,
    create_config,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)

mtypes = {"cpu": "int8", "cuda": "float16"}

# --- Dataclasses for segment/info compatibility ---
@dataclass
class SegmentObj:
    start: float
    end: float
    text: str

@dataclass
class InfoObj:
    language: str

mtypes = {"cpu": "int8", "cuda": "float16"}
mtypes = {"cpu": "int8", "cuda": "float16"}

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=False, default=None
)
parser.add_argument(
    "--no-stem",
    action="store_false",
    dest="stemming",
    default=True,
    help="Disables source separation."
    "This helps with long files that don't contain a lot of music.",
)

parser.add_argument(
    "--suppress_numerals",
    action="store_true",
    dest="suppress_numerals",
    default=False,
    help="Suppresses Numerical Digits."
    "This helps the diarization accuracy but converts all digits into written text.",
)

parser.add_argument(
    "--whisper-model",
    dest="model_name",
    default="medium.en",
    help="name of the Whisper model to use",
)

parser.add_argument(
    "--batch-size",
    type=int,
    dest="batch_size",
    default=8,
    help="Batch size for batched inference, reduce if you run out of memory, "
    "set to 0 for original whisper longform inference",
)

parser.add_argument(
    "--language",
    type=str,
    default=None,
    choices=whisper_langs,
    help="Language spoken in the audio, specify None to perform language detection",
)

parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)

parser.add_argument(
    "--mode",
    type=str,
    choices=["full", "asr"],
    default="full",
    help="Select 'full' (ASR + CTC aligner + VAD + NeMo) or 'asr' for Faster-Whisper-only transcription."
)

# -------- Speaker enrollment CLI args --------
parser.add_argument(
    "--enroll-speaker",
    action="store_true",
    help="Enrollment mode: create and store a speaker embedding from --speaker-audio under --speaker-label, then exit.",
)
parser.add_argument(
    "--speaker-audio",
    type=str,
    default=None,
    help="Path to the speaker enrollment audio file (e.g., Speaker Audios/Ayush.mp3). Required when --enroll-speaker is set.",
)
parser.add_argument(
    "--speaker-label",
    type=str,
    default=None,
    help="Label/name for the enrolled speaker (e.g., Ayush). Required when --enroll-speaker is set.",
)

parser.add_argument(
    "--speakers-db",
    type=str,
    default=os.path.join("Speaker Audios", "speakers_db.json"),
    help="Path to JSON file where enrolled speaker embeddings are stored.",
)

# -------- Known-speaker identification (MVP: args only for now) --------
parser.add_argument(
    "--identify-known",
    action="store_true",
    help="If set (full mode), match diarized speakers against enrolled embeddings in --speakers-db and annotate output JSON.",
)
parser.add_argument(
    "--candidate-labels",
    type=str,
    default=None,
    help="Optional comma-separated list of expected speaker labels (e.g., 'Ayush,Amber'). If provided, matching will be restricted to these labels when they exist in the DB; otherwise it will fall back to all enrolled speakers.",
)
parser.add_argument(
    "--id-threshold",
    type=float,
    default=0.68,
    help="Cosine similarity threshold for assigning a known speaker label (open-set).",
)
parser.add_argument(
    "--top-k",
    type=int,
    default=1,
    help="Top-K candidates considered for identification (reserved for next step).",
)


args = parser.parse_args()

# ---------- Speaker enrollment mode (A3: NeMo embedding centroid, fallback to MFCC) ----------

def _l2_normalize(vec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = torch.norm(vec) + eps
    return vec / denom

def _compute_mfcc_embedding(audio_16k_mono: "torch.Tensor") -> "torch.Tensor":
    """Compute a simple fixed-length speaker embedding from 16kHz mono waveform.

    This is an MVP embedding (MFCC mean+std). We can swap to a proper speaker model later
    without changing the enrollment interface.
    """
    if audio_16k_mono.dim() != 1:
        audio_16k_mono = audio_16k_mono.view(-1)

    # MFCC: [n_mfcc, frames]
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=40,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80},
    )(audio_16k_mono.unsqueeze(0)).squeeze(0)

    # Basic stats over time to form a stable vector
    mu = mfcc.mean(dim=-1)
    sd = mfcc.std(dim=-1)
    emb = torch.cat([mu, sd], dim=0).float()  # 80-dim
    return _l2_normalize(emb)

# ---------- Option B: NeMo speaker-model embeddings (no MSDD artifacts) ----------
_NEMO_SPK_MODEL = None

def _get_nemo_speaker_model(device: str):
    """Lazy-load a NeMo speaker embedding model once per process."""
    global _NEMO_SPK_MODEL
    if _NEMO_SPK_MODEL is not None:
        return _NEMO_SPK_MODEL

    # If CUDA requested but not available, fall back to CPU
    dev = device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"

    # TiTANet is a commonly used NeMo speaker embedding model.
    model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
    model = model.to(dev)
    model.eval()
    _NEMO_SPK_MODEL = model
    return _NEMO_SPK_MODEL

def _compute_nemo_speaker_embedding(audio_16k_mono: "torch.Tensor", device: str) -> "torch.Tensor":
    """Compute a speaker embedding using a NeMo speaker model.

    We write a short temp wav and call NeMo's `get_embedding` API.
    Returns an L2-normalized 1D torch.Tensor.
    """
    if audio_16k_mono.dim() != 1:
        audio_16k_mono = audio_16k_mono.view(-1)

    # Write a temp wav for NeMo (expects file paths)
    ROOT = os.getcwd()
    tmp_dir = os.path.join(ROOT, "temp_outputs", "spk_emb_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_wav = os.path.join(tmp_dir, f"spk_{int(time.time() * 1000)}.wav")

    torchaudio.save(
        tmp_wav,
        audio_16k_mono.view(1, -1).float(),
        16000,
        channels_first=True,
    )

    try:
        model = _get_nemo_speaker_model(device)
        # NeMo versions differ in `get_embedding` signature.
        # Try positional first (most common), then fall back.
        try:
            emb_np = model.get_embedding([tmp_wav])
        except TypeError:
            emb_np = model.get_embedding(tmp_wav)
        # emb_np is typically shape (1, D) but some versions may return (D,)
        if hasattr(emb_np, "shape") and len(getattr(emb_np, "shape", [])) == 1:
            emb = torch.tensor(emb_np, dtype=torch.float32)
        else:
            emb = torch.tensor(emb_np[0], dtype=torch.float32)
        emb = _l2_normalize(emb)
        return emb
    finally:
        try:
            os.remove(tmp_wav)
        except Exception:
            pass

# ---------- End Option B helpers ----------

# ---------- Helper for building cluster audio (used for both enrollment and cluster embedding) ----------
def _build_cluster_audio(audio_16k: "torch.Tensor", intervals_ms: list, max_seconds: float = 15.0) -> "torch.Tensor":
    """Concatenate up to max_seconds of audio for a given speaker from diarization intervals."""
    max_len = int(max_seconds * 16000)
    chunks = []
    used = 0
    for s_ms, e_ms in intervals_ms:
        s_idx = max(0, int((s_ms / 1000.0) * 16000))
        e_idx = max(0, int((e_ms / 1000.0) * 16000))
        if e_idx <= s_idx:
            continue
        seg = audio_16k[s_idx:e_idx]
        if seg.numel() == 0:
            continue
        take = min(seg.numel(), max_len - used)
        if take <= 0:
            break
        chunks.append(seg[:take])
        used += take
        if used >= max_len:
            break
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks, dim=0)

if getattr(args, "enroll_speaker", False):
    if not args.speaker_audio or not args.speaker_label:
        print("[ERROR] --enroll-speaker requires both --speaker-audio and --speaker-label")
        sys.exit(2)

    if not os.path.exists(args.speaker_audio):
        print(f"[ERROR] Speaker audio not found: {args.speaker_audio}")
        sys.exit(2)

    # Load audio via ffmpeg (supports mp3/webm/etc.) and ensure 16k mono float32
    try:
        wav = faster_whisper.decode_audio(args.speaker_audio)  # numpy float32, 16k mono
    except Exception as e:
        print(f"[ERROR] Failed to decode speaker audio: {e}")
        sys.exit(2)

    audio_t = torch.from_numpy(wav).float()

    # Guardrail: ensure at least ~5 seconds of audio
    min_len = 5 * 16000
    if audio_t.numel() < min_len:
        print("[ERROR] Enrollment audio too short. Please provide at least 5 seconds of speech.")
        sys.exit(2)

    label = args.speaker_label.strip()

    # Try to compute a NeMo-derived speaker centroid embedding from the enrollment audio.
    # If anything fails, fall back to the MFCC MVP embedding so enrollment never blocks.
    emb = None
    emb_type = None
    emb_dim = None

    try:
        # Build a short speech-only sample using Pyannote VAD, then embed with a NeMo speaker model.
        vad_pipeline = Pyannote(
            device=args.device,
            use_auth_token=hf_token,
            vad_onset=0.5,
            vad_offset=0.363,
        )

        # Save a temp wav to avoid mp3 decoding edge cases
        ROOT = os.getcwd()
        enroll_tmp = os.path.join(ROOT, "temp_outputs", "enroll_tmp")
        os.makedirs(enroll_tmp, exist_ok=True)
        mono_path = os.path.join(enroll_tmp, "enroll_mono.wav")
        torchaudio.save(mono_path, audio_t.view(1, -1).float(), 16000, channels_first=True)

        segmentation_raw = vad_pipeline({
            "uri": os.path.splitext(os.path.basename(mono_path))[0],
            "audio": mono_path,
        })
        segmentation_output = Pyannote.merge_chunks(
            segmentation_raw,
            chunk_size=30,
            onset=0.5,
            offset=0.363,
        )

        # Collect up to ~15 seconds of speech from VAD segments
        intervals = []
        for speech in segmentation_output:
            for start, end in speech.get("segments", []):
                intervals.append((float(start) * 1000.0, float(end) * 1000.0))

        speech_audio = _build_cluster_audio(audio_t.view(-1), intervals, max_seconds=15.0)
        secs = float(speech_audio.numel()) / 16000.0 if speech_audio.numel() else 0.0
        if speech_audio.numel() < 3 * 16000:
            raise ValueError(f"Not enough speech after VAD for enrollment ({secs:.2f}s). Provide cleaner/longer speech.")

        emb = _compute_nemo_speaker_embedding(speech_audio, args.device)
        emb_type = "nemo_titanet_centroid_v1"
        emb_dim = int(emb.numel())

        print(f"[INFO] Enrollment embedding computed using NeMo speaker model: dim={emb_dim} speech_used={secs:.2f}s")

    except Exception as e:
        print(f"[WARN] NeMo enrollment embedding failed; falling back to MFCC embedding. Reason: {e}")
        emb = _compute_mfcc_embedding(audio_t)
        emb_type = "mfcc_mean_std_v1"
        emb_dim = int(emb.numel())

    # Load or create DB
    db_path = args.speakers_db
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    db = {"version": 1, "speakers": {}}
    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                db = json.load(f) or db
        except Exception:
            # If corrupted, keep a fresh DB rather than crashing
            db = {"version": 1, "speakers": {}}

    if "speakers" not in db or not isinstance(db["speakers"], dict):
        db["speakers"] = {}

    record = {
        "label": label,
        "embedding": emb.tolist(),
        "embedding_type": emb_type,
        "embedding_dim": emb_dim,
        "sample_rate": 16000,
        "source_audio": os.path.basename(args.speaker_audio),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    db["speakers"][label] = record

    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Enrolled speaker '{label}' -> {db_path}")
    sys.exit(0)

# ---------- End speaker enrollment mode ----------

# In normal (non-enrollment) operation, --audio is required.
if not args.audio:
    print("[ERROR] Missing required argument: -a/--audio")
    sys.exit(2)

# ---------- Known-speaker identification (Step 1: load DB + candidate filter) ----------

known_speakers_db = {"version": 1, "speakers": {}}
known_enrolled = {}
known_candidate_labels = []
known_candidates = {}
known_used_fallback = False

def _safe_load_speakers_db(db_path: str) -> dict:
    if not db_path or not os.path.exists(db_path):
        return {"version": 1, "speakers": {}}
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        if not isinstance(data, dict):
            return {"version": 1, "speakers": {}}
        if "speakers" not in data or not isinstance(data.get("speakers"), dict):
            data["speakers"] = {}
        return data
    except Exception:
        return {"version": 1, "speakers": {}}

def _parse_candidate_labels(s: Optional[str]) -> list:
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]

if getattr(args, "identify_known", False):
    known_speakers_db = _safe_load_speakers_db(getattr(args, "speakers_db", ""))
    known_enrolled = known_speakers_db.get("speakers", {}) if isinstance(known_speakers_db, dict) else {}

    known_candidate_labels = _parse_candidate_labels(getattr(args, "candidate_labels", None))

    if known_candidate_labels:
        for lab in known_candidate_labels:
            if lab in known_enrolled:
                known_candidates[lab] = known_enrolled[lab]

    # Fallback to all enrolled if candidate list is empty/missing/invalid
    if not known_candidates:
        known_candidates = dict(known_enrolled)
        known_used_fallback = bool(known_candidate_labels)

    print(
        f"[INFO] identify-known enabled: enrolled={len(known_enrolled)} candidates={len(known_candidates)} "
        f"fallback={'yes' if known_used_fallback else 'no'} db='{getattr(args, 'speakers_db', '')}'"
    )

    if known_candidate_labels:
        print(f"[INFO] candidate-labels requested: {known_candidate_labels}")
        if known_used_fallback:
            print("[WARN] none of the requested candidate labels were found in DB; falling back to all enrolled speakers")

    if not known_candidates:
        print("[WARN] speakers DB empty or missing; identification will be skipped in later steps")

# ---------- End Step 1 ----------

language = process_language_arg(args.language, args.model_name)

if args.stemming:
    # Isolate vocals from the rest of the audio

    return_code = os.system(
        f'python -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o temp_outputs --device "{args.device}"'
    )

    if return_code != 0:
        logging.warning(
            "Source splitting failed, using original audio file. "
            "Use --no-stem argument to disable it."
        )
        vocal_target = args.audio
    else:
        vocal_target = os.path.join(
            "temp_outputs",
            "htdemucs",
            os.path.splitext(os.path.basename(args.audio))[0],
            "vocals.wav",
        )
else:
    vocal_target = args.audio


# Transcribe the audio file
# asr block using faster_whisper (original)
# asr_device = args.device  
# whisper_model = faster_whisper.WhisperModel(
#     args.model_name, device=asr_device, compute_type=mtypes[asr_device]
# )
# whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
# audio_waveform = faster_whisper.decode_audio(vocal_target)
# suppress_tokens = (
#     find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
#     if args.suppress_numerals
#     else [-1]
# )
#
# if args.batch_size > 0:
#     transcript_segments, info = whisper_pipeline.transcribe(
#         audio_waveform,
#         language,
#         suppress_tokens=suppress_tokens,
#         batch_size=args.batch_size,
#         word_timestamps=True if args.mode == "asr" else False,
#     )
# else:
#     transcript_segments, info = whisper_model.transcribe(
#         audio_waveform,
#         language,
#         suppress_tokens=suppress_tokens,
#         vad_filter=False if args.device == "cuda" else True,
#         word_timestamps=True if args.mode == "asr" else False,
#     )
#
# transcript_segments = list(transcript_segments)
# print(f"[DEBUG] Number of segments: {len(transcript_segments)}")
#
# full_transcript = "".join(segment.text for segment in transcript_segments)

# --- New Whisper (PyTorch) ASR block (no onnx / Silero VAD) ---
# Decide ASR device: honor --device, but fall back if CUDA isn't actually available
asr_device = args.device
if asr_device == "cuda" and not torch.cuda.is_available():
    print("[WARN] CUDA requested for ASR but not available, falling back to CPU")
    asr_device = "cpu"
print(f"[DEBUG] ASR device: {asr_device}")

# Decode audio with faster_whisper utility (ffmpeg + 16k mono)
audio_waveform = faster_whisper.decode_audio(vocal_target)

# Load OpenAI Whisper model on the chosen device
# Note: args.model_name (e.g., 'medium.en') should be compatible with whisper.load_model
whisper_model = whisper.load_model(args.model_name, device=asr_device)

# Run transcription once; batch_size is not used in this path
# fp16 only makes sense when running on CUDA
result = whisper_model.transcribe(
    audio_waveform,
    language=language,
    fp16=True if asr_device == "cuda" else False,
)

# Wrap segments into a small object with .start/.end/.text to keep rest of the code unchanged
transcript_segments = [
    SegmentObj(start=seg["start"], end=seg["end"], text=seg["text"])
    for seg in result.get("segments", [])
]

# Build an info-like object so downstream code (punctuation, JSON, etc.) still works
detected_lang = result.get("language", None) or language
info = InfoObj(language=detected_lang)

print(f"[DEBUG] Number of segments: {len(transcript_segments)}")
full_transcript = "".join(segment.text for segment in transcript_segments)
# --- End new Whisper ASR block ---

# Print and save the Whisper segments for debugging
print("[DEBUG] Whisper segments:")
print(f"[DEBUG] Number of segments: {len(transcript_segments)}")
if not transcript_segments:
    print("[DEBUG] transcript_segments is empty.")
for segment in transcript_segments:
    print(segment)

# Save Whisper segments to file
if args.audio and transcript_segments:
    seg_file = f"{os.path.splitext(args.audio)[0]}_whisper_segments.txt"
    try:
        with open(seg_file, 'w', encoding='utf-8') as f:
            for segment in transcript_segments:
                f.write(f"{segment.start:.2f} --> {segment.end:.2f}: {segment.text.strip()}\n")
        print(f"[INFO] Whisper segments saved to {seg_file}")
    except Exception as e:
        logging.warning(f"Failed to save Whisper segments: {e}")

# ---------- ASR-only early exit ----------
def _fmt_ts(t):
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int(round((t - int(t)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def _srt_from_segments(segments):
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = float(getattr(seg, "start", 0.0))
        end = float(getattr(seg, "end", 0.0))
        text = (getattr(seg, "text", "") or "").strip()
        lines.append(str(i))
        lines.append(f"{_fmt_ts(start)} --> {_fmt_ts(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)

if args.mode == "asr":
    # TXT
    with open(f"{os.path.splitext(args.audio)[0]}.txt", "w", encoding="utf-8-sig") as f:
        f.write(full_transcript.strip())
    # SRT
    with open(f"{os.path.splitext(args.audio)[0]}.srt", "w", encoding="utf-8-sig") as srtf:
        srtf.write(_srt_from_segments(transcript_segments))
    # JSON (OpenAI-shaped, no diarization; use FW word timestamps if available)
    segments_json = []
    words_json = []
    for i, seg in enumerate(transcript_segments):
        segments_json.append({
            "id": i,
            "start": float(getattr(seg, "start", 0.0)),
            "end": float(getattr(seg, "end", 0.0)),
            "text": (getattr(seg, "text", "") or "").strip(),
            "speaker": None
        })
        if hasattr(seg, "words") and seg.words:
            for w in seg.words:
                words_json.append({
                    "word": getattr(w, "word", None),
                    "start": float(getattr(w, "start", 0.0)),
                    "end": float(getattr(w, "end", 0.0)),
                    "speaker": None
                })
    payload_asr = {
        "task": "transcribe",
        "language": info.language,
        "text": full_transcript.strip(),
        "segments": segments_json,
        "words": words_json,
        "diarization": None
    }
    json_out = f"{os.path.splitext(args.audio)[0]}.json"
    try:
        with open(json_out, "w", encoding="utf-8") as jf:
            json.dump(payload_asr, jf, ensure_ascii=False, indent=2)
        print(f"[INFO] (ASR) JSON written to {json_out}")
    except Exception as e:
        logging.warning(f"Failed to save ASR JSON output: {e}")
    # Exit before alignment/VAD/NeMo
    sys.exit(0)
# ---------- End ASR-only early exit ----------

# clear gpu vram for ASR model
try:
    del whisper_model
except NameError:
    pass
torch.cuda.empty_cache()

# Forced Alignment
alignment_model, alignment_tokenizer = load_alignment_model(
    args.device,
    dtype=torch.float16 if args.device == "cuda" else torch.float32,
)

emissions, stride = generate_emissions(
    alignment_model,
    torch.from_numpy(audio_waveform)
    .to(alignment_model.dtype)
    .to(alignment_model.device),
    batch_size=args.batch_size,
)

del alignment_model
torch.cuda.empty_cache()

tokens_starred, text_starred = preprocess_text(
    full_transcript,
    romanize=True,
    language=langs_to_iso[info.language],
)

segments, scores, blank_token = get_alignments(
    emissions,
    tokens_starred,
    alignment_tokenizer,
)

spans = get_spans(tokens_starred, segments, blank_token)

word_timestamps = postprocess_results(text_starred, spans, stride, scores)


# convert audio to mono for NeMo combatibility
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
torchaudio.save(
    os.path.join(temp_path, "mono_file.wav"),
    torch.from_numpy(audio_waveform).unsqueeze(0).float(),
    16000,
    channels_first=True,
)

# pyannote_model = PyannoteModel.from_pretrained("pyannote/segmentation-3.0", 
#   use_auth_token=hf_token)
# vad_pipeline = VoiceActivityDetection(segmentation=pyannote_model)
# HYPER_PARAMETERS = {
#     "min_duration_on": 0, # Threshold for small non_speech deletion
#     "min_duration_off": 0.2, # Threshold for short speech segment deletion
# }
# vad_pipeline.instantiate(HYPER_PARAMETERS)  
# payannote_vad = vad_pipeline(vocal_target)

# from vad.silero import apply_vad  # Silero VAD is deprecated, now using Pyannote VAD
'''from whisperx.vads.pyannote import VoiceActivitySegmentation


# Load the segmentation model
segmentation_model = PyannoteModel.from_pretrained( "pyannote/segmentation",use_auth_token=hf_token).to(args.device)

# Perform segmentation
HYPER_PARAMETERS = {
    "onset": 0.5,
    "offset": 0.363,
    "min_duration_on": 0.1,
    "min_duration_off": 0.1
}
segmentation = VoiceActivitySegmentation(segmentation=segmentation_model)
segmentation.instantiate(HYPER_PARAMETERS)
segmentation_output = segmentation({'uri': os.path.splitext(os.path.basename(vocal_target))[0],
                                    'audio': vocal_target})'''
vad_pipeline = Pyannote(
    device=args.device,
    use_auth_token=hf_token,
    vad_onset=0.5,
    vad_offset=0.363,
)

# Always run VAD on the 16kHz mono wav we created above to avoid mp3 backend issues
mono_file_path = os.path.join(temp_path, "mono_file.wav")
pyannote_manifest = os.path.join(temp_path, "pyannote_manifest.json")

segmentation_raw = vad_pipeline({
    "uri": os.path.splitext(os.path.basename(args.audio))[0] if args.audio else "mono_file",
    "audio": mono_file_path,
})

segmentation_output = Pyannote.merge_chunks(
    segmentation_raw,
    chunk_size=30,
    onset=0.5,
    offset=0.363
)

print(f"[DEBUG] Number of VAD segments: {len(segmentation_output)}")
for seg in segmentation_output[:5]:  # Just print first 5
    print(seg)

print(f"[DEBUG] Sample VAD segment: {segmentation_output[0]}")
print(f"[DEBUG] Type: {type(segmentation_output[0])}")

with open(pyannote_manifest, "w") as f:
    for speech in segmentation_output:
        for start, end in speech["segments"]:
            segment = {
                "audio_filepath": mono_file_path,
                "offset": start,
                "duration": end - start,
                "label": "speech",
                "uniq_id": "mono_file"  # Using a static ID for simplicity
            }
            f.write(f"{json.dumps(segment)}\n")
    '''for speech in segmentation_output:
        segment = {
            "audio_filepath": mono_file_path,
            "offset": speech[0],
            "duration": speech[1]-speech[0],
            "label": "speech",
            "uniq_id": segmentation_input['uri']
        }
        f.write(f"{json.dumps(segment)}\n")'''
    '''for speech in segmentation_output:
        segment = {
            "audio_filepath": mono_file_path,
            "offset": speech["start"],
            "duration": speech["end"] - speech["start"],
            "label": "speech",
            "uniq_id": os.path.splitext(os.path.basename(vocal_target))[0]
        }
        f.write(f"{json.dumps(segment)}\n")'''   
# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)
msdd_model._cfg.diarizer.manifest_filepath = pyannote_manifest
msdd_model.diarize()

del msdd_model
torch.cuda.empty_cache()



# Reading timestamps <> Speaker Labels mapping



speaker_ts = []
with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(" ")
        s = int(float(line_list[5]) * 1000)
        e = s + int(float(line_list[8]) * 1000)
        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

# ---------- Known-speaker identification (Step A4: compute NeMo cluster centroids when available; fallback to MFCC) ----------

cluster_embeddings = {}  # maps 'SPEAKER_0' -> torch.Tensor(D,)
cluster_audio_seconds = {}  # maps 'SPEAKER_0' -> float seconds used (best-effort)

if getattr(args, "identify_known", False):
    # Only proceed if diarization exists and we have any enrolled speakers
    if not known_candidates:
        print("[INFO] identify-known enabled but no candidates loaded (DB empty/missing); skipping cluster embedding computation")
    else:
        # Best-effort: record cluster durations from RTTM, regardless of embedding method
        intervals_by_spk = {}
        for s_ms, e_ms, spk_idx in speaker_ts:
            intervals_by_spk.setdefault(spk_idx, []).append((s_ms, e_ms))
        for spk_idx, intervals in intervals_by_spk.items():
            spk_label = f"SPEAKER_{spk_idx}"
            total_ms = sum(max(0, (e - s)) for s, e in intervals)
            cluster_audio_seconds[spk_label] = round(float(total_ms) / 1000.0, 3)

        try:
            audio_t_full = torch.from_numpy(audio_waveform).float().view(-1)
        except Exception:
            audio_t_full = torch.tensor(audio_waveform, dtype=torch.float32).view(-1)

        for spk_idx, intervals in intervals_by_spk.items():
            spk_label = f"SPEAKER_{spk_idx}"
            cluster_audio = _build_cluster_audio(audio_t_full, intervals, max_seconds=15.0)
            secs = float(cluster_audio.numel()) / 16000.0 if cluster_audio.numel() else 0.0

            # Guardrail: need a bit of speech for a stable embedding
            if cluster_audio.numel() < 3 * 16000:
                print(f"[INFO] {spk_label}: only {secs:.2f}s collected; skipping embedding (need >= 3.0s)")
                continue

            try:
                emb = _compute_nemo_speaker_embedding(cluster_audio, args.device)
                cluster_embeddings[spk_label] = emb
                print(f"[INFO] {spk_label}: collected={secs:.2f}s embedding_dim={int(emb.numel())} (NeMo)")
            except Exception as e:
                emb = _compute_mfcc_embedding(cluster_audio)
                cluster_embeddings[spk_label] = emb
                print(f"[WARN] {spk_label}: NeMo speaker embedding failed ({e}); using MFCC dim={int(emb.numel())}")

# ---------- End Step A4 ----------

speaker_identity_map = {}  # maps 'SPEAKER_0' -> {'name': <str|None>, 'similarity': <float|None>, 'status': 'MATCH'|'UNKNOWN'}
# ---------- Known-speaker identification (Step 4: cosine scoring, no JSON changes yet) ----------

if getattr(args, "identify_known", False):
    if not cluster_embeddings:
        print("[INFO] identify-known enabled but no cluster embeddings computed; skipping scoring")
    elif not known_candidates:
        print("[INFO] identify-known enabled but no candidates loaded; skipping scoring")
    else:
        # Prepare candidate embedding tensors
        cand_embs = {}
        for lab, rec in (known_candidates or {}).items():
            try:
                vec = rec.get("embedding")
                if not isinstance(vec, list) or len(vec) == 0:
                    continue
                t = torch.tensor(vec, dtype=torch.float32)
                t = _l2_normalize(t)
                cand_embs[lab] = t
            except Exception:
                continue

        if not cand_embs:
            print("[INFO] identify-known enabled but candidate embeddings invalid/empty; skipping scoring")
        else:
            thr = float(getattr(args, "id_threshold", 0.68))

            def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
                # a and b should be L2-normalized
                return float(torch.dot(a, b).clamp(-1.0, 1.0).item())

            # Score each diarized cluster against candidate speakers, skip mismatched dims
            for spk_label, emb_cluster in cluster_embeddings.items():
                best_lab = None
                best_sim = -1.0
                compared = 0

                for lab, emb_ref in cand_embs.items():
                    # Skip if dimensions don't match (e.g., NeMo 192 vs MFCC 80)
                    if int(emb_cluster.numel()) != int(emb_ref.numel()):
                        continue
                    sim = _cosine_sim(emb_cluster, emb_ref)
                    compared += 1
                    if sim > best_sim:
                        best_sim = sim
                        best_lab = lab

                if compared == 0:
                    print(
                        f"[WARN] SCORE {spk_label}: no candidates with matching embedding_dim={int(emb_cluster.numel())}; "
                        "skipping"
                    )
                    continue

                status = "MATCH" if (best_lab is not None and best_sim >= thr) else "UNKNOWN"
                print(f"[INFO] SCORE {spk_label}: best={best_lab} sim={best_sim:.4f} (thr={thr:.2f}) -> {status}")
                speaker_identity_map[spk_label] = {
                    "name": best_lab if status == "MATCH" else None,
                    "similarity": float(best_sim) if best_lab is not None else None,
                    "status": status,
                }

# ---------- End Step 4 ----------

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

if info.language in punct_model_langs:
    # restoring punctuation in the transcript to help realign the sentences
    punct_model = PunctuationModel(model="kredor/punctuate-all")

    words_list = list(map(lambda x: x["word"], wsm))

    labled_words = punct_model.predict(words_list, chunk_size=230)

    ending_puncts = ".?!"
    model_puncts = ".,;:!?"

    # We don't want to punctuate U.S.A. with a period. Right?
    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

    for word_dict, labeled_tuple in zip(wsm, labled_words):
        word = word_dict["word"]
        if (
            word
            and labeled_tuple[1] in ending_puncts
            and (word[-1] not in model_puncts or is_acronym(word))
        ):
            word += labeled_tuple[1]
            if word.endswith(".."):
                word = word.rstrip(".")
            word_dict["word"] = word

else:
    logging.warning(
        f"Punctuation restoration is not available for {info.language} language."
        " Using the original punctuation."
    )

wsm = get_realigned_ws_mapping_with_punctuation(wsm)
ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

with open(f"{os.path.splitext(args.audio)[0]}.txt", "w", encoding="utf-8-sig") as f:
    get_speaker_aware_transcript(ssm, f)

with open(f"{os.path.splitext(args.audio)[0]}.srt", "w", encoding="utf-8-sig") as srt:
    write_srt(ssm, srt)

# --- JSON export (OpenAI-style envelope with diarization extras) ---

def _normalize_speaker_label(sp):
    """Normalize speaker value to 'SPEAKER_<n>' for joining with speaker_identity_map."""
    if sp is None:
        return None
    # wsm currently uses int speaker indexes (0,1,...) in many paths
    if isinstance(sp, int):
        return f"SPEAKER_{sp}"
    # sometimes it may already be a label
    if isinstance(sp, str):
        s = sp.strip()
        if s.startswith("SPEAKER_"):
            return s
        # if it looks like a digit string, normalize it
        if s.isdigit():
            return f"SPEAKER_{int(s)}"
    return None

def _as_seconds(val):
    try:
        v = float(val)
    except Exception:
        return 0.0
    # if it looks like milliseconds, convert to seconds
    return v / 1000.0 if v > 1000.0 else v
def _word_time(w, *keys):
    for k in keys:
        if k in w and w[k] is not None:
            return _as_seconds(w[k])
    return 0.0

def _segment_speaker(seg, words):
    """
    Determine the dominant speaker for a Whisper segment by accumulating
    word-level speaking duration within the segment window.
    Returns speaker label (e.g., 'SPEAKER_0') or None.
    """
    try:
        start = float(seg.start)
        end = float(seg.end)
    except Exception:
        # if seg is a dict-like fallback
        start = float(getattr(seg, "start", 0.0) or seg.get("start", 0.0))
        end = float(getattr(seg, "end", 0.0) or seg.get("end", 0.0))
    duration_by_speaker = {}
    for w in words:
        ws = _word_time(w, "start", "start_time", "start_ms", "ts_start")
        we = _word_time(w, "end", "end_time", "end_ms", "ts_end")
        sp = w.get("speaker", None)
        if sp is None:
            continue
        # include words fully inside segment window
        if ws >= start and we <= end:
            duration_by_speaker[sp] = duration_by_speaker.get(sp, 0.0) + max(0.0, we - ws)
    if not duration_by_speaker:
        return None
    return max(duration_by_speaker, key=duration_by_speaker.get)

# Build segment list with speakers (fallback to [] if no segments)
segments_json = []
for i, seg in enumerate(transcript_segments or []):
    sp = _segment_speaker(seg, wsm)
    sp_norm = _normalize_speaker_label(sp)
    identity = speaker_identity_map.get(sp_norm) if getattr(args, "identify_known", False) else None
    segments_json.append({
        "id": i,
        "start": float(getattr(seg, "start", 0.0)),
        "end": float(getattr(seg, "end", 0.0)),
        "text": (getattr(seg, "text", "") or "").strip(),
        "speaker": sp,
        "speaker_identity": identity,
    })

# Build words list (already contains speaker from wsm)
words_json = []
for w in (wsm or []):
    ws = _word_time(w, "start", "start_time", "start_ms", "ts_start")
    we = _word_time(w, "end", "end_time", "end_ms", "ts_end")
    sp_w = w.get("speaker", None)
    sp_w_norm = _normalize_speaker_label(sp_w)
    identity = speaker_identity_map.get(sp_w_norm) if getattr(args, "identify_known", False) else None
    words_json.append({
        "word": w.get("word"),
        "start": ws,
        "end": we,
        "speaker": sp_w,
        "speaker_identity": identity,
    })

if words_json:
    print("[DEBUG] Sample words with times:", words_json[:5])

# Collect speakers and construct payload
unique_speakers = sorted({w.get("speaker") for w in (wsm or []) if w.get("speaker") is not None})
payload = {
    "task": "transcribe",
    "language": info.language,
    "text": full_transcript.strip(),
    "segments": segments_json,
    "words": words_json,
    "diarization": {
        "num_speakers": len(unique_speakers),
        "method": {"vad": "pyannote", "embedding": "nemo-titanet", "clustering": "msdd"}
    }
}

json_out = f"{os.path.splitext(args.audio)[0]}.json"
try:
    with open(json_out, "w", encoding="utf-8") as jf:
        json.dump(payload, jf, ensure_ascii=False, indent=2)
    print(f"[INFO] JSON written to {json_out}")
except Exception as e:
    logging.warning(f"Failed to save JSON output: {e}")


cleanup(temp_path)
