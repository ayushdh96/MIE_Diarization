import argparse
import logging
import os
import re
import sys

import faster_whisper
import torch
import torchaudio
import json
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
    "-a", "--audio", help="name of the target audio file", required=True
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

args = parser.parse_args()
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

segmentation_raw = vad_pipeline({
    "uri": os.path.splitext(os.path.basename(vocal_target))[0],
    "audio": vocal_target
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

mono_file_path = os.path.join(temp_path, "mono_file.wav")
pyannote_manifest = os.path.join(temp_path, "pyannote_manifest.json")
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
    segments_json.append({
        "id": i,
        "start": float(getattr(seg, "start", 0.0)),
        "end": float(getattr(seg, "end", 0.0)),
        "text": (getattr(seg, "text", "") or "").strip(),
        "speaker": sp
    })

# Build words list (already contains speaker from wsm)
words_json = []
for w in (wsm or []):
    ws = _word_time(w, "start", "start_time", "start_ms", "ts_start")
    we = _word_time(w, "end", "end_time", "end_ms", "ts_end")
    words_json.append({
        "word": w.get("word"),
        "start": ws,
        "end": we,
        "speaker": w.get("speaker", None)
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
        "method": {"vad": "pyannote", "embedding": "nemo-ecapa", "clustering": "msdd"}
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
