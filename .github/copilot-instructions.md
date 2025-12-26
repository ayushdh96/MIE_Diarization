# GitHub Copilot Instructions for MIE Diarization

## Project Overview

MIE_Diarization is a full-stack audio diarization and summarization project that combines:
- **Backend**: Python (Flask) for audio processing, transcription, and diarization
- **Frontend**: React + Vite with Tailwind CSS and Shadcn UI components
- **Core Technologies**: Pyannote, NeMo, Faster-Whisper, Demucs, CTC Forced Aligner

The system processes audio files to identify speakers (diarization), transcribe speech, and generate intelligent summaries using Ozwell AI. It supports two processing modes: **full** (diarization + transcription) and **asr** (transcription-only), plus an experimental speaker enrollment/identification feature.

## Architecture

### Backend (`/whisper-diarization`)
- **Language**: Python 3.11
- **Framework**: Flask with CORS enabled
- **Main Components**:
  - `app.py`: Flask API server with endpoints for diarization, summarization, and file cleanup
  - `diarize.py`: Main diarization pipeline orchestration with CLI arguments
  - `diarize_parallel.py`: Parallel processing implementation (legacy)
  - `nemo_process.py`: NeMo speaker embedding and clustering
  - `helpers.py`: Utility functions for text processing, alignment, and SRT generation
- **Key Dependencies**: faster-whisper, nemo_toolkit, pyannote, demucs, flask-cors, python-dotenv

### Frontend (`/diarization-ui`)
- **Language**: JavaScript (ES2020+)
- **Framework**: React 19 with Vite
- **Styling**: Tailwind CSS 4.x + Shadcn UI components
- **Key Features**: Audio upload, real-time recording, waveform visualization (WaveSurfer.js)

### Processing Pipeline
1. **Audio Input** → Demucs (source separation for vocals)
2. **Transcription** → Faster-Whisper (ASR) → CTC Forced Aligner (word-level timestamps)
3. **Diarization** → Pyannote (VAD/segmentation) → NeMo (speaker embeddings + clustering)
4. **Post-processing** → Speaker-to-word mapping → Punctuation restoration
5. **Summarization** → Ozwell AI API (medical or general summaries)

## Development Setup

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local backend development)
- Node.js 20+ (for local frontend development)
- NVIDIA GPU with CUDA support (recommended for faster processing)

### Running with Docker (Recommended)
```bash
# Build and start both frontend and backend
docker-compose up --build

# Frontend: http://localhost:5173
# Backend: http://localhost:5001
```

### Local Backend Development
```bash
cd whisper-diarization
pip install -r requirements.txt -c constraints.txt
python app.py  # Runs on port 5001
```

### Local Frontend Development
```bash
cd diarization-ui
npm install --legacy-peer-deps
npm run dev  # Runs on port 5173
```

### Environment Variables
Create `.env` files in respective directories:
- `whisper-diarization/.env`: 
  - `HF_TOKEN` (required): Hugging Face token for Pyannote models
  - `OZWELL_API_KEY` (required): API key for Ozwell AI summarization
  - `UPLOAD_DIR` (optional, default: "uploads"): Directory for uploaded audio files
  - `KEEP_UPLOAD_ARTIFACTS` (optional, default: "false"): Set to "true" to keep all upload artifacts
  - `UPLOAD_RETENTION_HOURS` (optional, default: "24"): Hours to retain old upload files before cleanup
- Frontend connects to backend at `http://localhost:5001` by default

## Coding Standards

### Python Backend
- **Style**: Follow PEP 8 conventions
- **Imports**: Standard library → Third-party → Local imports
- **Error Handling**: Use try-except blocks with specific exceptions
- **Logging**: Use Python's `logging` module (already configured in diarize.py)
- **File Paths**: Use `os.path.join()` for cross-platform compatibility
- **API Responses**: Return JSON with appropriate HTTP status codes

### JavaScript Frontend
- **Style**: ESLint configuration in `eslint.config.js`
- **Components**: Functional components with hooks
- **Naming**: PascalCase for components, camelCase for functions/variables
- **Imports**: React imports → Third-party → Local components/utils
- **CSS**: Use Tailwind utility classes; custom styles in `.css` files when needed

## Key Components and Files

### Backend Critical Files
- `app.py`: Flask API endpoints (`/api/test`, `/api/diarize`, `/api/test_ozwell`, `/api/ozwell_chat`)
- `diarize.py`: Main diarization script with CLI arguments
- `nemo_process.py`: Speaker embedding extraction and clustering
- `helpers.py`: Text processing, alignment, SRT generation utilities
- `requirements.txt`: Python dependencies (locked versions in constraints.txt)

### Frontend Critical Files
- `src/App.jsx`: Main React application component
- `src/components/ui/MicRecorder.jsx`: Audio recorder, uploader, and processing UI
- `src/components/ui/Waveform.jsx`: WaveSurfer.js integration for audio visualization
- `src/components/ui/StatusBanner.jsx`: Loading/completion status display
- `src/components/`: Reusable UI components (Shadcn UI based)
- `vite.config.js`: Vite build configuration
- `eslint.config.js`: Linting rules

### Configuration Files
- `docker-compose.yml`: Local development multi-container setup
- `docker-compose.deploy.yml`: Production deployment configuration
- `Dockerfile`: Multi-stage build (frontend + backend)
- `nemo_msdd_configs/`: NeMo diarization model configurations

## Testing and Quality

### Backend Testing
- Test assets located in `whisper-diarization/tests/assets/`
- Run diarization manually: `python diarize.py -a <audio_file>`
- Test API endpoints: `curl http://localhost:5001/api/test`

### Frontend Testing
- Linting: `npm run lint` (in diarization-ui/)
- ESLint enforces React hooks rules and unused variable checks
- Manual testing via browser at http://localhost:5173

### Code Quality Tools
- **Python**: Follow existing patterns in codebase (no automated linter configured)
- **JavaScript**: ESLint with React hooks and refresh plugins

## Build and Deployment

### Docker Build
```bash
# Build individual stages
docker build --target backend -t mie-backend .
docker build --target frontend -t mie-frontend .

# Or use docker-compose
docker-compose build
```

### Frontend Build
```bash
cd diarization-ui
npm run build  # Outputs to dist/
npm run preview  # Preview production build
```

### Deployment
- Uses `docker-compose.deploy.yml` for production
- GitHub Actions workflows: `docker-publish.yml`, `proxmox-deployment.yml`
- Hosted on server with limited resources (4GB RAM, 4 CPU cores)

## Important Notes

### Processing Modes
The system supports two processing modes:
- **Full Mode** (`--mode full`): Complete pipeline with diarization + transcription
  - Uses Demucs for vocal separation (unless `--no-stem` is set)
  - Faster-Whisper for ASR
  - CTC Forced Aligner for word-level timestamps
  - Pyannote for VAD/segmentation
  - NeMo ECAPA-TDNN for speaker embeddings and MSDD for clustering
  - Outputs: `.txt`, `.srt`, `.json` (with speaker labels and word-level timestamps)
- **ASR Mode** (`--mode asr`): Transcription-only, bypasses diarization
  - Uses Faster-Whisper with word-level timestamps enabled
  - Skips VAD, speaker embedding, and clustering steps
  - Outputs: `.txt`, `.srt`, `.json` (no speaker labels)
  - Faster processing, lower resource usage

### Speaker Enrollment & Identification (Experimental)
The system includes experimental speaker enrollment and identification features:
- **Enrollment**: `python diarize.py --enroll-speaker --speaker-audio <path> --speaker-label <name>`
  - Stores speaker embeddings (NeMo ECAPA-TDNN or MFCC fallback) in `Speaker Audios/speakers_db.json`
  - Used for creating a database of known speakers
- **Identification**: `python diarize.py -a <audio> --identify-known --candidate-labels "Speaker1,Speaker2"`
  - Matches diarized speakers against enrolled embeddings using cosine similarity
  - `--id-threshold` (default: 0.68): Minimum similarity score for a match
  - Logs matching results to console (JSON annotation not yet implemented)
  - Requires matching embedding dimensions between enrolled and cluster embeddings

### File Management & Cleanup
- Upload directory: Configurable via `UPLOAD_DIR` environment variable (default: "uploads")
- Automatic cleanup: Old files (> `UPLOAD_RETENTION_HOURS`) are deleted on each new upload
- Artifact retention: Set `KEEP_UPLOAD_ARTIFACTS=true` to preserve all generated files
- Generated files per audio: `.txt` (transcript), `.srt` (subtitles), `.json` (diarization data), `_summary.txt` (AI summary), `_whisper_segments.txt` (debug)

### Performance Considerations
- Audio processing is resource-intensive (requires GPU for optimal performance)
- Long audio files (>1 minute) require significant memory and CPU
- Use `--batch-size` parameter to control memory usage
- Consider `--no-stem` flag for long files without music

### Known Limitations
- Overlapping speakers not fully supported
- Hosted version limited to ~1 minute audio due to resource constraints
- Model hallucinations possible during summarization

### Dependencies to Watch
- `nemo_toolkit[asr]==2.0.0rc0`: Release candidate, may have breaking changes
- `faster-whisper>=1.1.0`: Core transcription engine
- Git-based dependencies: demucs, deepmultilingualpunctuation, ctc-forced-aligner
- Frontend: React 19 (latest), Tailwind CSS 4.x

## API Endpoints

### Backend Routes
- `GET /api/test`: Health check endpoint
- `POST /api/diarize`: Main diarization endpoint
  - Accepts: `multipart/form-data` with `audio` file, `interaction_type` field, and `mode` field
  - `interaction_type`: "medical" or "general" (for summarization context)
  - `mode`: "full" (diarization + transcription) or "asr" (transcription only)
  - Returns: JSON with transcript, summary, and diarization_json (OpenAI-style format with segments, words, and speaker labels)
- `GET /api/test_ozwell`: Test Ozwell AI credentials
- `POST /api/ozwell_chat`: Interactive chat endpoint with Ozwell AI

### CORS Configuration
- Backend allows requests from `http://localhost:5173` (frontend dev server) and `https://voice.opensource.mieweb.org` (production)
- Update `app.py` CORS settings for different origins

## Workflow Integration

### CI/CD Pipelines
- **docker-publish.yml**: Builds and publishes Docker images
- **proxmox-deployment.yml**: Automated deployment to Proxmox server

## Tips for Contributors

1. **Audio Processing**: Understand the pipeline flow (see mermaid diagrams in README.md)
2. **Model Downloads**: First run downloads large models (Whisper, NeMo, Pyannote)
3. **Token Requirements**: HuggingFace token needed for Pyannote models
4. **Frontend State**: React components manage audio recording and file upload states
5. **Error Handling**: Check Flask logs and browser console for debugging
6. **File Outputs**: Generated files (.txt, .srt) saved in `whisper-diarization/uploads/`

## Common Tasks

### Adding New API Endpoints
1. Define route in `app.py` with appropriate HTTP method
2. Add CORS configuration if needed
3. Return JSON responses with proper error handling

### Adding UI Components
1. Use Shadcn UI components from `src/components/`
2. Follow existing patterns in `App.jsx`
3. Apply Tailwind classes for styling
4. Run `npm run lint` to check code quality

### Modifying Diarization Pipeline
1. Core logic in `diarize.py` with argparse configuration
2. Helper functions in `helpers.py`
3. NeMo-specific processing in `nemo_process.py`
4. Test with sample audio files in `whisper-diarization/tests/assets/`

## Getting Help

- **Documentation**: See README.md in root and subdirectories
- **Issues**: Repository issues for bug reports and feature requests
- **Videos**: Project overview and pipeline explanations linked in main README.md
- **Base Project**: Original diarization work from https://github.com/MahmoudAshraf97/whisper-diarization
