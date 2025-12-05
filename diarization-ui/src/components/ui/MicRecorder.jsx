import Waveform from "./Waveform";
import { useState, useRef, useEffect } from "react";
import { Button } from "./button";
import StatusBanner from "./StatusBanner";
import { Textarea } from "./textarea";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "./dropdown-menu";
import { ChevronDown } from "lucide-react";

const MicRecorderComponent = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioURL, setAudioURL] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [summary, setSummary] = useState("");
  const [selectedOption, setSelectedOption] = useState("Summary");
  const [interactionType, setInteractionType] = useState("Doctor-Patient");
  const [showWaveformPlayer, setShowWaveformPlayer] = useState(false);
  const [mode, setMode] = useState("full"); // "full" = diarization + transcription, "asr" = transcription only
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const analyserRef = useRef(null);
  const dataArrayRef = useRef(null);
  const audioCtxRef = useRef(null);
  const sourceRef = useRef(null);
  const audioElementRef = useRef(null);

  const secondaryOption = mode === "full" ? "Diarization" : "Transcription";

  const waveformRef = useRef(null);
  const togglePlay = () => {
    if (!waveformRef.current) return;
    waveformRef.current.playPause();
    setIsPlaying(waveformRef.current.isPlaying());
  };
  const handleAudioFinish = () => {
    setIsPlaying(false); // this resets the button to "Play"
  };
  const drawWave = () => {
    if (!canvasRef.current || !analyserRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const bufferLength = analyserRef.current.fftSize;

    analyserRef.current.getByteTimeDomainData(dataArrayRef.current);

    ctx.fillStyle = "#f3f4f6";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 2;
    ctx.strokeStyle = "#3b82f6";
    ctx.beginPath();

    const sliceWidth = canvas.width / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArrayRef.current[i] / 128.0;
      const y = (v * canvas.height) / 2;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }

      x += sliceWidth;
    }

    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();

    animationRef.current = requestAnimationFrame(drawWave);
  };

  const startRecording = async () => {
    setAudioURL(null);
    setShowWaveformPlayer(false);
    setIsPlaying(false);
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    audioCtxRef.current = new AudioContext();
    sourceRef.current = audioCtxRef.current.createMediaStreamSource(stream);
    analyserRef.current = audioCtxRef.current.createAnalyser();
    analyserRef.current.fftSize = 2048;

    const bufferLength = analyserRef.current.fftSize;
    dataArrayRef.current = new Uint8Array(bufferLength);

    sourceRef.current.connect(analyserRef.current);
    drawWave();

    mediaRecorderRef.current = new MediaRecorder(stream);
    audioChunksRef.current = [];

    mediaRecorderRef.current.ondataavailable = (e) => {
      if (e.data.size > 0) {
        audioChunksRef.current.push(e.data);
      }
    };

    mediaRecorderRef.current.onstop = () => {
      const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
      const url = URL.createObjectURL(audioBlob);
      setAudioURL(url);

      setShowWaveformPlayer(true);

      cancelAnimationFrame(animationRef.current);
      if (analyserRef.current) analyserRef.current.disconnect();
      if (sourceRef.current) sourceRef.current.disconnect();
      if (audioCtxRef.current) audioCtxRef.current.close();
    };

    mediaRecorderRef.current.start();
    setIsRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      setIsLoading(false);
      setIsComplete(false);
      setIsRecording(false);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setAudioURL(url);
      setShowWaveformPlayer(true);
      setIsPlaying(false);
    }
  };

  useEffect(() => {
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (audioCtxRef.current) audioCtxRef.current.close();
    };
  }, []);

  useEffect(() => {
    if (mode === "asr" && selectedOption === "Diarization") {
      setSelectedOption("Transcription");
    }
    if (mode === "full" && selectedOption === "Transcription") {
      setSelectedOption("Diarization");
    }
  }, [mode, selectedOption]);

  return (
    <div className="flex flex-col items-center gap-3 p-6 bg-white shadow-md rounded-xl w-full max-w-md mx-auto">
      <h2 className="text-xl font-bold">Mic Recorder for Conversation Summarization</h2>
      <StatusBanner isLoading={isLoading} isComplete={isComplete} />
      <div className="rounded border w-full max-w-full">
        {!showWaveformPlayer ? (
          <canvas ref={canvasRef} width={400} height={100} className="w-full h-24" />
        ) : (
          <Waveform ref={waveformRef} audioUrl={audioURL} onFinish={handleAudioFinish} />
        )}
      </div>
      {showWaveformPlayer && !isRecording && (
        <div className="flex justify-center mt-2">
          <button onClick={togglePlay} className="bg-blue-500 text-white px-4 py-2 rounded">
            {isPlaying ? "Pause" : "Play"}
          </button>
        </div>
      )}
      <div className="flex gap-4">
        <Button
          onClick={startRecording}
          disabled={isRecording}
          className={`${isRecording ? "bg-green-300" : "bg-green-500"
            } text-white px-4 py-2 rounded disabled:opacity-50`}
        >
          Start Recording
        </Button>
        <Button
          onClick={stopRecording}
          disabled={!isRecording}
          className={`${isRecording ? "bg-red-500" : "bg-red-300"
            } text-white px-4 py-2 rounded disabled:opacity-50`}
        >
          Stop Recording
        </Button>
        <label htmlFor="file-upload" title="Upload Audio" className="flex flex-col items-center gap-2 cursor-pointer">
          <img src="/upload-icon.png" alt="Upload" className="w-6 h-6" />
          <span className="text-xs text-gray-700">Upload Audio</span>
          <input id="file-upload" type="file" accept="audio/*" onChange={handleFileUpload} className="hidden" />
        </label>
      </div>
      <div className="w-full mt-2 mb-2">
        <div className="flex flex-col sm:flex-row gap-4">
          {/* Interaction type radios */}
          <div className="flex-1">
            <h3 className="text-sm font-medium mb-1 text-gray-700 text-center sm:text-left">
              Select Interaction Type
            </h3>
            <div className="flex flex-col gap-1 text-sm text-gray-700">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="interactionType"
                  value="Doctor-Patient"
                  checked={interactionType === "Doctor-Patient"}
                  onChange={() => {
                    console.log("Selected: Doctor-Patient");
                    setInteractionType("Doctor-Patient");
                  }}
                />
                <span>Doctor-Patient</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="interactionType"
                  value="General"
                  checked={interactionType === "General"}
                  onChange={() => {
                    console.log("Selected: General");
                    setInteractionType("General");
                  }}
                />
                <span>General</span>
              </label>
            </div>
          </div>
          {/* Processing mode radios */}
          <div className="flex-1">
            <h3 className="text-sm font-medium mb-1 text-gray-700 text-center sm:text-left">
              Select Processing Mode
            </h3>
            <div className="flex flex-col gap-1 text-sm text-gray-700">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="processingMode"
                  value="full"
                  checked={mode === "full"}
                  onChange={() => {
                    console.log("Selected mode: full (Diarization + Transcription)");
                    setMode("full");
                  }}
                />
                <span>Diarization + Transcription</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="processingMode"
                  value="asr"
                  checked={mode === "asr"}
                  onChange={() => {
                    console.log("Selected mode: asr (Transcription Only)");
                    setMode("asr");
                  }}
                />
                <span>Transcription Only</span>
              </label>
            </div>
          </div>
        </div>
      </div>
      <div>
        <Button
          onClick={async () => {
            if (!audioURL) return alert("Please record or upload an audio file first.");

            setIsLoading(true);
            setIsComplete(false);

            try {
              const response = await fetch(audioURL);
              const blob = await response.blob();
              const formData = new FormData();
              formData.append("audio", blob, "recording.webm");
              console.log("Interaction type being sent:", interactionType);
              formData.append("interaction_type", interactionType);
              console.log("Processing mode being sent:", mode);
              formData.append("mode", mode); // "full" or "asr"

              const res = await fetch("http://127.0.0.1:5001/api/diarize", {
                method: "POST",
                body: formData,
              });

              const data = await res.json();
              console.log("Diarization filename:", data.filename);
              console.log("Transcript:", data.transcript);
              console.log("Summary:", data.summary);
              // Log full diarization JSON (if provided)
              if (data.diarization_json) {
                console.log("Diarization JSON:", JSON.stringify(data.diarization_json, null, 2));
              }

              if (data.transcript) {
                setTranscript(data.transcript);
              }
              if (data.summary) {
                setSummary(data.summary);
              }

              setIsLoading(false);
              setIsComplete(true);
            } catch (err) {
              console.error("Error during diarization:", err);
              setIsLoading(false);
              setIsComplete(false);
              alert("Diarization failed. Check console for error.");
            }
          }}
          className="bg-blue-500 text-white px-4 py-2 rounded mt-0"
        >
          Summarize
        </Button>
      </div>
      <div className="w-full mt-2 mb-2">
        <h3 className="text-sm font-medium mb-1 text-gray-700 text-center">Select View</h3>
        <DropdownMenu>
          <DropdownMenuTrigger className="flex justify-center items-center bg-gray-200 text-gray-800 px-2 py-0.5 rounded w-full text-center">
            {selectedOption}
          </DropdownMenuTrigger>
          <DropdownMenuContent className="w-full">
            <DropdownMenuLabel>Select View</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onClick={() => setSelectedOption("Summary")}
              className="hover:bg-gray-100"
            >
              Summary
            </DropdownMenuItem>
            <DropdownMenuItem
              onClick={() => setSelectedOption(secondaryOption)}
              className="hover:bg-gray-100"
            >
              {secondaryOption}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      <Textarea
        className="mt-4 w-full max-w-full max-h-60 overflow-y-auto resize-none border rounded-md p-2"
        placeholder={`${selectedOption} will appear here...`}
        readOnly
        value={
          selectedOption === "Summary"
            ? summary
            : (() => {
              try {
                const parsed = JSON.parse(transcript);
                return parsed.transcript || transcript;
              } catch {
                return transcript;
              }
            })()
        }
      />
    </div>
  );
};

export default MicRecorderComponent;