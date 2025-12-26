import React from "react";

const StatusBanner = ({ isLoading, isComplete, mode = "full" }) => {
  const taskLabel = mode === "asr" ? "Transcription" : "Diarization";
  return (
    <div className="w-full text-center my-4">
      {isLoading && (
        <div className="text-blue-600 font-semibold animate-pulse">
          {taskLabel} in progress...
        </div>
      )}
      {isComplete && (
        <div className="text-green-600 font-semibold">
          {taskLabel} complete ✅
        </div>
      )}
    </div>
  );
};

export default StatusBanner;