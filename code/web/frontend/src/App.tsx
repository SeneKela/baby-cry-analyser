import { useState, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Mic, Square, Upload } from "lucide-react";
import { ResultCard } from "./components/ResultCard";
import './App.css';

interface AnalysisResult {
  type: string;
  confidence: number;
  description: string;
  suggestions: string[];
  color: string;
  icon: string;
}

// Map API response to result format
const mapPredictionToResult = (prediction: string, confidence: number, interpretation: string): AnalysisResult => {
  const mapping: Record<string, Partial<AnalysisResult>> = {
    hungry: {
      type: "Hungry",
      color: "#FF8389",
      icon: "🍼"
    },
    sleepy: {
      type: "Tired",
      color: "#5856D6",
      icon: "😴"
    },
    pain: {
      type: "Pain",
      color: "#FF3B30",
      icon: "😢"
    },
    discomfort: {
      type: "Uncomfortable",
      color: "#FF9500",
      icon: "🔄"
    },
    diaper: {
      type: "Needs Change",
      color: "#34C759",
      icon: "👶"
    }
  };

  const result = mapping[prediction.toLowerCase()] || {
    type: prediction.charAt(0).toUpperCase() + prediction.slice(1),
    color: "#34C759",
    icon: "💚"
  };

  // Parse interpretation from backend (markdown format from utils.py)
  // Format: ### 🍼 CrySense Analysis\n**Detected Category:** **{prediction}**\n**Confidence:** {confidence}%\n**Urgency:** {urgency}\n\n**Recommendation:**\n{emoji} {text}

  const lines = interpretation.split('\n').map(line => line.trim()).filter(line => line);

  // Find the recommendation line (comes after "**Recommendation:**")
  const recommendationIndex = lines.findIndex(line => line.includes('**Recommendation:**'));
  const recommendationText = recommendationIndex >= 0 && recommendationIndex + 1 < lines.length
    ? lines[recommendationIndex + 1]
    : '';

  // Extract the description (remove emoji prefix if present)
  const description = recommendationText.replace(/^[🍼⚠️😴👕👶]\s*/, '').trim();

  // Split the recommendation into multiple suggestions (by period or sentence)
  const suggestions = description
    ? description.split('.').map(s => s.trim()).filter(s => s.length > 0).map(s => s + (s.endsWith('.') ? '' : '.'))
    : [`Monitor your baby's ${prediction} signals`];

  return {
    ...result,
    confidence: Math.round(confidence * 100),
    description: description || `Your baby may be experiencing ${prediction}`,
    suggestions: suggestions
  } as AnalysisResult;
};

// API URL - uses environment variable in production, localhost in development
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const mimeType = MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : 'audio/ogg';

      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType });
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType });
        const extension = mimeType.includes('webm') ? 'webm' : 'ogg';
        analyzeAudio(blob, `recording.${extension}`);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setError(null);
      setShowResult(false);
    } catch (err) {
      setError("Could not access microphone.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
  };

  const handleRecord = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const handleUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      analyzeAudio(file, file.name);
    }
  };

  const analyzeAudio = async (blob: Blob, filename: string) => {
    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', blob, filename);

    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Analysis failed');

      const data = await response.json();
      console.log('Backend API Response:', data);
      const result = mapPredictionToResult(data.prediction, data.confidence, data.interpretation);
      console.log('Mapped Result:', result);
      setAnalysisResult(result);
      setShowResult(true);
    } catch (err) {
      setError("Analysis failed. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="relative size-full min-h-screen flex items-center justify-center bg-gray-50 overflow-hidden">
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Ambient background effect */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-br from-red-100/30 via-pink-100/30 to-orange-100/30"
        animate={{
          scale: isRecording ? [1, 1.1, 1] : 1,
          opacity: isRecording ? [0.3, 0.5, 0.3] : 0.3,
        }}
        transition={{
          duration: 3,
          repeat: isRecording ? Infinity : 0,
          ease: "easeInOut"
        }}
      />

      <div className="relative w-full max-w-6xl aspect-[1512/982]">
        <svg className="block size-full" fill="none" preserveAspectRatio="xMidYMid meet" viewBox="0 0 1512 982">
          <defs>
            <linearGradient gradientUnits="userSpaceOnUse" id="paint0_linear_1_47" x1="45.5" x2="1483.5" y1="36" y2="968">
              <stop stopColor="#FFC2C5" />
              <stop offset="1" stopColor="#FA4D56" />
            </linearGradient>
            <filter colorInterpolationFilters="sRGB" filterUnits="userSpaceOnUse" height="106" id="filter0_d_1_47" width="106" x="703" y="742">
              <feFlood floodOpacity="0" result="BackgroundImageFix" />
              <feColorMatrix in="SourceAlpha" result="hardAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" />
              <feOffset dy="4" />
              <feGaussianBlur stdDeviation="2" />
              <feComposite in2="hardAlpha" operator="out" />
              <feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.1 0" />
              <feBlend in2="BackgroundImageFix" mode="normal" result="effect1_dropShadow_1_47" />
              <feBlend in="SourceGraphic" in2="effect1_dropShadow_1_47" mode="normal" result="shape" />
            </filter>
          </defs>

          <rect fill="url(#paint0_linear_1_47)" height="982" rx="12" width="1512" />

          {/* Pulsing circles when recording */}
          <AnimatePresence>
            {isRecording && (
              <>
                <motion.circle
                  cx="755.5"
                  cy="490.5"
                  fill="#FF8389"
                  fillOpacity="0.15"
                  r="397.5"
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{
                    scale: [0.8, 1.2, 0.8],
                    opacity: [0, 0.15, 0]
                  }}
                  exit={{ scale: 0, opacity: 0 }}
                  transition={{
                    duration: 2.5,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                />
                <motion.circle
                  cx="755.5"
                  cy="490.5"
                  fill="#FF8389"
                  fillOpacity="0.2"
                  r="460"
                  stroke="#FF8389"
                  strokeWidth="3"
                  strokeOpacity="0.3"
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{
                    scale: [0.8, 1.15, 0.8],
                    opacity: [0, 0.3, 0],
                    strokeOpacity: [0, 0.5, 0]
                  }}
                  exit={{ scale: 0, opacity: 0 }}
                  transition={{
                    duration: 2.5,
                    repeat: Infinity,
                    ease: "easeInOut",
                    delay: 0.4
                  }}
                />
                <motion.circle
                  cx="755.5"
                  cy="490.5"
                  fill="#FF8389"
                  fillOpacity="0.5"
                  r="288.5"
                  initial={{ scale: 0.9, opacity: 0 }}
                  animate={{
                    scale: [0.9, 1.05, 0.9],
                    opacity: [0.3, 0.6, 0.3]
                  }}
                  exit={{ scale: 0, opacity: 0 }}
                  transition={{
                    duration: 2,
                    repeat: Infinity,
                    ease: "easeInOut",
                    delay: 0.8
                  }}
                />
              </>
            )}
          </AnimatePresence>

          {/* Analyzing state indicator */}
          {isAnalyzing && (
            <motion.circle
              cx="755.5"
              cy="490.5"
              fill="none"
              stroke="#FF8389"
              strokeWidth="4"
              r="130"
              initial={{ pathLength: 0, rotate: 0 }}
              animate={{
                pathLength: [0, 1, 0],
                rotate: 360
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                ease: "linear"
              }}
              style={{ transformOrigin: "center" }}
            />
          )}

          {/* Center microphone button */}
          <g className="cursor-pointer" onClick={handleRecord}>
            <motion.circle
              cx="755.5"
              cy="490.5"
              fill={isRecording ? "#FA4D56" : isAnalyzing ? "#FFD6D8" : "#FFB3B8"}
              r="108.5"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            />
            <g>
              <rect fill="white" fillOpacity="0.01" height="101" transform="translate(705 440)" width="101" />
              {isRecording ? (
                <foreignObject x="705" y="440" width="101" height="101">
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>
                    <Square size={48} color="#F4F4F4" fill="#F4F4F4" />
                  </div>
                </foreignObject>
              ) : (
                <foreignObject x="705" y="440" width="101" height="101">
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>
                    <Mic size={56} color="#F4F4F4" strokeWidth={2} />
                  </div>
                </foreignObject>
              )}
            </g>
          </g>

          {/* Upload button */}
          {!isRecording && !isAnalyzing && (
            <g className="cursor-pointer" onClick={handleUpload} filter="url(#filter0_d_1_47)">
              <motion.circle
                cx="756"
                cy="791"
                fill="#FFB3B8"
                r="49"
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0, opacity: 0 }}
                whileHover={{ scale: 1.15, y: -2 }}
                whileTap={{ scale: 0.85 }}
                transition={{ type: "spring", stiffness: 400, damping: 20 }}
              />
              <g>
                <foreignObject x="731" y="766" width="51" height="51">
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>
                    <Upload size={28} color="#F4F4F4" strokeWidth={2.5} />
                  </div>
                </foreignObject>
              </g>
            </g>
          )}

          {/* Status text */}
          <text x="756" y="650" textAnchor="middle" fill="#FFFFFF" fontSize="24" fontFamily="Inter, sans-serif" fontWeight="500">
            {isAnalyzing ? "Analyzing..." : isRecording ? "Tap to stop" : "Tap to listen"}
          </text>
        </svg>

        {/* Error message */}
        {error && (
          <motion.div
            className="absolute bottom-8 left-1/2 -translate-x-1/2 bg-red-500 text-white px-6 py-3 rounded-full shadow-lg"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
          >
            {error}
          </motion.div>
        )}

        {/* Result Card */}
        <AnimatePresence>
          {showResult && analysisResult && (
            <ResultCard
              result={analysisResult}
              onClose={() => setShowResult(false)}
            />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
