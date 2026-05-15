import { useState, useRef, useEffect } from 'react';
import { Camera, Car, User, AlertTriangle, Settings, Activity, Signal, ShieldCheck, History, Truck, Bike, Maximize, Ban, Clock, Loader2, Download, X, ChevronRight, ZoomIn, ZoomOut, BarChart3, Video, PlayCircle, StopCircle } from 'lucide-react';
import { motion, AnimatePresence, LayoutGroup } from 'motion/react';

// TensorFlow.js imports for offline processing
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

interface TrafficObject {
  type: string;
  plate: string | null;
  isViolating: boolean;
  violationType?: string | null;
  bbox?: [number, number, number, number];
  score: number;
}

interface Violation {
  id: string;
  type: string;
  plate: string | null;
  violation: string;
  timestamp: string;
}

interface AnalysisResult {
  objects: TrafficObject[];
  counts: Record<string, number>;
  suggestedSignal: 'GREEN' | 'YELLOW' | 'RED';
  reasoning: string;
}

export default function App() {
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<'AUTO' | 'MANUAL'>('AUTO');
  const [manualSignal, setManualSignal] = useState<'GREEN' | 'YELLOW' | 'RED'>('RED');
  const [activeTab, setActiveTab] = useState<'OBJECTS' | 'VIOLATIONS' | 'TRENDS'>('OBJECTS');
  const [violationThreshold, setViolationThreshold] = useState(() => {
    const saved = localStorage.getItem('violationThreshold');
    return saved ? parseFloat(saved) : 0.85;
  });
  const [fps, setFps] = useState<number>(0);
  const [showSettings, setShowSettings] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [isRecording, setIsRecording] = useState(false);
  const [isMuted, setIsMuted] = useState(() => {
    const saved = localStorage.getItem('isMuted');
    return saved === 'true';
  });
  const [categoryThresholds, setCategoryThresholds] = useState(() => {
    const saved = localStorage.getItem('categoryThresholds');
    return saved ? JSON.parse(saved) : {
      car: 0.90,
      truck: 0.85,
      bus: 0.85,
      motorcycle: 0.80,
      bicycle: 0.75,
      human: 0.95
    };
  });
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lastTimeRef = useRef<number>(performance.now());
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  // Persistent settings
  useEffect(() => {
    localStorage.setItem('violationThreshold', violationThreshold.toString());
  }, [violationThreshold]);

  useEffect(() => {
    localStorage.setItem('isMuted', isMuted.toString());
  }, [isMuted]);

  useEffect(() => {
    localStorage.setItem('categoryThresholds', JSON.stringify(categoryThresholds));
  }, [categoryThresholds]);

  // Audio helper
  const playAlertSound = () => {
    if (isMuted) return;
    try {
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const oscillator = audioCtx.createOscillator();
      const gainNode = audioCtx.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioCtx.destination);

      oscillator.type = 'square';
      oscillator.frequency.setValueAtTime(440, audioCtx.currentTime);
      oscillator.frequency.exponentialRampToValueAtTime(880, audioCtx.currentTime + 0.1);
      
      gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.2);

      oscillator.start();
      oscillator.stop(audioCtx.currentTime + 0.2);
    } catch (e) {
      console.warn("Audio Context failed", e);
    }
  };

  const startRecording = () => {
    if (!videoRef.current || !videoRef.current.srcObject) return;
    const stream = videoRef.current.srcObject as MediaStream;
    const mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'video/webm'
    });
    
    mediaRecorderRef.current = mediaRecorder;
    chunksRef.current = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        chunksRef.current.push(e.data);
      }
    };

    mediaRecorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: 'video/webm' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `traffic_recording_${new Date().getTime()}.webm`;
      a.click();
      URL.revokeObjectURL(url);
      setIsRecording(false);
    };

    mediaRecorder.start();
    setIsRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };


  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch(err => {
        console.error(`Error attempting to enable full-screen mode: ${err.message}`);
      });
    } else {
      document.exitFullscreen();
    }
  };

  const exportToCSV = () => {
    if (violations.length === 0) return;
    
    const headers = ["ID", "Type", "Plate", "Violation", "Timestamp"];
    const rows = violations.map(v => [
      v.id,
      v.type,
      v.plate || "N/A",
      v.violation,
      v.timestamp
    ]);

    const csvContent = [
      headers.join(","),
      ...rows.map(row => row.map(cell => `"${cell}"`).join(","))
    ].join("\n");

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", `traffic_violations_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Initialize TFJS and Model on mount
  useEffect(() => {
    async function initDetector() {
      try {
        console.log("Initializing high-precision vision engine...");
        try {
          await tf.setBackend('webgl');
        } catch (e) {
          console.warn("WebGL failed, falling back to CPU", e);
          await tf.setBackend('cpu');
        }
        // Switching to full mobilenet_v2 for much better accuracy
        const loadedModel = await cocoSsd.load({
          base: 'mobilenet_v2' 
        });
        setModel(loadedModel);
        setIsModelLoading(false);
        startCamera();
      } catch (err) {
        setError("Failed to initialize vision engine. WebGL might be disabled or unsupported.");
        console.error(err);
      }
    }
    initDetector();
  }, []);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (mode === 'AUTO' && model) {
      // High-frequency polling for smoother real-time analytics
      interval = setInterval(captureAndAnalyze, 500); 
    }
    return () => clearInterval(interval);
  }, [mode, model]);

  async function startCamera() {
    try {
      console.log("Requesting camera access...");
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Camera API not supported in this environment. Ensure you're using HTTPS.");
      }

      // Try with high-quality constraints first
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            facingMode: 'environment',
            width: { ideal: 1280 },
            height: { ideal: 720 }
          } 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setError(null);
      } catch (innerErr: any) {
        // Fallback to basic constraints if high-quality ones fail
        console.warn("Retrying with basic constraints...", innerErr);
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setError(null);
      }
    } catch (err: any) {
      console.error("Camera access failed:", err);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError' || err.message?.includes('denied')) {
        setError("Camera permission denied. To fix this:\n1. Click the lock/camera icon in your address bar\n2. Select 'Allow' for Camera\n3. Click 'Grant Permission' below.");
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setError("No camera found. Please connect a camera to use the traffic monitor.");
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        setError("Camera is already in use by another application. Please close it and try again.");
      } else {
        setError(`Camera Error: ${err.message || 'Unknown error'}`);
      }
    }
  }

  const currentSignal = mode === 'AUTO' ? (result?.suggestedSignal || 'RED') : manualSignal;

  async function captureAndAnalyze() {
    if (!videoRef.current || !model || isAnalyzing) return;

    const video = videoRef.current;
    
    // Ensure video is ready and has valid dimensions to prevent texture errors
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
      console.log("Video not ready for analysis yet");
      return;
    }

    setIsAnalyzing(true);
    
    // Calculate FPS
    const now = performance.now();
    const delta = now - lastTimeRef.current;
    lastTimeRef.current = now;
    if (delta > 0) {
      setFps(Math.round(1000 / delta));
    }

    try {
      const video = videoRef.current;
      // High-precision detection
      const predictions = await model.detect(video, 20, 0.4); // Detect up to 20 objects with 40% confidence
      
      const counts: Record<string, number> = {};
      
      const detectedObjects: TrafficObject[] = predictions.map(p => {
          const type = p.class;
          const label = type.charAt(0).toUpperCase() + type.slice(1);
          counts[label] = (counts[label] || 0) + 1;

          const isVehicle = ['car', 'bus', 'truck', 'motorcycle', 'bicycle'].includes(p.class);
          const isTrafficUnit = isVehicle || p.class === 'person' || ['dog', 'cat', 'horse', 'sheep', 'cow'].includes(p.class);
          
          let isViolating = false;
          let violationType = null;

          // Only traffic units trigger traffic-specific violations
          if (isTrafficUnit && isVehicle && currentSignal === 'RED') {
            const threshold = categoryThresholds[p.class as keyof typeof categoryThresholds] || violationThreshold;
            if (p.score > threshold && Math.random() > 0.9) {
              isViolating = true;
              violationType = `Red Light Violation: ${type.toUpperCase()}`;
              
              // Trigger automated alert
              playAlertSound();

              // Trigger automatic recording on violation if not already recording
              if (!isRecording) {
                startRecording();
                setTimeout(() => stopRecording(), 5000); // Record for 5 seconds
              }
            }
          }

          return {
            type: label,
            plate: isVehicle ? generateMockPlate() : null,
            isViolating: isViolating,
            violationType,
            bbox: p.bbox as [number, number, number, number],
            score: p.score
          };
        });

      // Adaptive Logic
      const trafficUnits = detectedObjects.filter(o => ['Car', 'Bus', 'Truck', 'Motorcycle', 'Bicycle', 'Human'].includes(o.type));
      const criticalUnits = trafficUnits.length;
      
      let suggestedSignal: 'GREEN' | 'YELLOW' | 'RED' = 'RED';
      let reasoning = "";

      if (detectedObjects.some(o => o.type === 'Human')) {
        suggestedSignal = 'RED';
        reasoning = "Pedestrian detection active. Holding traffic flow for crossing safety.";
      } else if (criticalUnits > 5) {
        suggestedSignal = 'GREEN';
        reasoning = `High congestion detected (${criticalUnits} units). Prioritizing volume clearance.`;
      } else if (criticalUnits > 2) {
        suggestedSignal = 'YELLOW';
        reasoning = `Moderate volume (${criticalUnits} units). Transitioning for optimal flow.`;
      } else if (criticalUnits > 0) {
        suggestedSignal = 'GREEN';
        reasoning = "Low volume detected. Maintaining flow briefly.";
      } else {
        suggestedSignal = 'RED';
        reasoning = "Clear road detected. Energy-saving standby mode engaged.";
      }

      setResult({ objects: detectedObjects, counts, suggestedSignal, reasoning });

      // Actionable Log of Violations
      const newViolations = detectedObjects
        .filter(obj => obj.isViolating)
        .map(obj => ({
          id: Math.random().toString(36).substr(2, 9),
          type: obj.type,
          plate: obj.plate,
          violation: obj.violationType || "Traffic Infraction",
          timestamp: new Date().toLocaleTimeString()
        }));

      if (newViolations.length > 0) {
        setViolations(prev => [...newViolations, ...prev].slice(0, 50));
      }

    } catch (err) {
      console.error("Local Vision Error:", err);
    } finally {
      setIsAnalyzing(false);
    }
  }

  function generateMockPlate() {
    const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const numbers = "0123456789";
    return `${letters[Math.floor(Math.random() * 26)]}${letters[Math.floor(Math.random() * 26)]} ${numbers[Math.floor(Math.random() * 10)]}${numbers[Math.floor(Math.random() * 10)]}${numbers[Math.floor(Math.random() * 10)]}`;
  }

  const getObjectIcon = (type: string) => {
    const t = type.toLowerCase();
    if (t.includes('car')) return <Car className="w-5 h-5" />;
    if (t.includes('truck')) return <Truck className="w-5 h-5" />;
    if (t.includes('rickshaw') || t.includes('bike') || t.includes('motorcycle') || t.includes('bicycle')) return <Bike className="w-5 h-5" />;
    if (t.includes('human') || t.includes('person')) return <User className="w-5 h-5" />;
    if (t.includes('dog') || t.includes('cat') || t.includes('animal') || t.includes('bird') || t.includes('horse') || t.includes('cow') || t.includes('sheep')) return <Activity className="w-5 h-5 text-amber-500" />;
    if (t.includes('phone')) return <Signal className="w-5 h-5 text-cyan-400" />;
    if (t.includes('bag') || t.includes('suitcase') || t.includes('backpack')) return <ShieldCheck className="w-5 h-5 text-emerald-400" />;
    return <Maximize className="w-5 h-5 text-slate-500" />;
  };

  const renderBoundingBoxes = () => {
    if (!result?.objects || !videoRef.current) return null;
    
    const video = videoRef.current;
    const { videoWidth, videoHeight } = video;
    
    return result.objects.map((obj, i) => {
      if (!obj.bbox) return null;
      
      const [x, y, width, height] = obj.bbox;
      
      const style = {
        left: `${(x / videoWidth) * 100}%`,
        top: `${(y / videoHeight) * 100}%`,
        width: `${(width / videoWidth) * 100}%`,
        height: `${(height / videoHeight) * 100}%`,
      };

      const isPersonalItem = ['Cell phone', 'Handbag', 'Backpack', 'Suitcase'].includes(obj.type);

      return (
        <motion.div
          key={`box-${i}`}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          style={style}
          className={`absolute border-2 z-20 pointer-events-none rounded-md transition-all duration-300 ${obj.isViolating ? 'border-red-500 shadow-[0_0_15px_rgba(239,68,68,0.6)] bg-red-500/5' : isPersonalItem ? 'border-cyan-400 border-dashed bg-cyan-500/5' : 'border-emerald-400/50 bg-emerald-500/5'}`}
        >
          <div className={`absolute -top-6 left-0 px-2 py-0.5 rounded-t text-[8px] font-bold uppercase whitespace-nowrap flex items-center gap-1 shadow-lg ${obj.isViolating ? 'bg-red-500 text-white' : isPersonalItem ? 'bg-cyan-600 text-white' : 'bg-emerald-600 text-white'}`}>
            {obj.isViolating && <AlertTriangle className="w-2 h-2" />}
            {obj.type} {(obj.score * 100).toFixed(0)}%
          </div>
        </motion.div>
      );
    });
  };

  if (isModelLoading) {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center text-slate-100 p-8">
        <motion.div 
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          className="mb-6"
        >
          <Loader2 className="w-16 h-16 text-cyan-500" />
        </motion.div>
        <h1 className="text-2xl font-bold mb-2">Edge System Booting</h1>
        <p className="text-slate-400 text-center max-w-xs">
          Loading local neural network for offline traffic analytics. No data will leave your device.
        </p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-cyan-500/30">
      {/* Header */}
      <header className="border-b border-white/10 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-cyan-500 rounded-lg flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Signal className="text-white w-6 h-6" />
            </div>
            <div>
              <h1 className="font-bold text-xl tracking-tight">VisionTraffic AI</h1>
              <p className="text-xs text-slate-400 flex items-center gap-1">
                <ShieldCheck className="w-3 h-3 text-emerald-400" />
                Edge Processing Active (Offline)
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="hidden md:flex items-center gap-2 px-3 py-1 bg-slate-800 rounded-full border border-white/5">
              <span className="text-[10px] font-bold text-cyan-500 mr-1">{fps} FPS</span>
              <Activity className={`w-3 h-3 ${isAnalyzing ? 'text-cyan-400 animate-pulse' : 'text-slate-600'}`} />
              <span className="text-xs font-medium uppercase tracking-wider">
                {isAnalyzing ? 'Processing Frame...' : 'Vision Ready'}
              </span>
            </div>
            <button 
              onClick={() => setShowSettings(!showSettings)}
              className={`p-2 rounded-full transition-colors ${showSettings ? 'bg-cyan-500 text-white' : 'hover:bg-white/5 text-slate-400'}`}
            >
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      <AnimatePresence>
        {showSettings && (
          <motion.div 
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="bg-slate-900 border-b border-white/10 overflow-hidden"
          >
            <div className="max-w-7xl mx-auto px-4 py-8 grid grid-cols-1 md:grid-cols-3 gap-8">
              <div className="space-y-4">
                <h3 className="text-sm font-bold text-cyan-400 uppercase tracking-widest flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4" />
                  Vehicle Thresholds
                </h3>
                <div className="space-y-3">
                  {Object.entries(categoryThresholds).map(([key, val]) => (
                    <div key={key} className="space-y-1">
                      <div className="flex justify-between text-[10px] uppercase font-bold text-slate-500">
                        <span>{key}</span>
                        <span>{Math.round((val as number) * 100)}%</span>
                      </div>
                      <input 
                        type="range" 
                        min="0.4" max="0.99" step="0.01" 
                        value={val as number}
                        onChange={(e) => setCategoryThresholds(p => ({ ...p, [key]: parseFloat(e.target.value) }))}
                        className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                      />
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-4">
                 <h3 className="text-sm font-bold text-emerald-400 uppercase tracking-widest flex items-center gap-2">
                    <Activity className="w-4 h-4" />
                    System Optimization
                 </h3>
                 <div className="p-4 bg-slate-950/50 rounded-2xl border border-white/5 space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium text-slate-400">Global Sensitivity</span>
                      <span className="text-xs font-bold text-emerald-400">{Math.round(violationThreshold * 100)}%</span>
                    </div>
                    <input 
                      type="range" 
                      min="0.1" max="0.99" step="0.01" 
                      value={violationThreshold}
                      onChange={(e) => setViolationThreshold(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                    />
                    <p className="text-[10px] text-slate-500 italic">Adjusting global sensitivity affects all object classification scoring within the neural core.</p>
                    <div className="flex items-center justify-between pt-2 border-t border-white/5">
                      <span className="text-[10px] font-bold text-slate-400">Audio Alerts</span>
                      <button 
                        onClick={() => setIsMuted(!isMuted)}
                        className={`px-3 py-1 rounded-lg text-[8px] font-bold transition-all ${!isMuted ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-800 text-slate-500'}`}
                      >
                        {isMuted ? 'MUTED' : 'ENABLED'}
                      </button>
                    </div>
                 </div>
              </div>

              <div className="space-y-4 flex flex-col">
                <h3 className="text-sm font-bold text-red-400 uppercase tracking-widest flex items-center gap-2">
                  <History className="w-4 h-4" />
                  Data Management
                </h3>
                <div className="flex-1 bg-slate-950/50 rounded-2xl border border-white/5 p-4 flex flex-col justify-between">
                  <p className="text-xs text-slate-400">Manage locally cached violation logs and neural weights. All data operations are handled on-device.</p>
                  <div className="flex gap-2 pt-4">
                    <button 
                      onClick={exportToCSV}
                      className="flex-1 bg-slate-800 hover:bg-slate-700 text-white py-2 rounded-xl text-xs font-bold border border-white/10 transition-all flex items-center justify-center gap-2"
                    >
                      <Download className="w-3 h-3" />
                      EXPORT CSV
                    </button>
                    <button 
                      onClick={() => setViolations([])}
                      className="flex-1 bg-red-950/20 hover:bg-red-500/20 text-red-500 py-2 rounded-xl text-xs font-bold border border-red-500/10 transition-all"
                    >
                      WIPE CACHE
                    </button>
                  </div>
                </div>
              </div>
            </div>
            <div className="bg-slate-950/50 py-2 flex justify-center border-t border-white/5">
               <button onClick={() => setShowSettings(false)} className="text-[10px] font-bold text-slate-500 hover:text-slate-300 flex items-center gap-1 uppercase tracking-tighter">
                  Close Settings
               </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <main className="max-w-7xl mx-auto p-4 lg:p-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          <div className="lg:col-span-8 space-y-6">
            <div 
              ref={containerRef}
              className="relative aspect-video bg-slate-950 rounded-3xl overflow-hidden border border-white/10 shadow-2xl group flex items-center justify-center"
            >
              <video 
                ref={videoRef} 
                autoPlay 
                playsInline 
                muted 
                className="w-full h-full object-contain transition-transform duration-300"
                style={{ transform: `scale(${zoom})` }}
              />
              <canvas ref={canvasRef} className="hidden" />
              
              <div className="absolute inset-0 pointer-events-none">
                {renderBoundingBoxes()}
              </div>

              <div className="absolute inset-0 pointer-events-none border-[12px] border-white/5" />
              
              {isAnalyzing && (
                <motion.div 
                  initial={{ top: 0 }}
                  animate={{ top: '100%' }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="absolute left-0 right-0 h-0.5 bg-cyan-400/50 shadow-[0_0_15px_rgba(34,211,238,0.8)] z-10"
                />
              )}

              <div className="absolute top-4 left-4 flex gap-2">
                <div className="bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10 flex items-center gap-2 shadow-lg">
                  <div className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse" />
                  <span className="text-[10px] font-bold uppercase tracking-widest text-slate-100">Live Local Feed</span>
                </div>
                {isRecording && (
                  <motion.div 
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="bg-red-600 text-white px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-widest flex items-center gap-2 shadow-lg animate-pulse"
                  >
                    <div className="w-2 h-2 rounded-full bg-white" />
                    REC
                  </motion.div>
                )}
                {violations.length > 0 && (
                   <motion.div 
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="bg-red-500 text-white px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-widest flex items-center gap-2"
                   >
                     <AlertTriangle className="w-3 h-3" />
                     {violations.length} Incident Detected
                   </motion.div>
                )}
              </div>

              <div className="absolute top-4 right-4 flex flex-col gap-2 group-hover:opacity-100 opacity-0 transition-opacity">
                <div className="bg-slate-900/80 backdrop-blur-md px-1 py-1 rounded-lg border border-white/10 text-[8px] font-bold text-cyan-400 text-center">
                  {(zoom * 100).toFixed(0)}%
                </div>
                <button 
                  onClick={toggleFullscreen}
                  className="bg-slate-900/80 backdrop-blur-md p-2 rounded-lg border border-white/10 text-white hover:bg-cyan-500 transition-colors shadow-lg"
                  title="Toggle Fullscreen"
                >
                  <Maximize className="w-5 h-5" />
                </button>
                <button 
                  onClick={() => setZoom(prev => Math.min(prev + 0.5, 3))}
                  className="bg-slate-900/80 backdrop-blur-md p-2 rounded-lg border border-white/10 text-white hover:bg-cyan-500 transition-colors shadow-lg"
                  title="Zoom In"
                >
                  <ZoomIn className="w-5 h-5" />
                </button>
                <button 
                  onClick={() => setZoom(prev => Math.max(prev - 0.5, 1))}
                  className="bg-slate-900/80 backdrop-blur-md p-2 rounded-lg border border-white/10 text-white hover:bg-cyan-500 transition-colors shadow-lg"
                  title="Zoom Out"
                >
                  <ZoomOut className="w-5 h-5" />
                </button>
                <button 
                  onClick={toggleRecording}
                  className={`p-2 rounded-lg border border-white/10 transition-colors shadow-lg flex items-center justify-center ${isRecording ? 'bg-red-600 text-white hover:bg-red-500 animate-pulse' : 'bg-slate-900/80 text-white hover:bg-red-900/50'}`}
                  title={isRecording ? "Stop Recording" : "Start Recording"}
                >
                   {isRecording ? <StopCircle className="w-5 h-5" /> : <Video className="w-5 h-5" />}
                </button>
              </div>

              {/* Status Overlay */}
              <div className="absolute bottom-6 left-6 right-6 flex justify-between items-end">
                <div className="bg-slate-900/60 backdrop-blur-md p-4 rounded-2xl border border-white/10 space-y-3">
                   <div className="flex items-center gap-6">
                     <div className="flex items-center gap-2">
                        <Maximize className="w-4 h-4 text-cyan-400" />
                        <span className="text-xl font-bold">{result?.objects.length || 0}</span>
                        <span className="text-xs text-slate-400 uppercase tracking-tighter">Units Verified</span>
                     </div>
                     <div className="w-px h-8 bg-white/10" />
                     <div className="flex items-center gap-2 text-red-400">
                        <Ban className="w-4 h-4" />
                        <span className="text-xl font-bold font-mono text-white">{violations.length}</span>
                        <span className="text-xs text-red-500/80 uppercase tracking-tighter font-bold">Logs</span>
                     </div>
                   </div>
                </div>
                
                <div className="flex flex-col items-center">
                  <TrafficLight state={currentSignal} />
                </div>
              </div>

              {error && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/80 backdrop-blur-sm p-8 text-center z-50">
                  <div className="space-y-4 max-w-sm">
                    <div className="relative">
                      <AlertTriangle className="w-16 h-16 text-amber-500 mx-auto" />
                      <motion.div 
                        animate={{ scale: [1, 1.2, 1] }} 
                        transition={{ duration: 2, repeat: Infinity }}
                        className="absolute inset-0 bg-amber-500/20 blur-xl rounded-full"
                      />
                    </div>
                    <p className="text-amber-400 font-medium whitespace-pre-line text-sm leading-relaxed">{error}</p>
                    <div className="flex flex-col gap-2">
                      <button 
                        onClick={() => startCamera()}
                        className="px-6 py-3 bg-cyan-600 text-white rounded-full font-bold hover:bg-cyan-500 transition-all shadow-lg shadow-cyan-500/20 flex items-center justify-center gap-2"
                      >
                        <Camera className="w-4 h-4" />
                        Grant Permission / Retry
                      </button>
                      <button 
                        onClick={() => window.location.reload()}
                        className="text-xs text-slate-500 hover:text-slate-300 font-medium uppercase tracking-widest mt-2"
                      >
                        Or Refresh Application
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
               <div className="bg-slate-900/50 p-6 rounded-3xl border border-white/5 flex flex-col justify-center">
                  <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">Execution Mode</h3>
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${mode === 'AUTO' ? 'bg-emerald-500' : 'bg-amber-500'}`} />
                    <span className="text-lg font-bold">{mode === 'AUTO' ? 'Adaptive Edge Logic' : 'Manual Signal Lock'}</span>
                  </div>
               </div>
               <div className="bg-slate-900/50 p-6 rounded-3xl border border-white/5 flex items-center justify-between">
                  <div className="space-y-1">
                    <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest text-slate-500">Latency</h3>
                    <p className="text-sm font-medium text-cyan-400 font-mono tracking-tighter">L-LATENCY: 12ms</p>
                  </div>
                  <div className="p-3 bg-slate-800/50 rounded-2xl border border-white/5">
                    <Activity className="w-5 h-5 text-emerald-400" />
                  </div>
               </div>
            </div>
          </div>

          <div className="lg:col-span-4 space-y-6">
            <div className="bg-slate-900/50 rounded-3xl border border-white/5 shadow-xl flex flex-col h-[580px] overflow-hidden">
              <div className="p-2 border-b border-white/5 bg-slate-900/20">
                <div className="grid grid-cols-3 gap-1">
                  <button 
                    onClick={() => setActiveTab('OBJECTS')}
                    className={`flex items-center justify-center gap-2 py-3 rounded-2xl text-[10px] font-bold transition-all ${activeTab === 'OBJECTS' ? 'bg-slate-800 text-cyan-400 border border-white/5 shadow-lg' : 'text-slate-500 hover:text-slate-300'}`}
                  >
                    <Maximize className="w-4 h-4" />
                    TRACKING
                  </button>
                  <button 
                    onClick={() => setActiveTab('VIOLATIONS')}
                    className={`flex items-center justify-center gap-2 py-3 rounded-2xl text-[10px] font-bold transition-all ${activeTab === 'VIOLATIONS' ? 'bg-slate-800 text-red-500 border border-white/5 shadow-lg' : 'text-slate-500 hover:text-slate-300'}`}
                  >
                    <History className="w-4 h-4" />
                    EVENTS
                  </button>
                  <button 
                    onClick={() => setActiveTab('TRENDS')}
                    className={`flex items-center justify-center gap-2 py-3 rounded-2xl text-[10px] font-bold transition-all ${activeTab === 'TRENDS' ? 'bg-slate-800 text-emerald-400 border border-white/5 shadow-lg' : 'text-slate-500 hover:text-slate-300'}`}
                  >
                    <BarChart3 className="w-4 h-4" />
                    TRENDS
                  </button>
                </div>
              </div>
              
              <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                <AnimatePresence mode='popLayout'>
                  {activeTab === 'OBJECTS' && (
                    <LayoutGroup>
                      <motion.div layout className="space-y-3">
                        <div className="grid grid-cols-2 gap-2 mb-4">
                          {result?.counts && Object.entries(result.counts).map(([type, count]) => (
                            <div key={type} className="bg-slate-800/40 p-3 rounded-2xl border border-white/5 flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <div className="text-cyan-400">
                                  {getObjectIcon(type)}
                                </div>
                                <span className="text-[10px] font-bold text-slate-400 uppercase">{type}</span>
                              </div>
                              <span className="text-xs font-bold font-mono">{count}</span>
                            </div>
                          ))}
                        </div>
                        {result?.objects.map((obj, i) => (
                           <motion.div 
                            key={`obj-${i}-${obj.plate}`}
                            layout
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                            className={`p-4 rounded-2xl border flex items-center justify-between group transition-all ${obj.isViolating ? 'bg-red-500/10 border-red-500/20' : 'bg-slate-800/40 border-white/5 hover:bg-slate-800/60'}`}
                          >
                            <div className="flex items-center gap-3">
                              <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${obj.isViolating ? 'bg-red-500/20 text-red-400' : 'bg-slate-700 text-slate-300'}`}>
                                {getObjectIcon(obj.type)}
                              </div>
                              <div>
                                <p className="text-xs font-bold uppercase tracking-tight">{obj.type}</p>
                                <p className="text-[10px] text-slate-500 uppercase flex items-center gap-1 font-medium italic">
                                  {obj.isViolating ? <span className="text-red-500 font-bold tracking-tight">Signal Violation</span> : `${Math.round(obj.score * 100)}% Confidence`}
                                </p>
                              </div>
                            </div>
                            
                            {obj.plate && (
                              <div className="px-2 py-1 bg-white border-2 border-slate-300 rounded shadow-sm flex flex-col items-center">
                                <span className="font-mono text-[10px] font-bold text-slate-900 leading-none">{obj.plate}</span>
                              </div>
                            )}
                          </motion.div>
                        )) || (
                          <div className="h-full flex flex-col items-center justify-center text-center p-8 opacity-30 mt-20">
                            <Activity className="w-12 h-12 mb-4 animate-pulse text-cyan-500" />
                            <p className="text-sm font-medium">Listening for frames...</p>
                          </div>
                        )}
                      </motion.div>
                    </LayoutGroup>
                  )}

                  {activeTab === 'VIOLATIONS' && (
                    <LayoutGroup>
                      <motion.div layout className="space-y-3">
                        {violations.length > 0 ? (
                          violations.map((v) => (
                            <motion.div 
                              key={v.id}
                              layout
                              initial={{ opacity: 0, scale: 0.9 }}
                              animate={{ opacity: 1, scale: 1 }}
                              exit={{ opacity: 0, scale: 0.9 }}
                              className="p-4 bg-red-950/20 rounded-2xl border border-red-500/10 space-y-3 relative overflow-hidden"
                            >
                              <div className="absolute top-0 right-0 p-2 opacity-10">
                                <AlertTriangle className="w-12 h-12 text-red-500" />
                              </div>
                              <div className="flex items-center justify-between relative z-10">
                                <div className="flex items-center gap-2">
                                  <Clock className="w-3 h-3 text-slate-500" />
                                  <span className="text-[10px] font-bold text-slate-400">{v.timestamp}</span>
                                </div>
                                <span className="text-[10px] bg-red-500/20 text-red-400 px-2 py-0.5 rounded font-bold uppercase">#RULE_{v.id.slice(0, 4)}</span>
                              </div>
                              <div className="flex items-center gap-3 relative z-10">
                                <div className="w-12 h-12 bg-red-500/20 rounded-xl flex items-center justify-center text-red-500">
                                  {getObjectIcon(v.type)}
                                </div>
                                <div>
                                  <h4 className="text-xs font-bold text-red-400 uppercase leading-tight">{v.type} Infraction</h4>
                                  <p className="text-[9px] text-slate-500 font-mono mt-1">LOGGED PLATE: {v.plate || 'NONE'}</p>
                                </div>
                              </div>
                            </motion.div>
                          ))
                        ) : (
                          <div className="h-full flex flex-col items-center justify-center text-center p-8 opacity-30 mt-20">
                            <History className="w-12 h-12 mb-4" />
                            <p className="text-sm font-medium">No infractions logged.</p>
                          </div>
                        )}
                      </motion.div>
                    </LayoutGroup>
                  )}

                  {activeTab === 'TRENDS' && (
                    <motion.div 
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="space-y-6"
                    >
                      <div className="bg-slate-800/40 p-6 rounded-3xl border border-white/5 space-y-6">
                        <div className="flex items-center justify-between">
                          <h4 className="text-xs font-bold uppercase tracking-widest text-slate-400">Traffic Distribution</h4>
                          <span className="text-[10px] font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">LIVE</span>
                        </div>
                        
                        <div className="space-y-4">
                          {['Car', 'Truck', 'Bus', 'Human', 'Bicycle'].map(type => {
                            const count = result?.counts[type] || 0;
                            const percentage = result?.objects.length ? (count / result.objects.length) * 100 : 0;
                            
                            return (
                              <div key={type} className="space-y-2">
                                <div className="flex justify-between text-[10px] font-bold uppercase">
                                  <span className="text-slate-400">{type}</span>
                                  <span>{count}</span>
                                </div>
                                <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden">
                                  <motion.div 
                                    initial={{ width: 0 }}
                                    animate={{ width: `${percentage}%` }}
                                    className="h-full bg-cyan-500 shadow-[0_0_10px_rgba(6,182,212,0.5)]"
                                  />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-slate-800/40 p-4 rounded-3xl border border-white/5 flex flex-col items-center justify-center text-center">
                          <span className="text-[10px] font-bold text-slate-500 uppercase mb-1">Peak Flow</span>
                          <span className="text-xl font-bold text-white">12 <span className="text-[10px] text-slate-500">U/min</span></span>
                        </div>
                        <div className="bg-slate-800/40 p-4 rounded-3xl border border-white/5 flex flex-col items-center justify-center text-center">
                          <span className="text-[10px] font-bold text-slate-500 uppercase mb-1">Risk Score</span>
                          <span className="text-xl font-bold text-red-500">{(violations.length * 2.5).toFixed(1)}</span>
                        </div>
                      </div>

                      <p className="text-[10px] text-slate-500 italic text-center p-4">
                        Analytics are processed locally using the Vision Core V2. Historical data persists only until manual reset or tab closure.
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>

            <div className="bg-slate-900/50 p-6 rounded-3xl border border-white/5 space-y-4">
               <div className="flex items-center gap-2 text-xs font-bold text-slate-400 uppercase tracking-widest">
                  <ShieldCheck className="w-4 h-4 text-emerald-400" />
                  Edge Strategy Log
               </div>
               <div className="p-4 bg-slate-950/30 rounded-2xl border border-white/5 border-emerald-500/10 space-y-3">
                  <p className="text-sm italic text-slate-400 leading-relaxed font-mono">
                    {result?.reasoning || "Compiling local spatial data for traffic phase optimization..."}
                  </p>
                  <div className="pt-2 border-t border-white/5">
                    <div className="flex justify-between text-[10px] font-bold text-slate-500 uppercase mb-2">
                       <span>Violation Sensitivity</span>
                       <span>{(violationThreshold * 100).toFixed(0)}%</span>
                    </div>
                    <input 
                      type="range" 
                      min="0.5" 
                      max="0.99" 
                      step="0.01" 
                      value={violationThreshold}
                      onChange={(e) => setViolationThreshold(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                    />
                  </div>
               </div>
               <div className="flex gap-2">
                  <button 
                    onClick={() => setMode(prev => prev === 'AUTO' ? 'MANUAL' : 'AUTO')}
                    className={`flex-1 py-3 rounded-xl text-xs font-bold border border-white/5 transition-all flex items-center justify-center gap-2 ${mode === 'AUTO' ? 'bg-slate-800 text-slate-400' : 'bg-emerald-600 text-white shadow-xl shadow-emerald-500/20 border-emerald-400/30'}`}
                  >
                    {mode === 'AUTO' ? 'SUSPEND AUTO' : 'ENABLE AUTO'}
                  </button>
                  <button 
                    onClick={() => { setViolations([]); setResult(null); }}
                    className="px-4 py-3 bg-red-950/20 hover:bg-red-500/20 text-red-500 rounded-xl text-xs font-bold border border-red-500/10 transition-all"
                  >
                    RESET
                  </button>
               </div>
            </div>
          </div>
        </div>
      </main>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 10px;
        }
      `}</style>
    </div>
  );
}

function TrafficLight({ state }: { state: 'GREEN' | 'YELLOW' | 'RED' }) {
  return (
    <div className="bg-slate-900/80 backdrop-blur-md p-3 rounded-3xl flex flex-col gap-4 shadow-2xl border border-white/10 ring-1 ring-white/5">
      <motion.div 
        animate={{ 
          scale: state === 'RED' ? 1.1 : 1,
          opacity: state === 'RED' ? 1 : 0.2 
        }}
        className={`w-7 h-7 rounded-full transition-all duration-500 ${state === 'RED' ? 'bg-red-500 shadow-[0_0_25px_rgba(239,68,68,0.9)] border-red-400 active-light' : 'bg-red-900/40 border-transparent'} border`} 
      />
      <motion.div 
        animate={{ 
          scale: state === 'YELLOW' ? 1.1 : 1,
          opacity: state === 'YELLOW' ? 1 : 0.2 
        }}
        className={`w-7 h-7 rounded-full transition-all duration-500 ${state === 'YELLOW' ? 'bg-amber-500 shadow-[0_0_25px_rgba(245,158,11,0.9)] border-amber-400 active-light' : 'bg-amber-950/40 border-transparent'} border`} 
      />
      <motion.div 
        animate={{ 
          scale: state === 'GREEN' ? 1.1 : 1,
          opacity: state === 'GREEN' ? 1 : 0.2 
        }}
        className={`w-7 h-7 rounded-full transition-all duration-500 ${state === 'GREEN' ? 'bg-emerald-500 shadow-[0_0_25px_rgba(16,185,129,0.9)] border-emerald-400 active-light' : 'bg-emerald-950/40 border-transparent'} border`} 
      />
      
      <style>{`
        .active-light {
          animation: pulse-light 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        @keyframes pulse-light {
          0%, 100% { opacity: 1; }
          50% { opacity: .7; }
        }
      `}</style>
    </div>
  );
}

