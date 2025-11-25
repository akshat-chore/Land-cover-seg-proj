import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { AnalysisMode } from '../types';
import { uploadImage } from '../services/api';
import MetricsWrapped from '../components/MetricsWrapped';
import Dashboard from '../components/Dashboard';
import { useAnalysis } from '../context/AnalysisContext';
import { UploadCloud, Loader2, ArrowLeft, AlertTriangle } from 'lucide-react';

const AnalysisPage: React.FC = () => {
  const { mode: paramMode } = useParams<{ mode: string }>();
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const { deficitData, setDeficitData, floodData, setFloodData } = useAnalysis();

  // Validate mode
  const mode = (paramMode === 'flood' ? 'flood' : 'deficit') as AnalysisMode;

  // Determine current data based on mode
  const currentData = mode === 'flood' ? floodData : deficitData;

  const [viewState, setViewState] = useState<'upload' | 'processing' | 'wrapped' | 'dashboard'>('upload');
  const [error, setError] = useState<string | null>(null);
  
  // NEW STATE: To manually trigger the wrapped view from the dashboard
  const [showWrapped, setShowWrapped] = useState(false);

  // If we already have data in context when entering the page, skip to dashboard
  useEffect(() => {
    // Check if the current mode already has data
    if (currentData && viewState === 'upload') {
        setViewState('dashboard');
    }
  }, [currentData, viewState]);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setViewState('processing');
    setError(null);

    try {
      // Simulate min loading time for UX
      const start = Date.now();
      const response = await uploadImage(file); // This response contains BOTH analyses
      const duration = Date.now() - start;
      const minDuration = 2500;
      
      if (duration < minDuration) {
        await new Promise(r => setTimeout(r, minDuration - duration));
      }

      // 🛑 FIXED: Set data for BOTH modes upon a single successful upload
      setDeficitData(response); 
      setFloodData(response); 

      setViewState('wrapped'); // Go to Story Mode
      // Reset showWrapped state on fresh upload
      setShowWrapped(false); 
    } catch (err: any) {
      setError(err.message || 'Failed to process image');
      setViewState('upload');
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        const file = e.dataTransfer.files[0];
        const syntheticEvent = { target: { files: [file] } } as any;
        handleFileSelect(syntheticEvent);
    }
  };

  const handleWrappedComplete = () => {
    // Resetting showWrapped state on completion to return to dashboard flow
    setShowWrapped(false); 
    setViewState('dashboard');
  };

  const reset = () => {
    // 🛑 FIXED: Clear both states to ensure consistency when uploading a new image.
    setDeficitData(null); 
    setFloodData(null); 
    setViewState('upload');
    setError(null);
    setShowWrapped(false); 
  };
  
  // Callback function to expose to Dashboard for re-triggering wrapped
  const reViewWrapped = useCallback(() => {
    setShowWrapped(true);
  }, []);

  // --- 🛑 CRITICAL FIX: Reorder Rendering Logic ---
  
  // 1. Story Wrapped View (Takes precedence when re-viewing via showWrapped or on initial load)
  // This must be checked first to enable re-viewing from the dashboard.
  if (currentData && (viewState === 'wrapped' || showWrapped)) { 
    return <MetricsWrapped data={currentData} mode={mode} onComplete={handleWrappedComplete} />;
  }
  
  // 2. Dashboard View
  if (viewState === 'dashboard' && currentData) {
    // Pass the new reViewWrapped function to Dashboard
    return <Dashboard data={currentData} mode={mode} onReset={reset} onReViewWrapped={reViewWrapped} />;
  }
  
  // --- END CRITICAL FIX ---


  // 3. Upload / Processing View
  return (
    <div className="min-h-screen bg-mntn-dark flex flex-col items-center justify-center p-6 relative">
      
      {/* Background Decor */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-0 left-0 w-full h-1/2 bg-gradient-to-b from-blue-900/10 to-transparent"></div>
      </div>

      <button 
        onClick={() => navigate('/')} 
        className="absolute top-8 left-8 flex items-center gap-2 text-gray-500 hover:text-white transition-colors z-20"
      >
        <ArrowLeft className="w-5 h-5" /> Back Home
      </button>

      <div className="max-w-xl w-full text-center z-10">
        
        <h1 className="font-serif text-3xl md:text-4xl text-white mb-2">
            {mode === 'flood' ? 'Flood Simulation' : 'Deficit Analysis'}
        </h1>
        <p className="text-gray-400 mb-12">Upload satellite imagery (GeoTIFF, PNG, JPG)</p>

        {viewState === 'processing' ? (
            <div className="bg-mntn-card border border-gray-800 rounded-2xl p-12 flex flex-col items-center justify-center h-80 shadow-2xl">
                <Loader2 className="w-12 h-12 text-mntn-accent animate-spin mb-6" />
                <h3 className="text-white font-medium text-lg">Processing Imagery</h3>
                <p className="text-gray-500 text-sm mt-2">Running computer vision models...</p>
                <div className="w-64 h-1 bg-gray-800 mt-6 rounded-full overflow-hidden">
                    <div className="h-full bg-mntn-accent animate-pulse w-2/3"></div>
                </div>
            </div>
        ) : (
            <div 
                className="group bg-mntn-card border-2 border-dashed border-gray-700 hover:border-mntn-accent hover:bg-white/5 transition-all duration-300 rounded-2xl p-12 flex flex-col items-center justify-center h-80 cursor-pointer relative shadow-xl"
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
            >
                <div className="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform group-hover:bg-mntn-accent/20">
                    <UploadCloud className="w-8 h-8 text-mntn-accent" />
                </div>
                <h3 className="text-white font-medium text-lg">Click to Upload</h3>
                <p className="text-gray-500 text-sm mt-2">or drag and drop your file here</p>
                <input 
                    type="file" 
                    ref={fileInputRef} 
                    onChange={handleFileSelect} 
                    className="hidden" 
                    accept="image/*,.tif,.tiff" 
                />
            </div>
        )}

        {error && (
            <div className="mt-6 p-4 bg-red-900/20 border border-red-500/50 text-red-400 rounded-lg text-sm flex items-center justify-center gap-2">
                <AlertTriangle className="w-4 h-4" /> {error}
            </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisPage;