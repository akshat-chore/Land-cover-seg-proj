import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AnalysisResponse, AnalysisMode } from '../types';
import { ChevronRight } from 'lucide-react';

interface Props {
  data: AnalysisResponse;
  mode: AnalysisMode;
  onComplete: () => void;
}

const MetricsWrapped: React.FC<Props> = ({ data, mode, onComplete }) => {
  const [step, setStep] = useState(0);
  const [isPaused, setIsPaused] = useState(false);

  // Construct story slides based on mode
  const getSlides = () => {
    const baseSlides = [
      {
        id: 'intro',
        title: "Analysis Complete",
        value: "Let's see what we found.",
        sub: "Your satellite data has been processed.",
        bg: "bg-gradient-to-br from-mntn-dark to-gray-900"
      },
      {
        id: 'classes',
        title: "Landscape Composition",
        value: `${data.unique_classes.length} Unique Classes`,
        sub: `Detected: ${data.unique_classes.join(' • ')}`,
        bg: "bg-gradient-to-br from-gray-900 to-mntn-card"
      }
    ];

    if (mode === 'flood') {
        const impact = data.flood_metrics?.impact_ratio 
            ? `${(data.flood_metrics.impact_ratio * 100).toFixed(1)}%` 
            : 'N/A';
        const buildings = data.flood_metrics?.flooded_buildings || 0;
        
        return [
            ...baseSlides,
            {
                id: 'impact',
                title: "Flood Impact",
                value: impact,
                sub: "of buildings are in the critical path.",
                bg: "bg-gradient-to-br from-blue-900 to-mntn-dark"
            },
            {
                id: 'buildings',
                title: "Structures at Risk",
                value: buildings.toLocaleString(),
                sub: "Buildings identified within flood zones.",
                bg: "bg-gradient-to-br from-blue-950 to-black"
            },
             {
                id: 'summary',
                title: "Risk Status",
                value: data.flood_metrics?.status || "Analysis Ready",
                sub: "Tap to view the full simulation.",
                bg: "bg-black"
            }
        ];
    } else {
        // Deficit Mode
        const score = data.equity_summary?.mean_deficit_score?.toFixed(2) || 'N/A';
        const urgent = data.equity_summary?.urgent_tiles || data.equity_summary?.num_urgent || 0;
        
        return [
            ...baseSlides,
            {
                id: 'score',
                title: "Infrastructure Deficit",
                value: score,
                sub: "Mean Deficit Score (Lower is better)",
                bg: "bg-gradient-to-br from-red-900 to-mntn-dark"
            },
            {
                id: 'urgent',
                title: "Urgent Attention",
                value: `${urgent} Zones`,
                sub: "Require immediate infrastructure intervention.",
                bg: "bg-gradient-to-br from-orange-900 to-gray-900"
            },
             {
                id: 'summary',
                title: "Planning Report",
                value: "Ready",
                sub: "Tap to view the comprehensive map and report.",
                bg: "bg-black"
            }
        ];
    }
  };

  const slides = getSlides();
  const SLIDE_DURATION = 5000; // 5 seconds per slide

  // Auto-advance logic
  useEffect(() => {
    if (isPaused) return;
    
    const timer = setTimeout(() => {
        nextStep();
    }, SLIDE_DURATION);

    return () => clearTimeout(timer);
  }, [step, isPaused]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
        if(e.key === 'ArrowRight' || e.key === 'Space') {
            nextStep();
        }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [step]);

  const nextStep = () => {
    if (step < slides.length - 1) {
      setStep(step + 1);
    } else {
      onComplete();
    }
  };

  const currentSlide = slides[step];

  return (
    <div 
        className={`fixed inset-0 z-50 flex flex-col items-center justify-center text-white ${currentSlide.bg} transition-colors duration-1000 ease-in-out cursor-pointer`}
        onClick={nextStep}
        onMouseDown={() => setIsPaused(true)}
        onMouseUp={() => setIsPaused(false)}
        onTouchStart={() => setIsPaused(true)}
        onTouchEnd={() => setIsPaused(false)}
    >
      {/* Progress Bars */}
      <div className="absolute top-6 left-0 right-0 flex gap-2 px-4 md:px-8 z-20">
        {slides.map((_, idx) => (
            <div key={idx} className="h-1 flex-1 bg-white/20 rounded-full overflow-hidden">
                <motion.div 
                    className="h-full bg-white"
                    initial={{ width: "0%" }}
                    animate={{ width: idx < step ? "100%" : idx === step && !isPaused ? "100%" : "0%" }}
                    transition={idx === step ? { duration: SLIDE_DURATION / 1000, ease: "linear" } : { duration: 0 }}
                />
            </div>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
            key={currentSlide.id}
            initial={{ opacity: 0, y: 40, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -40, scale: 1.05 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="text-center px-6 max-w-4xl relative z-10"
        >
            <h3 className="font-serif text-xl md:text-2xl text-mntn-accent mb-8 uppercase tracking-[0.2em] font-medium opacity-90">
                {currentSlide.title}
            </h3>
            <h1 className="font-sans font-bold text-6xl md:text-8xl lg:text-9xl mb-8 tracking-tighter shadow-black drop-shadow-2xl">
                {currentSlide.value}
            </h1>
            <p className="font-sans text-xl md:text-3xl text-gray-200 font-light max-w-2xl mx-auto leading-relaxed">
                {currentSlide.sub}
            </p>
        </motion.div>
      </AnimatePresence>

      <div className="absolute bottom-12 right-12 animate-pulse">
        <div className="flex items-center gap-2 text-white/50 text-sm uppercase tracking-widest font-medium">
            {step === slides.length - 1 ? 'Open Dashboard' : 'Tap to skip'}
            <ChevronRight className="w-5 h-5" />
        </div>
      </div>
    </div>
  );
};

export default MetricsWrapped;