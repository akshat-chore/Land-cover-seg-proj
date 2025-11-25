import React, { createContext, useContext, useState, ReactNode } from 'react';
import { AnalysisResponse, AnalysisMode } from '../types';

interface AnalysisState {
  deficitData: AnalysisResponse | null;
  floodData: AnalysisResponse | null;
  setDeficitData: (data: AnalysisResponse | null) => void;
  setFloodData: (data: AnalysisResponse | null) => void;
}

const AnalysisContext = createContext<AnalysisState | undefined>(undefined);

export const AnalysisProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [deficitData, setDeficitData] = useState<AnalysisResponse | null>(null);
  const [floodData, setFloodData] = useState<AnalysisResponse | null>(null);

  return (
    <AnalysisContext.Provider value={{ deficitData, floodData, setDeficitData, setFloodData }}>
      {children}
    </AnalysisContext.Provider>
  );
};

export const useAnalysis = () => {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
};