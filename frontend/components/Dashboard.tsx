import React, { useState, useRef } from 'react';
import { AnalysisResponse, AnalysisMode, ReportResponse, RiskTile } from '../types';
import MapComponent from './MapComponent';
import { generateAIReport } from '../services/api';
// CORRECTED/FINALIZED IMPORTS: All required icons are here, including Locate and RotateCcw.
import { Download, Activity, AlertTriangle, Layers, Droplets, Grid, Map as MapIcon, ChevronLeft, ChevronRight, FileText, RotateCcw, Locate } from 'lucide-react'; 

interface Props {
  data: AnalysisResponse;
  mode: AnalysisMode;
  onReset: () => void;
  onReViewWrapped: () => void; 
}

// --- SIMULATED/DERIVED DATA STRUCTURES ---
interface DeficitTileReport {
    tile_id: string;
    region_name: string;
    deficit_score: number;
    required_infrastructure: string;
    estimated_cost_usd: number;
    lat: number;
    lon: number;
}
interface FloodEvacuationTile {
    tile_id: string;
    building_count: number;
    flood_onset_min: number;
    evacuation_priority: 'Urgent' | 'High' | 'Medium';
    lat: number;
    lon: number;
}
// --- END SIMULATED/DERIVED DATA STRUCTURES ---

// NEW HELPER FUNCTION to determine intervention and cost based on a tile's score
// This logic links the abstract risk score to a concrete, simulated action.
const getDeficitIntervention = (tile: RiskTile, index: number): { required_infrastructure: string, estimated_cost_usd: number } => {
    // Use deficit_score if available, otherwise fallback to general risk_score
    const score = tile.deficit_score || tile.risk_score;
    let infrastructure = 'General Infrastructure Upgrade';
    let cost = 500000;
    
    // Logic to assign intervention based on severity
    if (score > 0.8) {
        infrastructure = 'Water Pipeline Extension (5km)';
        cost = 1200000 + (index * 50000); // Add variance
    } else if (score > 0.6) {
        infrastructure = 'Road Repair & Paving (2km)';
        cost = 450000 + (index * 50000);
    } else if (score > 0.4) {
        infrastructure = 'Power Grid Micro-Upgrade';
        cost = 800000 + (index * 50000);
    }
    
    return { 
        required_infrastructure: infrastructure, 
        estimated_cost_usd: cost 
    };
};


// Helper to draw risk overlay on a small canvas thumbnail (omitted for brevity)
const RiskCanvasThumbnail: React.FC<{ 
    overlayBase64: string; 
    riskTiles?: RiskTile[]; 
    maskShape: [number, number]; 
}> = ({ overlayBase64, riskTiles, maskShape }) => {
    // ... (content remains the same)
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [loaded, setLoaded] = useState(false);

    const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
        const img = e.currentTarget;
        const canvas = canvasRef.current;
        if (!canvas || !riskTiles) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        canvas.width = img.width;
        canvas.height = img.height;
        
        const [maskH, maskW] = maskShape;
        const scaleX = canvas.width / maskW;
        const scaleY = canvas.height / maskH;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        riskTiles.forEach(tile => {
            const x = tile.x * scaleX;
            const y = tile.y * scaleY;
            const w = tile.w * scaleX;
            const h = tile.h * scaleY;
            
            let color = 'rgba(16, 185, 129, 0.4)'; // Green
            if (tile.risk_score > 0.33) color = 'rgba(245, 158, 11, 0.4)'; // Orange
            if (tile.risk_score > 0.66) color = 'rgba(239, 68, 68, 0.4)'; // Red

            ctx.fillStyle = color;
            ctx.fillRect(x, y, w, h);
        });
        setLoaded(true);
    };

    return (
        <div className="relative w-full aspect-video rounded-lg overflow-hidden border border-gray-700 bg-black">
            <img 
                src={overlayBase64} 
                alt="Original" 
                className="absolute inset-0 w-full h-full object-contain z-0" 
                onLoad={handleImageLoad}
            />
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none z-10" />
            <div className="absolute bottom-2 right-2 bg-black/70 px-2 py-1 text-[10px] text-white rounded">
                Detection Overlay
            </div>
        </div>
    );
};


const Dashboard: React.FC<Props> = ({ data, mode, onReset, onReViewWrapped }) => { 
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [loadingReport, setLoadingReport] = useState(false);
  const [showReportModal, setShowReportModal] = useState(false);
  const [selectedTile, setSelectedTile] = useState<[number, number] | null>(null); 
  
  // --- DYNAMIC DATA GENERATION: Filter Top 5 Deficit Regions ---
  
  const derivedDeficitData: DeficitTileReport[] = data.risk_tiles
    ? data.risk_tiles
        .filter(tile => tile.lat && tile.lon && (tile.deficit_score || tile.risk_score)) // Ensure we have coordinates and a score
        .sort((a, b) => (b.deficit_score || b.risk_score || 0) - (a.deficit_score || a.risk_score || 0)) // Sort by highest score first
        .slice(0, 5) // Take the absolute top 5
        .map((tile, index) => {
            const intervention = getDeficitIntervention(tile, index);
            return {
                tile_id: tile.tile_id || `D${index + 1}`,
                region_name: `Sector ${String.fromCharCode(65 + index)}`, // A, B, C, D, E
                deficit_score: tile.deficit_score || tile.risk_score || 0,
                lat: tile.lat!,
                lon: tile.lon!,
                ...intervention, // Connects the derived infrastructure/cost to the tile data
            };
        })
    : [];
  
  // Flood Data (Kept as simulated for now)
  const simulatedFloodData: FloodEvacuationTile[] = [
      { tile_id: 'F01', building_count: 55, flood_onset_min: 10, evacuation_priority: 'Urgent', lat: data.geo_center ? data.geo_center[0] + 0.01 : 52.01, lon: data.geo_center ? data.geo_center[1] + 0.01 : 19.01 },
      { tile_id: 'F02', building_count: 210, flood_onset_min: 25, evacuation_priority: 'High', lat: data.geo_center ? data.geo_center[0] - 0.005 : 51.995, lon: data.geo_center ? data.geo_center[1] + 0.005 : 19.005 },
      { tile_id: 'F03', building_count: 8, flood_onset_min: 50, evacuation_priority: 'Medium', lat: data.geo_center ? data.geo_center[0] + 0.005 : 52.005, lon: data.geo_center ? data.geo_center[1] - 0.005 : 18.995 },
  ];
  // --- END DYNAMIC DATA GENERATION ---

  const handleTileSelect = (lat: number, lon: number) => {
      setSelectedTile([lat, lon]);
      setIsSidebarOpen(false); 
  };

  const handleGenerateReport = async () => {
    try {
      setLoadingReport(true);
      setShowReportModal(true);
      const res = await generateAIReport(data);
      setReport(res);
    } catch (e) {
      console.error(e);
      alert("Failed to generate report");
      setShowReportModal(false);
    } finally {
      setLoadingReport(false);
    }
  };

  const handleDownload = () => {
    const jsonString = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analysis_${mode}_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // --- Dynamic Table Content ---
  const renderResponsePlanTable = () => {
      if (mode === 'deficit') {
          return (
              <div className="bg-black/20 border border-gray-800 rounded-lg overflow-hidden shadow-lg">
                  <table className="w-full text-left text-xs">
                      <thead className="bg-white/5 text-gray-400">
                          <tr>
                              <th className="p-3 font-medium">Region</th>
                              <th className="p-3 font-medium">Score</th>
                              <th className="p-3 font-medium">Cost (USD)</th>
                              <th className="p-3 font-medium text-center">Map</th>
                          </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-800 text-gray-300">
                          {derivedDeficitData.map((tile, i) => (
                              <tr key={tile.tile_id} className={`${selectedTile && selectedTile[0] === tile.lat ? 'bg-blue-900/20' : 'hover:bg-white/5'}`}>
                                  <td className="p-3">
                                      <span className="font-bold text-white block">{tile.region_name}</span>
                                      <span className="text-gray-500 block text-[10px]">{tile.required_infrastructure}</span>
                                  </td>
                                  <td className="p-3 text-red-400 font-bold">{(tile.deficit_score || 0).toFixed(2)}</td>
                                  <td className="p-3 text-right text-yellow-400 font-mono">
                                      {/* Format Costing */}
                                      ${(tile.estimated_cost_usd/1000).toFixed(0)}K
                                  </td>
                                  <td className="p-3 text-center">
                                      <button 
                                          onClick={() => handleTileSelect(tile.lat, tile.lon)}
                                          className="text-blue-400 hover:text-blue-300 transition-colors"
                                          title="View on Map"
                                      >
                                          <Locate className="w-4 h-4 mx-auto" />
                                      </button>
                                  </td>
                              </tr>
                          ))}
                      </tbody>
                  </table>
                  <p className="text-[10px] text-gray-500 p-2 text-right">
                    *Costing data is derived from risk score and is a simulation of the recommended fix.
                  </p>
              </div>
          );
      }
      
      if (mode === 'flood') {
          // Sort by lowest flood onset time (most urgent)
          const sortedFloodData = [...simulatedFloodData].sort((a, b) => a.flood_onset_min - b.flood_onset_min);

          return (
              <div className="bg-black/20 border border-gray-800 rounded-lg overflow-hidden shadow-lg">
                  <table className="w-full text-left text-xs">
                      <thead className="bg-white/5 text-gray-400">
                          <tr>
                              <th className="p-3 font-medium">Tile ID</th>
                              <th className="p-3 font-medium">Evac. Time</th>
                              <th className="p-3 font-medium">Buildings</th>
                              <th className="p-3 font-medium">Priority</th>
                          </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-800 text-gray-300">
                          {sortedFloodData.map((tile, i) => (
                              <tr key={tile.tile_id} className={`hover:bg-white/5 ${i === 0 ? 'bg-red-900/30 font-bold' : ''}`}>
                                  <td className="p-3 font-mono">{tile.tile_id}</td>
                                  <td className={`p-3 font-bold ${tile.flood_onset_min < 15 ? 'text-red-400' : 'text-orange-400'}`}>
                                      {tile.flood_onset_min} min
                                  </td>
                                  <td className="p-3">{tile.building_count.toLocaleString()}</td>
                                  <td className="p-3">
                                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase ${
                                          tile.evacuation_priority === 'Urgent' ? 'bg-red-900/50 text-red-300' 
                                          : tile.evacuation_priority === 'High' ? 'bg-orange-900/50 text-orange-300' 
                                          : 'bg-green-900/50 text-green-300'
                                      }`}>
                                          {tile.evacuation_priority}
                                      </span>
                                  </td>
                              </tr>
                          ))}
                      </tbody>
                  </table>
              </div>
          );
      }
      return null;
  };
  // --- End Dynamic Table Content ---

  return (
    <div className="relative w-full h-screen overflow-hidden bg-gray-900 flex">
      
      {/* 1. MAP VISUALIZATION (Background Layer) */}
      <div className="absolute inset-0 z-0">
        {data.has_geo ? (
             <MapComponent 
                center={data.geo_center || [52.0, 19.0]}
                bounds={data.geo_bounds}
                floodEvacuationData={mode === 'flood' ? simulatedFloodData : undefined}
                // Pass derivedDeficitData to MapComponent
                deficitTilesReport={mode === 'deficit' ? derivedDeficitData : undefined}
                riskTiles={data.risk_tiles}
                floodGif={data.flood_gif_base64}
                mode={mode}
                // Pass selected tile state
                selectedTileLatLon={selectedTile}
            />
        ) : (
            <div className="w-full h-full flex items-center justify-center bg-gray-900 text-gray-400 flex-col">
                <MapIcon className="w-16 h-16 mb-4 opacity-50" />
                <p>Geospatial data not available for this image.</p>
                <div className="w-96 mt-8">
                     <RiskCanvasThumbnail 
                        overlayBase64={data.overlay_base64} 
                        riskTiles={data.risk_tiles} 
                        maskShape={data.mask_shape} 
                    />
                </div>
            </div>
        )}
      </div>

      {/* 2. SIDEBAR (Data Overlay) */}
      <div 
        className={`relative z-10 h-full bg-mntn-card/95 backdrop-blur-md border-r border-gray-800 transition-all duration-500 ease-in-out flex flex-col ${isSidebarOpen ? 'w-full md:w-[480px] translate-x-0' : 'w-0 -translate-x-full opacity-0'}`}
      >
         {/* Sidebar Header */}
         <div className="p-6 border-b border-gray-800 flex justify-between items-center bg-mntn-card">
            <div>
                <h2 className="text-mntn-accent text-xs font-bold uppercase tracking-widest mb-1">
                    {mode === 'flood' ? 'Flood Risk Analysis' : 'Infrastructure Equity'}
                </h2>
                <h1 className="text-xl font-serif text-white font-bold">
                    Dashboard
                </h1>
            </div>
            <button onClick={onReset} className="text-xs text-gray-400 hover:text-white border border-gray-700 px-3 py-1 rounded-full transition-colors">
                New Upload
            </button>
         </div>

         {/* Sidebar Content (Scrollable) */}
         <div className="flex-1 overflow-y-auto custom-scrollbar p-6 space-y-8">
            
            {/* Quick Stats */}
            <div className="grid grid-cols-2 gap-4">
                <div className="bg-white/5 border border-white/10 p-4 rounded-lg">
                    <p className="text-gray-400 text-xs uppercase mb-1">{mode === 'flood' ? 'Impact Ratio' : 'Deficit Score'}</p>
                    <p className="text-2xl font-bold text-white">
                        {mode === 'flood' 
                            ? `${(data.flood_metrics?.impact_ratio || 0).toFixed(1)}%`
                            : data.equity_summary?.mean_deficit_score?.toFixed(2) || 'N/A'
                        }
                    </p>
                </div>
                <div className="bg-white/5 border border-white/10 p-4 rounded-lg">
                     <p className="text-gray-400 text-xs uppercase mb-1">{mode === 'flood' ? 'Flooded Bldgs' : 'Urgent Zones'}</p>
                     <p className={`text-2xl font-bold ${mode === 'flood' ? 'text-blue-400' : 'text-red-400'}`}>
                        {mode === 'flood'
                            ? data.flood_metrics?.flooded_buildings?.toLocaleString() || 0
                            : data.equity_summary?.urgent_tiles || 0
                        }
                     </p>
                </div>
            </div>

            {/* Response Plan Table (MODIFIED) */}
            <div>
                <h3 className="text-white font-serif mb-4 flex items-center gap-2">
                    {mode === 'deficit' 
                        ? <Activity className="w-4 h-4 text-red-400" /> 
                        : <Droplets className="w-4 h-4 text-blue-400" />
                    }
                    {mode === 'deficit' ? 'Top Deficit Regions & Costing' : 'Evacuation Timeline'}
                </h3>
                {renderResponsePlanTable()} 
            </div>

            {/* Metrics Table */}
            <div>
                 <h3 className="text-white font-serif mb-4 flex items-center gap-2">
                    <Grid className="w-4 h-4 text-mntn-accent" /> Data Metrics
                </h3>
                <div className="bg-black/20 border border-gray-800 rounded-lg overflow-hidden">
                    <table className="w-full text-left text-xs">
                        <thead className="bg-white/5 text-gray-400">
                            <tr>
                                <th className="p-3 font-medium">Metric</th>
                                <th className="p-3 font-medium text-right">Value</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-800 text-gray-300">
                            <tr>
                                <td className="p-3">Inference Time</td>
                                <td className="p-3 text-right">{(data.inference_time_ms/1000).toFixed(2)}s</td>
                            </tr>
                            <tr>
                                <td className="p-3">Total Pixels</td>
                                <td className="p-3 text-right">{data.class_statistics.total_pixels.toLocaleString()}</td>
                            </tr>
                            {data.risk_summary && (
                                <tr>
                                    <td className="p-3">Mean Risk</td>
                                    <td className="p-3 text-right">{data.risk_summary.mean_risk?.toFixed(3)}</td>
                                </tr>
                            )}
                            {data.class_statistics.per_class_percentages && Object.entries(data.class_statistics.per_class_percentages).map(([key, val]) => (
                                <tr key={key}>
                                    <td className="p-3 capitalize">{key} Coverage</td>
                                    <td className="p-3 text-right">{val}%</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Actions (Original + Wrapped Button) */}
            <div className="grid grid-cols-3 gap-3 pt-4 border-t border-gray-800">
                <button 
                    onClick={handleDownload}
                    className="flex items-center justify-center gap-2 bg-gray-800 hover:bg-gray-700 text-white py-3 rounded-lg text-sm font-medium transition-colors"
                >
                    <Download className="w-4 h-4" /> JSON
                </button>
                {/* Re-view Wrapped Button */}
                <button 
                    onClick={onReViewWrapped}
                    className="flex items-center justify-center gap-2 bg-blue-900/50 hover:bg-blue-900/80 text-blue-300 py-3 rounded-lg text-sm font-medium transition-colors"
                >
                    <RotateCcw className="w-4 h-4" /> Wrapped
                </button>
                <button 
                    onClick={handleGenerateReport}
                    className="flex items-center justify-center gap-2 bg-mntn-accent hover:bg-white text-mntn-dark py-3 rounded-lg text-sm font-bold transition-colors"
                >
                    <FileText className="w-4 h-4" /> AI Report
                </button>
            </div>

            {/* Segmentation Preview */}
            <div className="pt-6">
                <h4 className="text-gray-500 text-xs uppercase mb-2">Visual Verification</h4>
                <RiskCanvasThumbnail 
                    overlayBase64={data.overlay_base64} 
                    riskTiles={data.risk_tiles} 
                    maskShape={data.mask_shape} 
                />
            </div>

         </div>
      </div>

      {/* 3. SIDEBAR TOGGLE BUTTON */}
      <div className={`absolute top-6 z-20 transition-all duration-500 ${isSidebarOpen ? 'left-[480px] ml-4' : 'left-6'}`}>
        <button 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="bg-mntn-card text-white p-2 rounded-lg shadow-xl border border-gray-700 hover:bg-gray-700 transition-colors"
        >
            {isSidebarOpen ? <ChevronLeft className="w-6 h-6" /> : <ChevronRight className="w-6 h-6" />}
        </button>
      </div>
      
      {/* 4. REPORT MODAL */}
      {showReportModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
              <div className="bg-mntn-card border border-gray-700 w-full max-w-2xl max-h-[80vh] rounded-xl flex flex-col shadow-2xl">
                  <div className="p-6 border-b border-gray-700 flex justify-between items-center">
                      <h3 className="text-xl font-serif text-white">AI Strategy Report</h3>
                      <button onClick={() => setShowReportModal(false)} className="text-gray-400 hover:text-white">✕</button>
                  </div>
                  
                  <div className="flex-1 overflow-y-auto p-6 text-gray-300 space-y-6">
                      {loadingReport ? (
                          <div className="flex flex-col items-center justify-center h-64">
                              <div className="animate-spin w-12 h-12 border-4 border-mntn-accent border-t-transparent rounded-full mb-4"></div>
                              <p>Generating strategic insights...</p>
                          </div>
                      ) : report?.report ? (
                          <>
                             <div className="bg-gray-800/50 p-4 rounded-lg border-l-4 border-mntn-accent">
                                 <h4 className="text-white font-bold mb-2">Executive Summary</h4>
                                 <p className="text-sm leading-relaxed">{report.report.executive_summary}</p>
                             </div>
                             
                             <div>
                                 <h4 className="text-mntn-accent font-serif text-lg mb-2">Urban Planning Recommendations</h4>
                                 <ul className="list-disc list-inside space-y-2 text-sm">
                                    {Array.isArray(report.report.urban_planning) ? report.report.urban_planning.map((item, i) => (
                                        <li key={i}>{item}</li>
                                    )) : <p>{report.report.urban_planning}</p>}
                                 </ul>
                             </div>

                             <div>
                                 <h4 className="text-blue-400 font-serif text-lg mb-2">Disaster Management</h4>
                                 <ul className="list-disc list-inside space-y-2 text-sm">
                                    {Array.isArray(report.report.disaster_management) ? report.report.disaster_management.map((item, i) => (
                                        <li key={i}>{item}</li>
                                    )) : <p>{report.report.disaster_management}</p>}
                                 </ul>
                             </div>
                          </>
                      ) : (
                          <p className="text-red-400">Failed to load report.</p>
                      )}
                  </div>

                  <div className="p-4 border-t border-gray-700 bg-gray-900/50 rounded-b-xl flex justify-end">
                      <button onClick={() => setShowReportModal(false)} className="px-6 py-2 bg-white text-black font-bold rounded-lg hover:bg-gray-200">
                          Close
                      </button>
                  </div>
              </div>
          </div>
      )}

    </div>
  );
};

export default Dashboard;