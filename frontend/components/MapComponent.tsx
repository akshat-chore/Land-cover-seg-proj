import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { RiskTile } from '../types';

// --- SIMULATED DATA STRUCTURES (Must match Dashboard.tsx) ---
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
// --- END SIMULATED DATA STRUCTURES ---


interface MapComponentProps {
  center: [number, number];
  bounds?: [number, number, number, number];
  riskTiles?: RiskTile[];
  floodGif?: string;
  mode: 'deficit' | 'flood';
  // NEW PROPS FOR DYNAMIC DATA AND SELECTION
  floodEvacuationData?: FloodEvacuationTile[];
  deficitTilesReport?: DeficitTileReport[];
  selectedTileLatLon: [number, number] | null; 
}

const MapComponent: React.FC<MapComponentProps> = ({ 
  center, 
  bounds, 
  riskTiles, 
  floodGif, 
  mode,
  floodEvacuationData, 
  deficitTilesReport, 
  selectedTileLatLon 
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markerLayerRef = useRef<L.LayerGroup | null>(null); // To manage all dynamic markers/rectangles
  const [activeLayer, setActiveLayer] = useState<'satellite' | 'street'>('satellite');

  useEffect(() => {
    if (!mapContainerRef.current) return;

    // Initialize Map if not exists
    if (!mapInstanceRef.current) {
      mapInstanceRef.current = L.map(mapContainerRef.current, {
          zoomControl: false, 
          attributionControl: false 
      }).setView(center, 13);
      
      L.control.attribution({ position: 'bottomright' }).addTo(mapInstanceRef.current);
      markerLayerRef.current = L.layerGroup().addTo(mapInstanceRef.current);
    }

    const map = mapInstanceRef.current;
    const markerLayer = markerLayerRef.current!;

    // Layer Logic
    map.eachLayer((layer) => {
      // Remove previous tiles to switch layers cleanly
      if (layer instanceof L.TileLayer) {
        map.removeLayer(layer);
      }
    });

    if (activeLayer === 'satellite') {
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Tiles © Esri',
            maxZoom: 19
        }).addTo(map);
        // Add hybrid labels for better context
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png', {
             attribution: '',
             subdomains: 'abcd',
             maxZoom: 19
        }).addTo(map);
    } else {
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap',
            maxZoom: 19
        }).addTo(map);
    }

    // Cleanup Markers/Overlays before re-adding
    markerLayer.clearLayers(); 
    map.eachLayer((layer) => {
      if (layer instanceof L.ImageOverlay) { 
        map.removeLayer(layer);
      }
    });

    // --- NEW: Add Dynamic Markers based on Dashboard selection/reports ---
    if (mode === 'deficit' && deficitTilesReport) {
        deficitTilesReport.forEach(tile => {
            const isSelected = selectedTileLatLon && tile.lat === selectedTileLatLon[0] && tile.lon === selectedTileLatLon[1];

            // Use a specific color/size for the deficit marker
            const color = '#FFC107'; // Yellow/Gold for infrastructure
            const radius = isSelected ? 12 : 8;
            
            L.circleMarker([tile.lat, tile.lon], {
                radius: radius,
                fillColor: color,
                color: isSelected ? '#3B82F6' : '#fff', // Blue border for selected
                weight: isSelected ? 3 : 1.5,
                opacity: 0.9,
                fillOpacity: 0.7
            }).bindPopup(`
                <div style="font-family: sans-serif; min-width: 180px;">
                    <h3 style="margin:0 0 4px; color:#333; font-weight:bold">${tile.region_name} (${tile.tile_id})</h3>
                    <div style="font-size:12px; color:#666;">
                        <strong>Deficit Score:</strong> ${tile.deficit_score.toFixed(2)}<br/>
                        <strong>Required:</strong> ${tile.required_infrastructure}<br/>
                        <strong>Est. Cost:</strong> $${(tile.estimated_cost_usd).toLocaleString()}
                    </div>
                </div>
            `).addTo(markerLayer);
        });
        
    } else if (mode === 'flood' && floodEvacuationData) {
        floodEvacuationData.forEach(tile => {
            let color = '#3B82F6'; // Blue for Medium
            let radius = 6;
            
            // Visually stunning coloring/sizing based on urgency
            if (tile.flood_onset_min < 15) { 
                color = '#EF4444'; // Red for Urgent
                radius = 12; 
            } else if (tile.flood_onset_min < 30) { 
                color = '#F59E0B'; // Orange for High
                radius = 9;
            }

            L.circleMarker([tile.lat, tile.lon], {
                radius: radius,
                fillColor: color,
                color: '#fff',
                weight: 2,
                opacity: 1.0,
                fillOpacity: 0.85
            }).bindPopup(`
                <div style="font-family: sans-serif; min-width: 150px;">
                    <h3 style="margin:0 0 4px; color:#333; font-weight:bold">Tile ${tile.tile_id}</h3>
                    <div style="font-size:12px; color:#666;">
                        <strong>Evacuation Priority:</strong> ${tile.evacuation_priority}<br/>
                        <strong>Flood Onset:</strong> <span style="color:${color}; font-weight:bold;">${tile.flood_onset_min} minutes</span><br/>
                        <strong>Buildings at Risk:</strong> ${tile.building_count.toLocaleString()}
                    </div>
                </div>
            `).addTo(markerLayer);
        });
    } else {
        // Fallback to original riskTiles (if any)
        if (riskTiles) {
            riskTiles.forEach(tile => {
                if (tile.lat && tile.lon) {
                    let color = '#10B981'; // Low (Green)
                    let radius = 6;
                    
                    if (tile.risk_score > 0.33) { color = '#F59E0B'; radius = 8; } 
                    if (tile.risk_score > 0.66) { color = '#EF4444'; radius = 10; } 

                    if (mode === 'flood' && tile.risk_level === 'high') {
                        color = '#3B82F6'; // Blue for Flood
                    }

                    L.circleMarker([tile.lat, tile.lon], {
                        radius: radius,
                        fillColor: color,
                        color: '#fff',
                        weight: 1.5,
                        opacity: 0.9,
                        fillOpacity: 0.7
                    }).bindPopup(`
                        <div style="font-family: sans-serif; min-width: 150px;">
                            <h3 style="margin:0 0 4px; color:#333; font-weight:bold">Tile ${tile.tile_id}</h3>
                            <div style="font-size:12px; color:#666;">
                                <strong>Score:</strong> ${tile.risk_score.toFixed(3)}<br/>
                                <strong>Type:</strong> ${tile.priority_level || tile.risk_level}<br/>
                                ${tile.deficit_score ? `<strong>Deficit:</strong> ${tile.deficit_score.toFixed(2)}` : ''}
                            </div>
                        </div>
                    `).addTo(markerLayer);
                }
            });
        }
    }
    
    // --- NEW: Handle Selected Tile Centering ---
    if (selectedTileLatLon) {
        const [lat, lon] = selectedTileLatLon;
        // Fly to the selected point
        map.flyTo([lat, lon], map.getZoom() < 14 ? 14 : map.getZoom(), { duration: 1.5 });
        
        // Add a temporary highlight marker (a pulsing effect is created via the inline CSS)
        L.marker([lat, lon], {
             icon: L.divIcon({
                className: 'custom-div-icon',
                html: '<div style="width: 20px; height: 20px; background-color: #3B82F6; border-radius: 50%; border: 3px solid #fff; box-shadow: 0 0 0 5px rgba(59, 130, 246, 0.5); animation: pulse 1.5s infinite;"></div>',
                iconSize: [20, 20],
                iconAnchor: [10, 10]
            })
        }).addTo(markerLayer);

    }


    // Add Flood GIF Overlay
    if (mode === 'flood' && floodGif && bounds) {
       const southWest = L.latLng(bounds[0], bounds[1]);
       const northEast = L.latLng(bounds[2], bounds[3]);
       const latLngBounds = L.latLngBounds(southWest, northEast);
       
       L.imageOverlay(`data:image/gif;base64,${floodGif}`, latLngBounds, {
           opacity: 0.75,
           interactive: false
       }).addTo(map);
       
       // Add bounding box for context
       L.rectangle(latLngBounds, { color: "#3B82F6", weight: 2, fill: false }).addTo(markerLayer);
    } else if (bounds) {
       // Deficit mode bounding box
       const southWest = L.latLng(bounds[0], bounds[1]);
       const northEast = L.latLng(bounds[2], bounds[3]);
       const latLngBounds = L.latLngBounds(southWest, northEast);
       L.rectangle(latLngBounds, { color: "#D4AF37", weight: 2, fill: false }).addTo(markerLayer);
       
       // Only fit bounds if we have them and no specific tile is selected
       if (!selectedTileLatLon) {
            map.fitBounds(latLngBounds, { padding: [50, 50] });
       }
    }

  }, [center, bounds, riskTiles, floodGif, mode, activeLayer, floodEvacuationData, deficitTilesReport, selectedTileLatLon]); // Added new dependencies

  return (
    <div className="relative w-full h-full">
        {/* New CSS style for the pulsing effect (You'll need to add this to your global CSS file for it to work) */}
        <style>{`
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.5); }
                70% { box-shadow: 0 0 0 10px rgba(59, 130, 246, 0); }
                100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
            }
        `}</style>
        <div ref={mapContainerRef} className="w-full h-full z-0" />
        
        {/* Map Controls */}
        <div className="absolute bottom-8 right-8 z-[400] bg-white/90 backdrop-blur rounded-lg shadow-lg border border-gray-200 p-1 flex flex-col gap-1">
            <button 
                onClick={() => setActiveLayer('satellite')}
                className={`px-3 py-1.5 text-xs font-bold rounded ${activeLayer === 'satellite' ? 'bg-gray-800 text-white' : 'text-gray-600 hover:bg-gray-100'}`}
            >
                Satellite
            </button>
            <button 
                onClick={() => setActiveLayer('street')}
                className={`px-3 py-1.5 text-xs font-bold rounded ${activeLayer === 'street' ? 'bg-gray-800 text-white' : 'text-gray-600 hover:bg-gray-100'}`}
            >
                OpenStreetMap
            </button>
        </div>
    </div>
  );
};

export default MapComponent;