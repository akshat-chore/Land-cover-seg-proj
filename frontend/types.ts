export interface RiskTile {
  tile_id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  lat: number;
  lon: number;
  risk_score: number;
  risk_level: 'low' | 'medium' | 'high';
  building_density_pct?: number;
  road_coverage_pct?: number;
  green_space_pct?: number;
  deficit_score?: number;
  priority_level?: string;
  building_ratio?: number;
}

export interface ClassStatistics {
  total_pixels: number;
  per_class_pixels: Record<string, number>;
  per_class_percentages: Record<string, number>;
  per_class_area_km2: Record<string, number>;
}

export interface FloodMetrics {
  impact_ratio?: number;
  total_buildings?: number;
  flooded_buildings?: number;
  status?: string;
}

export interface EquitySummary {
  mean_deficit_score?: number;
  total_tiles?: number;
  urgent_tiles?: number;
  num_urgent?: number;
  important_tiles?: number;
  adequate_tiles?: number;
}

export interface RiskSummary {
  mean_risk?: number;
  high_risk_tile_pct?: number;
  buildings_near_water_pct?: number;
}

export interface ResponsePlanItem {
  tile_id: string;
  priority: string;
  risk_level: string;
  recommended_window: string;
  recommended_action: string;
  lat: number;
  lon: number;
}

export interface AnalysisResponse {
  success: boolean;
  inference_time_ms: number;
  image_shape: [number, number, number];
  mask_shape: [number, number];
  unique_classes: string[];
  mask_base64: string;
  overlay_base64: string;
  flood_gif_base64?: string;
  
  // Geospatial
  has_geo: boolean;
  geo_bounds?: [number, number, number, number]; // SW_lat, SW_lon, NE_lat, NE_lon
  geo_center?: [number, number];
  
  // Data Sections
  risk_tiles?: RiskTile[];
  class_statistics: ClassStatistics;
  risk_summary?: RiskSummary;
  risk_flags?: string[];
  flood_metrics?: FloodMetrics;
  equity_summary?: EquitySummary;
  equity_tiles?: RiskTile[];
  response_plan?: ResponsePlanItem[];
  
  // Timestamp for reporting
  timestamp?: string;
}

export interface ReportResponse {
  success: boolean;
  report: {
    executive_summary?: string;
    urban_planning?: string | string[];
    disaster_management?: string | string[];
    automation_accuracy?: string | string[];
    recommendations?: {
      model_improvements?: string[];
      deployment_notes?: string[];
    };
  };
  error?: string;
}

export type AnalysisMode = 'deficit' | 'flood';