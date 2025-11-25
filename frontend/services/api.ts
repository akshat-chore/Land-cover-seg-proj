import { AnalysisResponse, ReportResponse } from '../types';

const API_URL = 'http://127.0.0.1:8000';

export const uploadImage = async (file: File): Promise<AnalysisResponse> => {
  const formData = new FormData();
  formData.append('image', file);

  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    body: formData,
    mode: 'cors',
  });

  if (!response.ok) {
    throw new Error(`API Error: ${response.status} ${response.statusText}`);
  }

  return response.json();
};

export const generateAIReport = async (data: AnalysisResponse): Promise<ReportResponse> => {
  const payload = {
    metrics_json: {
      total_pixels: data.class_statistics.total_pixels,
      per_class_pixels: data.class_statistics.per_class_pixels || {},
      per_class_percentages: data.class_statistics.per_class_percentages || {},
      per_class_area_km2: data.class_statistics.per_class_area_km2 || {}
    },
    segmentation_summary: {
      image_shape: data.image_shape,
      mask_shape: data.mask_shape,
      inference_time_ms: data.inference_time_ms,
      unique_classes: data.unique_classes,
      risk_summary: data.risk_summary || {},
      risk_flags: data.risk_flags || [],
      flood_metrics: data.flood_metrics || {},
      equity_summary: data.equity_summary || {}
    },
    context: {
      analysis_type: 'land_cover_segmentation',
      timestamp: data.timestamp || new Date().toISOString(),
      satellite_image: true
    }
  };

  const response = await fetch(`${API_URL}/report`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload),
    mode: 'cors'
  });

  if (!response.ok) {
    throw new Error(`Report Generation Error: ${response.status}`);
  }

  return response.json();
};