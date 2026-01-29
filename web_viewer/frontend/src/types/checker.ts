import type { PointCloud } from './point-cloud';

/** A single layer in the multi-point-cloud viewer. */
export interface CheckerLayer {
  id: string;
  pointCloudId: string;
  name: string;
  visible: boolean;
  color: string;
  opacity: number;
  category: string | null;
  useHeatmap: boolean;
  /** 'points' for point cloud layers, 'mesh' for STL mesh layers */
  layerType: 'points' | 'mesh';
  /** For mesh layers: the source point cloud id that generated this mesh */
  sourcePointCloudId?: string;
}

/** Backend response for STL generation status. */
export interface STLStatusResponse {
  has_stl: boolean;
  stl_pc_id: string | null;
  stl_metadata: Record<string, unknown> | null;
  source_pc_id: string;
}

/** Cached fit metrics result from the backend. */
export interface FitMetricsResult {
  id: string;
  implant_pc_id: string;
  reference_pc_id: string;
  defective_skull_pc_id: string | null;
  dice_coefficient: number | null;
  hausdorff_distance: number | null;
  hausdorff_distance_95: number | null;
  boundary_dice: number | null;
  resolution: number;
  voxel_spacing: number | null;
  computation_mode: string;
  computation_time_ms: number | null;
  status: string;
  error_message: string | null;
  created_at: string;
  created_by: string;
}

/** Request body for computing fit metrics. */
export interface FitMetricsRequest {
  implant_pc_id: string;
  reference_pc_id: string;
  defective_skull_pc_id?: string;
  resolution?: number;
}

/** Request body for computing SDF heatmap. */
export interface SDFHeatmapRequest {
  query_pc_id: string;
  reference_pc_id: string;
}

/** Auto-matched skull + implant pair. */
export interface SkullImplantPair {
  skull_id: string;
  defective_skull: PointCloud;
  implants: PointCloud[];
}

/** Default layer colors by scan_category. */
export const DEFAULT_LAYER_COLORS: Record<string, string> = {
  complete_skull: '#888888',
  defective_skull: '#aaaaaa',
  implant: '#2563eb',
  generated_implant: '#10b981',
  stl_mesh: '#e879f9',
  other: '#f59e0b',
};
