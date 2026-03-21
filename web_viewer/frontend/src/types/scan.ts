export interface Scan {
  id: string;
  project_id: string | null;
  name: string;
  description: string | null;
  file_path: string;
  file_format: string;
  file_size_bytes: number | null;
  volume_dims_x: number | null;
  volume_dims_y: number | null;
  volume_dims_z: number | null;
  voxel_spacing_x: number | null;
  voxel_spacing_y: number | null;
  voxel_spacing_z: number | null;
  scan_category: string | null;
  defect_type: string | null;
  skull_id: string | null;
  metadata_json: string | null;
  checksum_sha256: string | null;
  created_at: string;
  updated_at: string;
  created_by: string;
}

export interface ScanCreate {
  file_path: string;
  name?: string;
  scan_category?: string;
  defect_type?: string;
  skull_id?: string;
  project_id?: string;
  description?: string;
}

export interface SkullBreakImportRequest {
  base_dir: string;
  project_id?: string;
  compute_checksums?: boolean;
}

export interface ImportResult {
  scans_created: number;
  point_clouds_created: number;
  skipped: number;
  errors: Array<{ file: string; error: string }>;
}

export interface VolumeMetadata {
  id: string;
  name: string;
  file_format: string;
  dims: [number | null, number | null, number | null];
  spacing: [number | null, number | null, number | null];
  scan_category: string | null;
  defect_type: string | null;
  skull_id: string | null;
}
