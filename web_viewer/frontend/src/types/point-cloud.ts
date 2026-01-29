export interface PointCloud {
  id: string;
  project_id: string | null;
  scan_id: string | null;
  name: string;
  description: string | null;
  file_path: string;
  file_format: string;
  file_size_bytes: number | null;
  num_points: number | null;
  point_dims: number;
  scan_category: string | null;
  defect_type: string | null;
  skull_id: string | null;
  metadata_json: string | null;
  checksum_sha256: string | null;
  created_at: string;
  updated_at: string;
  created_by: string;
}

export interface PointCloudCreate {
  file_path: string;
  name?: string;
  scan_category?: string;
  defect_type?: string;
  skull_id?: string;
  project_id?: string;
  scan_id?: string;
  description?: string;
}

export interface PointCloudUpdate {
  name?: string;
  description?: string;
  scan_category?: string;
  defect_type?: string;
  skull_id?: string;
  project_id?: string;
  scan_id?: string;
}
