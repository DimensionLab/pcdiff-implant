export interface InferenceResult {
  id: string;
  name: string;
  has_input: boolean;
  has_sample: boolean;
  num_samples: number;
  converted: boolean;
  web_dir: string | null;
  files: string[];
}

export interface ConvertedFile {
  name: string;
  type: 'ply' | 'stl';
  size: number;
  path: string;
}

export interface FilesResponse {
  files: ConvertedFile[];
  converted: boolean;
}

export interface ConversionStatus {
  result_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  message?: string;
  started_at?: string;
  completed_at?: string;
}

export interface ApiStatus {
  status: string;
  version: string;
  inference_results_dir: string;
  auto_convert: boolean;
}

