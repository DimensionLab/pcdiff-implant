export interface Project {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
  created_by: string;
}

export interface ProjectCreate {
  name: string;
  description?: string;
}
