import { apiV1 } from './api-v1';
import type { Project, ProjectCreate, ProjectUpdate } from '../types/project';

export const projectApi = {
  async list(): Promise<Project[]> {
    const { data } = await apiV1.get<Project[]>('/projects/');
    return data;
  },

  async get(id: string): Promise<Project> {
    const { data } = await apiV1.get<Project>(`/projects/${id}`);
    return data;
  },

  async create(body: ProjectCreate): Promise<Project> {
    const { data } = await apiV1.post<Project>('/projects/', body);
    return data;
  },

  async update(id: string, body: ProjectUpdate): Promise<Project> {
    const { data } = await apiV1.put<Project>(`/projects/${id}`, body);
    return data;
  },

  async delete(id: string): Promise<void> {
    await apiV1.delete(`/projects/${id}`);
  },
};
