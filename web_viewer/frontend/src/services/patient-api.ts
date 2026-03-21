/**
 * API client for patient endpoints.
 */
import { apiV1 } from './api-v1';
import type { Patient, PatientCreate, PatientUpdate } from '../types/patient';
import type { Project } from '../types/project';

export const patientApi = {
  async list(params?: { search?: string; limit?: number; offset?: number }): Promise<Patient[]> {
    const { data } = await apiV1.get<Patient[]>('/patients/', { params });
    return data;
  },

  async get(id: string): Promise<Patient> {
    const { data } = await apiV1.get<Patient>(`/patients/${id}`);
    return data;
  },

  async create(body: PatientCreate): Promise<Patient> {
    const { data } = await apiV1.post<Patient>('/patients/', body);
    return data;
  },

  async update(id: string, body: PatientUpdate): Promise<Patient> {
    const { data } = await apiV1.put<Patient>(`/patients/${id}`, body);
    return data;
  },

  async delete(id: string): Promise<void> {
    await apiV1.delete(`/patients/${id}`);
  },

  async getProjects(patientId: string): Promise<Project[]> {
    const { data } = await apiV1.get<Project[]>(`/patients/${patientId}/projects`);
    return data;
  },
};
