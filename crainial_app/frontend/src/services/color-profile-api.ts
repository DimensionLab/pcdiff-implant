import { apiV1 } from './api-v1';
import type { ColorProfile, ColorProfileCreate } from '../types/color-profile';

export const colorProfileApi = {
  async list(): Promise<ColorProfile[]> {
    const { data } = await apiV1.get<ColorProfile[]>('/color-profiles/');
    return data;
  },

  async get(profileId: string): Promise<ColorProfile> {
    const { data } = await apiV1.get<ColorProfile>(`/color-profiles/${profileId}`);
    return data;
  },

  async getDefault(): Promise<ColorProfile> {
    const { data } = await apiV1.get<ColorProfile>('/color-profiles/default');
    return data;
  },

  async create(body: ColorProfileCreate): Promise<ColorProfile> {
    const { data } = await apiV1.post<ColorProfile>('/color-profiles/', body);
    return data;
  },

  async update(profileId: string, body: Partial<ColorProfileCreate>): Promise<ColorProfile> {
    const { data } = await apiV1.put<ColorProfile>(`/color-profiles/${profileId}`, body);
    return data;
  },

  async delete(profileId: string): Promise<void> {
    await apiV1.delete(`/color-profiles/${profileId}`);
  },
};
