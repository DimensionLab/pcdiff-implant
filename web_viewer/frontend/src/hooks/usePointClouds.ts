import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { pointCloudApi } from '../services/point-cloud-api';
import type { PointCloudCreate, PointCloudUpdate } from '../types/point-cloud';

export function usePointClouds(params?: {
  project_id?: string;
  scan_id?: string;
  scan_category?: string;
  skull_id?: string;
}) {
  return useQuery({
    queryKey: ['point-clouds', params],
    queryFn: () => pointCloudApi.list(params),
  });
}

export function usePointCloud(pcId: string | null) {
  return useQuery({
    queryKey: ['point-cloud', pcId],
    queryFn: () => pointCloudApi.get(pcId!),
    enabled: !!pcId,
  });
}

export function usePointCloudData(pcId: string | null) {
  return useQuery({
    queryKey: ['point-cloud-data', pcId],
    queryFn: () => pointCloudApi.loadData(pcId!),
    enabled: !!pcId,
    staleTime: Infinity, // Raw data doesn't change
  });
}

export function usePointCloudSDF(pcId: string | null) {
  return useQuery({
    queryKey: ['point-cloud-sdf', pcId],
    queryFn: () => pointCloudApi.loadSDF(pcId!),
    enabled: !!pcId,
    staleTime: 5 * 60 * 1000, // Cache SDF for 5 minutes
  });
}

export function useCreatePointCloud() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: PointCloudCreate) => pointCloudApi.create(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['point-clouds'] });
      qc.invalidateQueries({ queryKey: ['project-point-clouds'] });
    },
  });
}

export function useUpdatePointCloud() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: PointCloudUpdate }) =>
      pointCloudApi.update(id, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['point-clouds'] });
      qc.invalidateQueries({ queryKey: ['project-point-clouds'] });
    },
  });
}

export function useDeletePointCloud() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (pcId: string) => pointCloudApi.delete(pcId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['point-clouds'] });
      qc.invalidateQueries({ queryKey: ['project-point-clouds'] });
    },
  });
}
