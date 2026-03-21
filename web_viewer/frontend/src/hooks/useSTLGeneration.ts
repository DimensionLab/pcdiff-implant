import { useQuery, useQueries, useMutation, useQueryClient } from '@tanstack/react-query';
import { pointCloudApi } from '../services/point-cloud-api';

/** Check if an STL mesh already exists for a source point cloud. */
export function useSTLStatus(pcId: string | null) {
  return useQuery({
    queryKey: ['stl-status', pcId],
    queryFn: () => pointCloudApi.getSTLStatus(pcId!),
    enabled: !!pcId,
    staleTime: 30_000,
  });
}

/** Trigger STL mesh generation from a point cloud. */
export function useGenerateSTL() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ pcId, method, depth }: { pcId: string; method?: string; depth?: number }) =>
      pointCloudApi.generateSTL(pcId, method, depth),
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({ queryKey: ['stl-status', variables.pcId] });
      qc.invalidateQueries({ queryKey: ['point-clouds'] });
      qc.invalidateQueries({ queryKey: ['project-point-clouds'] });
    },
  });
}

/** Fetch STL binary data for Three.js rendering. */
export function useSTLData(pcId: string | null) {
  return useQuery({
    queryKey: ['stl-data', pcId],
    queryFn: () => pointCloudApi.loadSTL(pcId!),
    enabled: !!pcId,
    staleTime: Infinity,
  });
}

/** Fetch STL binary data for multiple mesh layers in parallel. */
export function useMultiSTLData(pcIds: string[]) {
  return useQueries({
    queries: pcIds.map((id) => ({
      queryKey: ['stl-data', id],
      queryFn: () => pointCloudApi.loadSTL(id),
      staleTime: Infinity,
      enabled: !!id,
      retry: 5,
      retryDelay: (attempt: number) => Math.min(2000 * 2 ** attempt, 30_000),
    })),
  });
}
