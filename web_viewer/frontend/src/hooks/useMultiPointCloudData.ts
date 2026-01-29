import { useQueries } from '@tanstack/react-query';
import { pointCloudApi } from '../services/point-cloud-api';

/**
 * Fetches NPY point cloud data for multiple IDs in parallel.
 *
 * Uses React Query's useQueries to manage parallel requests with caching.
 * Each result is cached with staleTime: Infinity since raw data doesn't change.
 */
export function useMultiPointCloudData(pcIds: string[]) {
  return useQueries({
    queries: pcIds.map((id) => ({
      queryKey: ['point-cloud-data', id],
      queryFn: () => pointCloudApi.loadData(id),
      staleTime: Infinity,
      enabled: !!id,
    })),
  });
}
