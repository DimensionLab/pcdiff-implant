import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fitMetricsApi } from '../services/fit-metrics-api';
import type { FitMetricsRequest } from '../types/checker';

export function useComputeFitMetrics() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: FitMetricsRequest) => fitMetricsApi.compute(body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['fit-metrics'] }),
  });
}

export function useFitMetricsResult(resultId: string | null) {
  return useQuery({
    queryKey: ['fit-metrics', resultId],
    queryFn: () => fitMetricsApi.get(resultId!),
    enabled: !!resultId,
  });
}

export function useSDFHeatmap(
  queryPcId: string | null,
  referencePcId: string | null,
) {
  return useQuery({
    queryKey: ['sdf-heatmap', queryPcId, referencePcId],
    queryFn: () => fitMetricsApi.computeSDFHeatmap(queryPcId!, referencePcId!),
    enabled: !!queryPcId && !!referencePcId,
    staleTime: 5 * 60 * 1000,
  });
}
