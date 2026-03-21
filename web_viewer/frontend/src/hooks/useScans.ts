import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { scanApi } from '../services/scan-api';
import type { ScanCreate, SkullBreakImportRequest } from '../types/scan';

export function useScans(params?: {
  project_id?: string;
  scan_category?: string;
  skull_id?: string;
}) {
  return useQuery({
    queryKey: ['scans', params],
    queryFn: () => scanApi.list(params),
  });
}

export function useScan(scanId: string | null) {
  return useQuery({
    queryKey: ['scan', scanId],
    queryFn: () => scanApi.get(scanId!),
    enabled: !!scanId,
  });
}

export function useScanMetadata(scanId: string | null) {
  return useQuery({
    queryKey: ['scan-metadata', scanId],
    queryFn: () => scanApi.getMetadata(scanId!),
    enabled: !!scanId,
  });
}

export function useScanPointClouds(scanId: string | null) {
  return useQuery({
    queryKey: ['scan-point-clouds', scanId],
    queryFn: () => scanApi.getPointClouds(scanId!),
    enabled: !!scanId,
  });
}

export function useCreateScan() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ScanCreate) => scanApi.create(body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['scans'] }),
  });
}

export function useDeleteScan() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (scanId: string) => scanApi.delete(scanId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['scans'] }),
  });
}

export function useImportSkullBreak() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: SkullBreakImportRequest) => scanApi.importSkullBreak(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['scans'] });
      qc.invalidateQueries({ queryKey: ['point-clouds'] });
    },
  });
}
