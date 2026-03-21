/**
 * React Query hooks for case report management.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { caseReportApi } from '../services/case-report-api';
import type { CaseReportCreate } from '../types/case-report';

export function useCaseReports(projectId?: string | null) {
  return useQuery({
    queryKey: ['case-reports', projectId],
    queryFn: () => caseReportApi.list({ project_id: projectId ?? undefined }),
    enabled: projectId !== undefined,
  });
}

export function useCaseReport(reportId: string | null) {
  return useQuery({
    queryKey: ['case-report', reportId],
    queryFn: () => caseReportApi.get(reportId!),
    enabled: !!reportId,
  });
}

export function useGenerateCaseReport() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ body, model }: { body: CaseReportCreate; model?: string }) =>
      caseReportApi.generate(body, model),
    onSuccess: (report) => {
      qc.invalidateQueries({ queryKey: ['case-reports'] });
      qc.invalidateQueries({ queryKey: ['case-reports', report.project_id] });
    },
  });
}

export function useDeleteCaseReport() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => caseReportApi.delete(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['case-reports'] }),
  });
}
