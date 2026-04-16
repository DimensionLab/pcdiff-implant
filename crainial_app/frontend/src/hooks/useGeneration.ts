/**
 * React Query hooks for cran-2 implant generation jobs.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { generationApi } from '../services/generation-api';
import type { GenerationJobCreate } from '../types/generation';

/** Single job, auto-polling while pending or running. */
export function useGenerationJob(jobId: string | null) {
  return useQuery({
    queryKey: ['generation-job', jobId],
    queryFn: () => generationApi.getJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (data?.status === 'pending' || data?.status === 'running') {
        return 2000;
      }
      return false;
    },
    staleTime: 1000,
  });
}

export function useProjectJobs(projectId: string | null) {
  return useQuery({
    queryKey: ['generation-jobs', projectId],
    queryFn: () => generationApi.listJobs({ project_id: projectId ?? undefined }),
    enabled: !!projectId,
    staleTime: 10_000,
  });
}

export function useAllJobs() {
  return useQuery({
    queryKey: ['generation-jobs'],
    queryFn: () => generationApi.listJobs(),
    staleTime: 10_000,
  });
}

export function useCreateGenerationJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (body: GenerationJobCreate) => generationApi.createJob(body),
    onSuccess: (job) => {
      queryClient.invalidateQueries({ queryKey: ['generation-jobs'] });
      queryClient.invalidateQueries({ queryKey: ['notifications'] });
      queryClient.setQueryData(['generation-job', job.id], job);
    },
  });
}

export function useCancelJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (jobId: string) => generationApi.cancelJob(jobId),
    onSuccess: (_, jobId) => {
      queryClient.invalidateQueries({ queryKey: ['generation-job', jobId] });
      queryClient.invalidateQueries({ queryKey: ['generation-jobs'] });
    },
  });
}
