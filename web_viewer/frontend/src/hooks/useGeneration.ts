/**
 * React Query hooks for implant generation jobs.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { generationApi } from '../services/generation-api';
import type { GenerationJobCreate, RevoxelizeJobCreate } from '../types/generation';

/**
 * Query for a single generation job, with auto-polling when running.
 */
export function useGenerationJob(jobId: string | null) {
  return useQuery({
    queryKey: ['generation-job', jobId],
    queryFn: () => generationApi.getJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const data = query.state.data;
      // Poll every 2 seconds while job is pending or running
      if (data?.status === 'pending' || data?.status === 'running') {
        return 2000;
      }
      return false;
    },
    staleTime: 1000,
  });
}

/**
 * Query for listing generation jobs in a project.
 */
export function useProjectJobs(projectId: string | null) {
  return useQuery({
    queryKey: ['generation-jobs', projectId],
    queryFn: () => generationApi.listJobs({ project_id: projectId ?? undefined }),
    enabled: !!projectId,
    staleTime: 10_000,
  });
}

/**
 * Query for listing all generation jobs.
 */
export function useAllJobs() {
  return useQuery({
    queryKey: ['generation-jobs'],
    queryFn: () => generationApi.listJobs(),
    staleTime: 10_000,
  });
}

/**
 * Mutation to create a new generation job.
 */
export function useCreateGenerationJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (body: GenerationJobCreate) => generationApi.createJob(body),
    onSuccess: (job) => {
      queryClient.invalidateQueries({ queryKey: ['generation-jobs'] });
      queryClient.invalidateQueries({ queryKey: ['notifications'] });
      // Set the job in cache immediately
      queryClient.setQueryData(['generation-job', job.id], job);
    },
  });
}

/**
 * Mutation to cancel a generation job.
 */
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

/**
 * Mutation to select an output from ensemble results.
 */
export function useSelectOutput() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ jobId, outputId }: { jobId: string; outputId: string }) =>
      generationApi.selectOutput(jobId, outputId),
    onSuccess: (job) => {
      queryClient.setQueryData(['generation-job', job.id], job);
      queryClient.invalidateQueries({ queryKey: ['generation-jobs'] });
    },
  });
}

/**
 * Mutation to delete unselected outputs.
 */
export function useDeleteUnselectedOutputs() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (jobId: string) => generationApi.deleteUnselectedOutputs(jobId),
    onSuccess: (_, jobId) => {
      queryClient.invalidateQueries({ queryKey: ['generation-job', jobId] });
      queryClient.invalidateQueries({ queryKey: ['point-clouds'] });
      queryClient.invalidateQueries({ queryKey: ['project-point-clouds'] });
    },
  });
}

/**
 * Mutation to create a re-voxelization job.
 * Re-voxelizes an existing implant point cloud with a different mesh resolution.
 */
export function useCreateRevoxelizationJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (body: RevoxelizeJobCreate) => generationApi.createRevoxelizationJob(body),
    onSuccess: (job) => {
      queryClient.invalidateQueries({ queryKey: ['generation-jobs'] });
      queryClient.invalidateQueries({ queryKey: ['notifications'] });
      // Set the job in cache immediately
      queryClient.setQueryData(['generation-job', job.id], job);
    },
  });
}
