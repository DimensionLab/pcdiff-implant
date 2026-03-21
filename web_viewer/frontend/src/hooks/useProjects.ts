import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { projectApi } from '../services/fit-metrics-api';
import type { ProjectCreate, ProjectUpdate } from '../types/project';

export function useProjects() {
  return useQuery({
    queryKey: ['projects'],
    queryFn: () => projectApi.list(),
  });
}

export function useProject(projectId: string | null) {
  return useQuery({
    queryKey: ['project', projectId],
    queryFn: () => projectApi.get(projectId!),
    enabled: !!projectId,
  });
}

export function useProjectPointClouds(projectId: string | null) {
  return useQuery({
    queryKey: ['project-point-clouds', projectId],
    queryFn: () => projectApi.getPointClouds(projectId!),
    enabled: !!projectId,
  });
}

export function useAutoMatch(projectId: string | null) {
  return useQuery({
    queryKey: ['project-auto-match', projectId],
    queryFn: () => projectApi.getAutoMatch(projectId!),
    enabled: !!projectId,
  });
}

export function useCreateProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ProjectCreate) => projectApi.create(body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['projects'] }),
  });
}

export function useUpdateProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: ProjectUpdate }) =>
      projectApi.update(id, body),
    onSuccess: (_, { id }) => {
      qc.invalidateQueries({ queryKey: ['projects'] });
      qc.invalidateQueries({ queryKey: ['project', id] });
      qc.invalidateQueries({ queryKey: ['patient-projects'] });
    },
  });
}

export function useDeleteProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => projectApi.delete(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['projects'] }),
  });
}
