import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { colorProfileApi } from '../services/color-profile-api';
import type { ColorProfileCreate } from '../types/color-profile';

export function useColorProfiles() {
  return useQuery({
    queryKey: ['color-profiles'],
    queryFn: () => colorProfileApi.list(),
  });
}

export function useColorProfile(profileId: string | null) {
  return useQuery({
    queryKey: ['color-profile', profileId],
    queryFn: () => colorProfileApi.get(profileId!),
    enabled: !!profileId,
  });
}

export function useDefaultColorProfile() {
  return useQuery({
    queryKey: ['color-profile', 'default'],
    queryFn: () => colorProfileApi.getDefault(),
  });
}

export function useCreateColorProfile() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ColorProfileCreate) => colorProfileApi.create(body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['color-profiles'] }),
  });
}

export function useDeleteColorProfile() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (profileId: string) => colorProfileApi.delete(profileId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['color-profiles'] }),
  });
}
