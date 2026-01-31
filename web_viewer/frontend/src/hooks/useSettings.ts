/**
 * React Query hooks for application settings.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { settingsApi } from '../services/settings-api';
import type { SettingsUpdate } from '../types/settings';

/**
 * Query hook for fetching application settings.
 */
export function useSettings() {
  return useQuery({
    queryKey: ['settings'],
    queryFn: () => settingsApi.getSettings(),
    staleTime: 60_000, // Settings don't change often
  });
}

/**
 * Query hook for fetching system information.
 */
export function useSystemInfo() {
  return useQuery({
    queryKey: ['system-info'],
    queryFn: () => settingsApi.getSystemInfo(),
    staleTime: 300_000, // System info rarely changes
  });
}

/**
 * Mutation hook for updating settings.
 */
export function useUpdateSettings() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (updates: SettingsUpdate) => settingsApi.updateSettings(updates),
    onSuccess: (newSettings) => {
      queryClient.setQueryData(['settings'], newSettings);
    },
  });
}
