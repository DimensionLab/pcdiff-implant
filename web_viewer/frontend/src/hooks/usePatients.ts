/**
 * React Query hooks for patient management.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { patientApi } from '../services/patient-api';
import type { PatientCreate, PatientUpdate } from '../types/patient';

export function usePatients(params?: { search?: string }) {
  return useQuery({
    queryKey: ['patients', params],
    queryFn: () => patientApi.list(params),
  });
}

export function usePatient(patientId: string | null) {
  return useQuery({
    queryKey: ['patient', patientId],
    queryFn: () => patientApi.get(patientId!),
    enabled: !!patientId,
  });
}

export function usePatientProjects(patientId: string | null) {
  return useQuery({
    queryKey: ['patient-projects', patientId],
    queryFn: () => patientApi.getProjects(patientId!),
    enabled: !!patientId,
  });
}

export function useCreatePatient() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: PatientCreate) => patientApi.create(body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['patients'] }),
  });
}

export function useUpdatePatient() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: PatientUpdate }) =>
      patientApi.update(id, body),
    onSuccess: (_, { id }) => {
      qc.invalidateQueries({ queryKey: ['patients'] });
      qc.invalidateQueries({ queryKey: ['patient', id] });
    },
  });
}

export function useDeletePatient() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => patientApi.delete(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['patients'] }),
  });
}
