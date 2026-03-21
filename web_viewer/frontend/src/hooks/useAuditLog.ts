import { useQuery } from '@tanstack/react-query';
import { apiV1 } from '../services/api-v1';
import type { AuditLogList } from '../types/audit';

interface AuditLogParams {
  skip?: number;
  limit?: number;
  action?: string;
  entity_type?: string;
  entity_id?: string;
}

export function useAuditLog(params: AuditLogParams = {}) {
  return useQuery({
    queryKey: ['audit-log', params],
    queryFn: async () => {
      const { data } = await apiV1.get<AuditLogList>('/audit/', { params });
      return data;
    },
  });
}

export function useEntityAuditLog(entityType: string, entityId: string) {
  return useQuery({
    queryKey: ['audit-log', 'entity', entityType, entityId],
    queryFn: async () => {
      const { data } = await apiV1.get<AuditLogList>(
        `/audit/entity/${entityType}/${entityId}`,
      );
      return data;
    },
    enabled: !!entityType && !!entityId,
  });
}
