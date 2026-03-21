export interface AuditLogEntry {
  id: number;
  timestamp: string;
  action: string;
  entity_type: string | null;
  entity_id: string | null;
  user_id: string;
  details_json: string | null;
  ip_address: string | null;
  session_id: string | null;
  software_version: string;
}

export interface AuditLogList {
  items: AuditLogEntry[];
  total: number;
}
