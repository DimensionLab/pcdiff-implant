/**
 * Audit log panel showing recent actions.
 * Used in the controls sidebar to display entity-level audit trail.
 */
import { useAuditLog } from '../../hooks/useAuditLog';
import type { AuditLogEntry } from '../../types/audit';

interface AuditLogPanelProps {
  entityType?: string;
  entityId?: string;
  limit?: number;
}

export function AuditLogPanel({ limit = 20 }: AuditLogPanelProps) {
  const { data, isLoading } = useAuditLog({ limit });

  if (isLoading) {
    return <div style={styles.loading}>Loading audit log...</div>;
  }

  if (!data || data.items.length === 0) {
    return <div style={styles.empty}>No audit entries</div>;
  }

  return (
    <div style={styles.container}>
      <h4 style={styles.title}>Audit Trail</h4>
      <div style={styles.list}>
        {data.items.map((entry) => (
          <AuditEntry key={entry.id} entry={entry} />
        ))}
      </div>
      {data.total > limit && (
        <div style={styles.more}>
          Showing {data.items.length} of {data.total} entries
        </div>
      )}
    </div>
  );
}

function AuditEntry({ entry }: { entry: AuditLogEntry }) {
  const time = new Date(entry.timestamp).toLocaleString();

  return (
    <div style={styles.entry}>
      <div style={styles.entryHeader}>
        <span style={styles.action}>{entry.action}</span>
        <span style={styles.time}>{time}</span>
      </div>
      {entry.entity_type && (
        <div style={styles.entity}>
          {entry.entity_type}
          {entry.entity_id && (
            <span style={styles.entityId}>
              {entry.entity_id.substring(0, 8)}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  title: {
    margin: '0 0 8px',
    fontSize: '11px',
    fontWeight: 600,
    color: '#aaa',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  loading: {
    fontSize: '11px',
    color: '#666',
    padding: '8px 0',
  },
  empty: {
    fontSize: '11px',
    color: '#555',
    padding: '8px 0',
  },
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
    maxHeight: '300px',
    overflowY: 'auto',
  },
  entry: {
    padding: '6px 8px',
    background: '#111128',
    borderRadius: '3px',
    fontSize: '10px',
  },
  entryHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  action: {
    color: '#ccc',
    fontWeight: 500,
  },
  time: {
    color: '#555',
    fontSize: '9px',
  },
  entity: {
    marginTop: '2px',
    color: '#888',
    fontSize: '9px',
  },
  entityId: {
    marginLeft: '4px',
    color: '#666',
    fontFamily: 'monospace',
  },
  more: {
    fontSize: '10px',
    color: '#555',
    textAlign: 'center',
    padding: '4px',
  },
};
