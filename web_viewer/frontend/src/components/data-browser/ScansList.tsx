import { useScans } from '../../hooks/useScans';

interface ScansListProps {
  onSelect: (scanId: string) => void;
  selectedId: string | null;
  categoryFilter: string;
}

export function ScansList({ onSelect, selectedId, categoryFilter }: ScansListProps) {
  const { data: scans, isLoading, error } = useScans(
    categoryFilter ? { scan_category: categoryFilter } : undefined
  );

  if (isLoading) return <div style={styles.message}>Loading scans...</div>;
  if (error) return <div style={styles.error}>Failed to load scans</div>;
  if (!scans || scans.length === 0) {
    return <div style={styles.message}>No volumes registered. Use Import to add data.</div>;
  }

  return (
    <div style={styles.list}>
      {scans.map((scan) => (
        <button
          key={scan.id}
          style={{
            ...styles.item,
            ...(selectedId === scan.id ? styles.itemSelected : {}),
          }}
          onClick={() => onSelect(scan.id)}
        >
          <div style={styles.itemName}>{scan.name}</div>
          <div style={styles.itemMeta}>
            {scan.scan_category && (
              <span style={styles.badge}>{scan.scan_category.replace('_', ' ')}</span>
            )}
            {scan.volume_dims_x && (
              <span style={styles.dims}>
                {scan.volume_dims_x}x{scan.volume_dims_y}x{scan.volume_dims_z}
              </span>
            )}
          </div>
          {scan.defect_type && (
            <div style={styles.itemDefect}>{scan.defect_type}</div>
          )}
        </button>
      ))}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
    padding: '4px',
  },
  item: {
    display: 'block',
    width: '100%',
    textAlign: 'left',
    padding: '8px 12px',
    background: 'transparent',
    color: '#ccc',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '12px',
  },
  itemSelected: {
    background: '#1e3a5f',
    color: '#fff',
  },
  itemName: {
    fontWeight: 500,
    marginBottom: '2px',
    whiteSpace: 'nowrap' as const,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  itemMeta: {
    display: 'flex',
    gap: '6px',
    alignItems: 'center',
  },
  badge: {
    fontSize: '10px',
    padding: '1px 6px',
    borderRadius: '3px',
    background: '#2d3748',
    color: '#a0aec0',
    textTransform: 'capitalize' as const,
  },
  dims: {
    fontSize: '10px',
    color: '#666',
  },
  itemDefect: {
    fontSize: '10px',
    color: '#888',
    marginTop: '2px',
  },
  message: {
    padding: '24px 16px',
    textAlign: 'center' as const,
    color: '#666',
    fontSize: '12px',
  },
  error: {
    padding: '24px 16px',
    textAlign: 'center' as const,
    color: '#e53e3e',
    fontSize: '12px',
  },
};
