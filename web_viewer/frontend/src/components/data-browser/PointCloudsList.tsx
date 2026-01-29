import { usePointClouds } from '../../hooks/usePointClouds';

interface PointCloudsListProps {
  onSelect: (pcId: string) => void;
  selectedId: string | null;
  categoryFilter: string;
}

export function PointCloudsList({ onSelect, selectedId, categoryFilter }: PointCloudsListProps) {
  const { data: pointClouds, isLoading, error } = usePointClouds(
    categoryFilter ? { scan_category: categoryFilter } : undefined
  );

  if (isLoading) return <div style={styles.message}>Loading point clouds...</div>;
  if (error) return <div style={styles.error}>Failed to load point clouds</div>;
  if (!pointClouds || pointClouds.length === 0) {
    return <div style={styles.message}>No point clouds registered.</div>;
  }

  return (
    <div style={styles.list}>
      {pointClouds.map((pc) => (
        <button
          key={pc.id}
          style={{
            ...styles.item,
            ...(selectedId === pc.id ? styles.itemSelected : {}),
          }}
          onClick={() => onSelect(pc.id)}
        >
          <div style={styles.itemName}>{pc.name}</div>
          <div style={styles.itemMeta}>
            {pc.scan_category && (
              <span style={styles.badge}>{pc.scan_category.replace('_', ' ')}</span>
            )}
            {pc.num_points && (
              <span style={styles.points}>
                {pc.num_points.toLocaleString()} pts
              </span>
            )}
            <span style={styles.format}>{pc.file_format.toUpperCase()}</span>
          </div>
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
  points: {
    fontSize: '10px',
    color: '#888',
  },
  format: {
    fontSize: '10px',
    color: '#555',
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
