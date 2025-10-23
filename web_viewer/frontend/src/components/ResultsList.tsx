import { InferenceResult } from '../types';

interface ResultsListProps {
  results: InferenceResult[];
  selectedResultId: string | null;
  onSelect: (resultId: string) => void;
  loading: boolean;
}

export const ResultsList = ({ results, selectedResultId, onSelect, loading }: ResultsListProps) => {
  if (loading) {
    return (
      <div style={styles.container}>
        <h3 style={styles.title}>Inference Results</h3>
        <div style={styles.loading}>Loading...</div>
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div style={styles.container}>
        <h3 style={styles.title}>Inference Results</h3>
        <div style={styles.empty}>No results found</div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Inference Results ({results.length})</h3>
      <div style={styles.list}>
        {results.map((result) => (
          <div
            key={result.id}
            style={{
              ...styles.item,
              ...(selectedResultId === result.id ? styles.itemSelected : {}),
            }}
            onClick={() => onSelect(result.id)}
          >
            <div style={styles.itemName}>{result.name}</div>
            <div style={styles.itemInfo}>
              {result.num_samples > 1 && (
                <span style={styles.badge}>{result.num_samples} samples</span>
              )}
              {result.converted ? (
                <span style={{ ...styles.badge, ...styles.badgeSuccess }}>✓ Converted</span>
              ) : (
                <span style={{ ...styles.badge, ...styles.badgeWarning }}>Not converted</span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '20px',
    backgroundColor: '#2a2a2a',
    borderRadius: '8px',
    height: '100%',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
  },
  title: {
    margin: '0 0 15px 0',
    color: '#ffffff',
    fontSize: '18px',
    fontWeight: '600',
  },
  list: {
    flex: 1,
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  item: {
    padding: '12px',
    backgroundColor: '#333333',
    borderRadius: '6px',
    cursor: 'pointer',
    transition: 'all 0.2s',
    border: '2px solid transparent',
  },
  itemSelected: {
    backgroundColor: '#404040',
    borderColor: '#4CAF50',
  },
  itemName: {
    color: '#ffffff',
    fontSize: '14px',
    fontWeight: '500',
    marginBottom: '6px',
  },
  itemInfo: {
    display: 'flex',
    gap: '6px',
    flexWrap: 'wrap',
  },
  badge: {
    padding: '2px 8px',
    fontSize: '11px',
    borderRadius: '4px',
    backgroundColor: '#444444',
    color: '#cccccc',
  },
  badgeSuccess: {
    backgroundColor: '#2e7d32',
    color: '#ffffff',
  },
  badgeWarning: {
    backgroundColor: '#f57c00',
    color: '#ffffff',
  },
  loading: {
    color: '#cccccc',
    textAlign: 'center',
    padding: '20px',
  },
  empty: {
    color: '#cccccc',
    textAlign: 'center',
    padding: '20px',
  },
};

export default ResultsList;

