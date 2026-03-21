import type { ViewerMode } from '../../types/viewer';

interface ViewerToolbarProps {
  mode: ViewerMode;
  onModeChange: (mode: ViewerMode) => void;
  hasScan: boolean;
  hasPointCloud: boolean;
}

export function ViewerToolbar({ mode, onModeChange, hasScan, hasPointCloud }: ViewerToolbarProps) {
  return (
    <div style={styles.toolbar}>
      <div style={styles.modeGroup}>
        <button
          style={{
            ...styles.modeBtn,
            ...(mode === 'volume' ? styles.modeBtnActive : {}),
          }}
          onClick={() => onModeChange('volume')}
          disabled={!hasScan}
          title="Volume rendering (NRRD)"
        >
          Volume
        </button>
        <button
          style={{
            ...styles.modeBtn,
            ...(mode === 'point_cloud' ? styles.modeBtnActive : {}),
          }}
          onClick={() => onModeChange('point_cloud')}
          disabled={!hasPointCloud}
          title="Point cloud rendering (NPY)"
        >
          Points
        </button>
        <button
          style={{
            ...styles.modeBtn,
            ...(mode === 'split' ? styles.modeBtnActive : {}),
          }}
          onClick={() => onModeChange('split')}
          disabled={!hasScan || !hasPointCloud}
          title="Side-by-side view"
        >
          Split
        </button>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  toolbar: {
    display: 'flex',
    alignItems: 'center',
    padding: '4px 8px',
    background: '#111128',
    borderBottom: '1px solid #222',
    minHeight: '36px',
  },
  modeGroup: {
    display: 'flex',
    gap: '2px',
    background: '#1a1a2e',
    borderRadius: '4px',
    padding: '2px',
  },
  modeBtn: {
    padding: '4px 12px',
    fontSize: '11px',
    background: 'transparent',
    color: '#888',
    border: 'none',
    borderRadius: '3px',
    cursor: 'pointer',
  },
  modeBtnActive: {
    background: '#2563eb',
    color: '#fff',
  },
};
