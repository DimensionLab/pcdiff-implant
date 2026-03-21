/**
 * Layer manager for the Implant Checker.
 *
 * Displays loaded point cloud layers with visibility toggles,
 * color indicators, heatmap toggle, and remove buttons.
 * Mesh layers show a "3D Print" badge and download button.
 */
import type { CheckerLayer } from '../../types/checker';

interface LayerManagerProps {
  layers: CheckerLayer[];
  onToggleVisibility: (layerId: string) => void;
  onSetColor: (layerId: string, color: string) => void;
  onSetHeatmap: (layerId: string, useHeatmap: boolean) => void;
  onRemove: (layerId: string) => void;
  onGenerateSTL?: (pcId: string) => void;
  onDownloadSTL?: (pcId: string) => void;
  generatingSTLForId?: string | null;
}

export function LayerManager({
  layers,
  onToggleVisibility,
  onSetColor,
  onSetHeatmap,
  onRemove,
  onGenerateSTL,
  onDownloadSTL,
  generatingSTLForId,
}: LayerManagerProps) {
  if (layers.length === 0) {
    return (
      <div style={styles.empty}>
        No layers loaded. Add point clouds from the list above.
      </div>
    );
  }

  return (
    <div style={styles.container}>
      {layers.map((layer) => {
        const isMesh = layer.layerType === 'mesh';
        const isGenerating = generatingSTLForId === layer.pointCloudId;

        return (
          <div key={layer.id} style={styles.layerRow}>
            <button
              style={{
                ...styles.visBtn,
                opacity: layer.visible ? 1 : 0.3,
              }}
              onClick={() => onToggleVisibility(layer.id)}
              title={layer.visible ? 'Hide' : 'Show'}
            >
              {layer.visible ? '\u25C9' : '\u25CE'}
            </button>

            <input
              type="color"
              value={layer.color}
              onChange={(e) => onSetColor(layer.id, e.target.value)}
              style={styles.colorInput}
              title="Layer color"
            />

            <div style={styles.layerInfo}>
              <div style={styles.layerName}>{layer.name}</div>
              <div style={{ display: 'flex', gap: '3px', alignItems: 'center' }}>
                {layer.category && (
                  <span style={styles.categoryBadge}>{layer.category}</span>
                )}
                {isMesh && (
                  <span style={styles.meshBadge}>3D Print</span>
                )}
              </div>
            </div>

            {/* Heatmap toggle — only for point cloud layers */}
            {!isMesh && (
              <button
                style={{
                  ...styles.heatmapBtn,
                  ...(layer.useHeatmap ? styles.heatmapBtnActive : {}),
                }}
                onClick={() => onSetHeatmap(layer.id, !layer.useHeatmap)}
                title="Toggle SDF heatmap"
              >
                H
              </button>
            )}

            {/* Generate STL — only for point cloud layers */}
            {!isMesh && onGenerateSTL && (
              <button
                style={styles.stlBtn}
                onClick={() => onGenerateSTL(layer.pointCloudId)}
                disabled={isGenerating}
                title="Generate 3D printable STL mesh"
              >
                {isGenerating ? '...' : '3D'}
              </button>
            )}

            {/* Download STL — only for mesh layers */}
            {isMesh && onDownloadSTL && (
              <button
                style={styles.downloadBtn}
                onClick={() => onDownloadSTL(layer.pointCloudId)}
                title="Download STL file"
              >
                {'\u2913'}
              </button>
            )}

            <button
              style={styles.removeBtn}
              onClick={() => onRemove(layer.id)}
              title="Remove layer"
            >
              {'\u00D7'}
            </button>
          </div>
        );
      })}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
  empty: {
    fontSize: '11px',
    color: '#555',
    padding: '8px 0',
  },
  layerRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    padding: '4px 6px',
    borderRadius: '4px',
    background: 'rgba(255,255,255,0.03)',
  },
  visBtn: {
    background: 'none',
    border: 'none',
    color: '#ccc',
    fontSize: '14px',
    cursor: 'pointer',
    padding: '0 2px',
    flexShrink: 0,
  },
  colorInput: {
    width: '20px',
    height: '20px',
    border: 'none',
    borderRadius: '3px',
    cursor: 'pointer',
    padding: 0,
    flexShrink: 0,
    background: 'none',
  },
  layerInfo: {
    flex: 1,
    minWidth: 0,
    overflow: 'hidden',
  },
  layerName: {
    fontSize: '11px',
    color: '#ccc',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  categoryBadge: {
    fontSize: '9px',
    color: '#888',
    background: 'rgba(255,255,255,0.05)',
    padding: '1px 4px',
    borderRadius: '2px',
  },
  heatmapBtn: {
    background: 'rgba(255,255,255,0.05)',
    border: '1px solid #333',
    color: '#666',
    fontSize: '9px',
    fontWeight: 700,
    borderRadius: '3px',
    cursor: 'pointer',
    padding: '2px 5px',
    flexShrink: 0,
  },
  heatmapBtnActive: {
    background: '#2563eb',
    borderColor: '#2563eb',
    color: '#fff',
  },
  meshBadge: {
    fontSize: '8px',
    color: '#e879f9',
    background: 'rgba(232,121,249,0.12)',
    padding: '1px 4px',
    borderRadius: '2px',
    fontWeight: 700,
    textTransform: 'uppercase',
    letterSpacing: '0.3px',
  },
  stlBtn: {
    background: 'rgba(232,121,249,0.1)',
    border: '1px solid #7c3aed',
    color: '#c084fc',
    fontSize: '9px',
    fontWeight: 700,
    borderRadius: '3px',
    cursor: 'pointer',
    padding: '2px 5px',
    flexShrink: 0,
  },
  downloadBtn: {
    background: 'rgba(16,185,129,0.1)',
    border: '1px solid #10b981',
    color: '#34d399',
    fontSize: '12px',
    borderRadius: '3px',
    cursor: 'pointer',
    padding: '1px 5px',
    flexShrink: 0,
  },
  removeBtn: {
    background: 'none',
    border: 'none',
    color: '#ef4444',
    fontSize: '14px',
    cursor: 'pointer',
    padding: '0 4px',
    flexShrink: 0,
    opacity: 0.6,
  },
};
