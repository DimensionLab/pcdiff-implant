/**
 * Right sidebar controls for the Implant Checker.
 *
 * Sections:
 * 1. Fit Metrics — select layers to compare, compute, view results
 * 2. SDF Heatmap — select layers for distance heatmap
 * 3. Display Settings — point size, grid, axes
 */
import { useState } from 'react';
import { FitMetricsDisplay } from './FitMetricsDisplay';
import { useComputeFitMetrics } from '../../hooks/useFitMetrics';
import { ColorProfileSelector } from '../color-profiles/ColorProfileSelector';
import type { CheckerLayer } from '../../types/checker';
import type { FitMetricsResult } from '../../types/checker';

interface CheckerControlsProps {
  layers: CheckerLayer[];
  pointSize: number;
  onPointSizeChange: (size: number) => void;
  showGrid: boolean;
  onShowGridChange: (show: boolean) => void;
  showAxes: boolean;
  onShowAxesChange: (show: boolean) => void;
  // Heatmap
  heatmapLayerId: string | null;
  heatmapReferenceId: string | null;
  onHeatmapLayerChange: (layerId: string | null) => void;
  onHeatmapReferenceChange: (layerId: string | null) => void;
  colorProfileId: string | null;
  onColorProfileChange: (id: string | null) => void;
}

export function CheckerControls({
  layers,
  pointSize,
  onPointSizeChange,
  showGrid,
  onShowGridChange,
  showAxes,
  onShowAxesChange,
  heatmapLayerId,
  heatmapReferenceId,
  onHeatmapLayerChange,
  onHeatmapReferenceChange,
  colorProfileId,
  onColorProfileChange,
}: CheckerControlsProps) {
  const computeMetrics = useComputeFitMetrics();
  const [metricsResult, setMetricsResult] = useState<FitMetricsResult | null>(null);
  const [implantLayerId, setImplantLayerId] = useState<string | null>(null);
  const [referenceLayerId, setReferenceLayerId] = useState<string | null>(null);
  const [skullLayerId, setSkullLayerId] = useState<string | null>(null);

  const handleCompute = () => {
    if (!implantLayerId || !referenceLayerId) return;
    computeMetrics.mutate(
      {
        implant_pc_id: implantLayerId,
        reference_pc_id: referenceLayerId,
        defective_skull_pc_id: skullLayerId ?? undefined,
      },
      {
        onSuccess: (result) => setMetricsResult(result),
      },
    );
  };

  return (
    <div style={styles.container}>
      {/* Fit Metrics */}
      <section style={styles.section}>
        <h4 style={styles.sectionTitle}>Fit Metrics</h4>

        <label style={styles.label}>Implant Layer</label>
        <select
          style={styles.select}
          value={implantLayerId ?? ''}
          onChange={(e) => setImplantLayerId(e.target.value || null)}
        >
          <option value="">Select...</option>
          {layers.map((l) => (
            <option key={l.id} value={l.pointCloudId}>
              {l.name}
            </option>
          ))}
        </select>

        <label style={styles.label}>Reference Layer</label>
        <select
          style={styles.select}
          value={referenceLayerId ?? ''}
          onChange={(e) => setReferenceLayerId(e.target.value || null)}
        >
          <option value="">Select...</option>
          {layers.map((l) => (
            <option key={l.id} value={l.pointCloudId}>
              {l.name}
            </option>
          ))}
        </select>

        <label style={styles.label}>Defective Skull (optional, for BDC)</label>
        <select
          style={styles.select}
          value={skullLayerId ?? ''}
          onChange={(e) => setSkullLayerId(e.target.value || null)}
        >
          <option value="">None</option>
          {layers
            .filter((l) => l.category === 'defective_skull')
            .map((l) => (
              <option key={l.id} value={l.pointCloudId}>
                {l.name}
              </option>
            ))}
        </select>

        <button
          style={{
            ...styles.computeBtn,
            ...((!implantLayerId || !referenceLayerId) ? styles.computeBtnDisabled : {}),
          }}
          onClick={handleCompute}
          disabled={!implantLayerId || !referenceLayerId || computeMetrics.isPending}
        >
          {computeMetrics.isPending ? 'Computing...' : 'Compute Metrics'}
        </button>

        <FitMetricsDisplay
          result={metricsResult}
          isLoading={computeMetrics.isPending}
        />
      </section>

      {/* SDF Heatmap */}
      <section style={styles.section}>
        <h4 style={styles.sectionTitle}>SDF Heatmap</h4>

        <label style={styles.label}>Target Layer</label>
        <select
          style={styles.select}
          value={heatmapLayerId ?? ''}
          onChange={(e) => onHeatmapLayerChange(e.target.value || null)}
        >
          <option value="">None</option>
          {layers.map((l) => (
            <option key={l.id} value={l.id}>
              {l.name}
            </option>
          ))}
        </select>

        <label style={styles.label}>Reference Layer</label>
        <select
          style={styles.select}
          value={heatmapReferenceId ?? ''}
          onChange={(e) => onHeatmapReferenceChange(e.target.value || null)}
        >
          <option value="">None</option>
          {layers.map((l) => (
            <option key={l.id} value={l.id}>
              {l.name}
            </option>
          ))}
        </select>

        <label style={styles.label}>Color Profile</label>
        <ColorProfileSelector
          selectedId={colorProfileId}
          onSelect={onColorProfileChange}
        />
      </section>

      {/* Display Settings */}
      <section style={styles.section}>
        <h4 style={styles.sectionTitle}>Display Settings</h4>

        <label style={styles.label}>Point Size</label>
        <input
          type="range"
          min="0.001"
          max="0.05"
          step="0.001"
          value={pointSize}
          onChange={(e) => onPointSizeChange(parseFloat(e.target.value))}
          style={styles.slider}
        />
        <span style={styles.sliderValue}>{pointSize.toFixed(3)}</span>

        <div style={styles.checkboxRow}>
          <label style={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={showGrid}
              onChange={(e) => onShowGridChange(e.target.checked)}
            />
            Show Grid
          </label>
        </div>

        <div style={styles.checkboxRow}>
          <label style={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={showAxes}
              onChange={(e) => onShowAxesChange(e.target.checked)}
            />
            Show Axes
          </label>
        </div>
      </section>

      {/* Help */}
      <section style={styles.section}>
        <h4 style={styles.sectionTitle}>Navigation</h4>
        <div style={styles.help}>
          <div>Left Click: Rotate</div>
          <div>Right Click: Pan</div>
          <div>Scroll: Zoom</div>
        </div>
      </section>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0',
  },
  section: {
    paddingBottom: '12px',
    marginBottom: '12px',
    borderBottom: '1px solid #222',
  },
  sectionTitle: {
    margin: '0 0 8px',
    fontSize: '11px',
    fontWeight: 600,
    color: '#aaa',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  label: {
    display: 'block',
    fontSize: '11px',
    color: '#888',
    marginBottom: '4px',
    marginTop: '6px',
  },
  select: {
    width: '100%',
    background: '#111',
    color: '#ccc',
    border: '1px solid #333',
    borderRadius: '4px',
    padding: '4px 8px',
    fontSize: '11px',
    marginBottom: '2px',
  },
  computeBtn: {
    width: '100%',
    marginTop: '8px',
    marginBottom: '8px',
    padding: '6px 12px',
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    fontSize: '11px',
    fontWeight: 600,
    cursor: 'pointer',
  },
  computeBtnDisabled: {
    background: '#1a2a3a',
    color: '#555',
    cursor: 'not-allowed',
  },
  slider: {
    width: '100%',
    marginBottom: '2px',
  },
  sliderValue: {
    fontSize: '10px',
    color: '#666',
  },
  checkboxRow: {
    marginTop: '6px',
  },
  checkboxLabel: {
    fontSize: '11px',
    color: '#ccc',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    cursor: 'pointer',
  },
  help: {
    fontSize: '11px',
    color: '#666',
    lineHeight: '1.8',
  },
};
