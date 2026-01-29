/**
 * Displays computed fit metrics with color-coded quality indicators.
 */
import type { FitMetricsResult } from '../../types/checker';

interface FitMetricsDisplayProps {
  result: FitMetricsResult | null;
  isLoading: boolean;
}

export function FitMetricsDisplay({ result, isLoading }: FitMetricsDisplayProps) {
  if (isLoading) {
    return <div style={styles.status}>Computing metrics...</div>;
  }

  if (!result) {
    return (
      <div style={styles.status}>
        Select implant and reference layers, then click "Compute" to see metrics.
      </div>
    );
  }

  if (result.status === 'failed') {
    return (
      <div style={{ ...styles.status, color: '#ef4444' }}>
        Computation failed: {result.error_message}
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <MetricRow
        label="Dice Coefficient"
        value={result.dice_coefficient}
        format={(v) => v.toFixed(4)}
        thresholds={[0.9, 0.7]}
      />
      <MetricRow
        label="Hausdorff Distance"
        value={result.hausdorff_distance}
        format={(v) => `${v.toFixed(2)} mm`}
        thresholds={[2, 5]}
        invertColor
      />
      <MetricRow
        label="HD95"
        value={result.hausdorff_distance_95}
        format={(v) => `${v.toFixed(2)} mm`}
        thresholds={[1, 3]}
        invertColor
      />
      <MetricRow
        label="Border Dice"
        value={result.boundary_dice}
        format={(v) => v.toFixed(4)}
        thresholds={[0.9, 0.7]}
      />

      {result.computation_time_ms != null && (
        <div style={styles.meta}>
          Computed in {result.computation_time_ms}ms at {result.resolution}^3 resolution
        </div>
      )}
    </div>
  );
}

function MetricRow({
  label,
  value,
  format,
  thresholds,
  invertColor = false,
}: {
  label: string;
  value: number | null;
  format: (v: number) => string;
  thresholds: [number, number]; // [good, warn] boundaries
  invertColor?: boolean; // true = lower is better (e.g. Hausdorff)
}) {
  if (value == null) {
    return (
      <div style={styles.metricRow}>
        <span style={styles.metricLabel}>{label}</span>
        <span style={{ ...styles.metricValue, color: '#555' }}>N/A</span>
      </div>
    );
  }

  const [good, warn] = thresholds;
  let color: string;
  if (invertColor) {
    // Lower is better
    color = value <= good ? '#10b981' : value <= warn ? '#f59e0b' : '#ef4444';
  } else {
    // Higher is better
    color = value >= good ? '#10b981' : value >= warn ? '#f59e0b' : '#ef4444';
  }

  return (
    <div style={styles.metricRow}>
      <span style={styles.metricLabel}>{label}</span>
      <span style={{ ...styles.metricValue, color }}>{format(value)}</span>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  status: {
    fontSize: '11px',
    color: '#666',
    padding: '4px 0',
  },
  metricRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '4px 6px',
    borderRadius: '3px',
    background: 'rgba(255,255,255,0.02)',
  },
  metricLabel: {
    fontSize: '11px',
    color: '#aaa',
  },
  metricValue: {
    fontSize: '12px',
    fontWeight: 600,
    fontFamily: 'monospace',
  },
  meta: {
    fontSize: '9px',
    color: '#555',
    marginTop: '4px',
  },
};
