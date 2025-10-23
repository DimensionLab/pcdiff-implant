import { useState } from 'react';

interface ControlPanelProps {
  showInput: boolean;
  showSample: boolean;
  onToggleInput: () => void;
  onToggleSample: () => void;
  onShowBoth: () => void;
  onResetCamera: () => void;
  numSamples: number;
  currentSampleIndex: number;
  onSampleChange: (index: number) => void;
  inputPoints: number;
  samplePoints: number;
}

export const ControlPanel = ({
  showInput,
  showSample,
  onToggleInput,
  onToggleSample,
  onShowBoth,
  onResetCamera,
  numSamples,
  currentSampleIndex,
  onSampleChange,
  inputPoints,
  samplePoints,
}: ControlPanelProps) => {
  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Controls</h3>
      
      {/* Point Cloud Info */}
      <div style={styles.section}>
        <div style={styles.infoRow}>
          <span style={styles.label}>Input Points:</span>
          <span style={styles.value}>{inputPoints.toLocaleString()}</span>
        </div>
        <div style={styles.infoRow}>
          <span style={styles.label}>Sample Points:</span>
          <span style={styles.value}>{samplePoints.toLocaleString()}</span>
        </div>
      </div>

      {/* Visibility Controls */}
      <div style={styles.section}>
        <h4 style={styles.sectionTitle}>Visibility</h4>
        <div style={styles.buttonGroup}>
          <button
            style={{
              ...styles.button,
              ...(showInput ? styles.buttonActive : {}),
            }}
            onClick={onToggleInput}
          >
            {showInput ? '✓' : '○'} Input (Defective)
          </button>
          <button
            style={{
              ...styles.button,
              ...(showSample ? styles.buttonActive : {}),
            }}
            onClick={onToggleSample}
          >
            {showSample ? '✓' : '○'} Sample (Implant)
          </button>
          <button style={styles.button} onClick={onShowBoth}>
            Show Both
          </button>
        </div>
      </div>

      {/* Sample Selection (if ensemble) */}
      {numSamples > 1 && (
        <div style={styles.section}>
          <h4 style={styles.sectionTitle}>
            Sample Selection ({currentSampleIndex + 1}/{numSamples})
          </h4>
          <div style={styles.sampleControls}>
            <button
              style={styles.smallButton}
              onClick={() => onSampleChange(Math.max(0, currentSampleIndex - 1))}
              disabled={currentSampleIndex === 0}
            >
              ← Prev
            </button>
            <select
              style={styles.select}
              value={currentSampleIndex}
              onChange={(e) => onSampleChange(Number(e.target.value))}
            >
              {Array.from({ length: numSamples }, (_, i) => (
                <option key={i} value={i}>
                  Sample {i + 1}
                </option>
              ))}
            </select>
            <button
              style={styles.smallButton}
              onClick={() => onSampleChange(Math.min(numSamples - 1, currentSampleIndex + 1))}
              disabled={currentSampleIndex === numSamples - 1}
            >
              Next →
            </button>
          </div>
        </div>
      )}

      {/* Camera Controls */}
      <div style={styles.section}>
        <h4 style={styles.sectionTitle}>Camera</h4>
        <button style={styles.button} onClick={onResetCamera}>
          Reset Camera
        </button>
      </div>

      {/* Help */}
      <div style={styles.section}>
        <h4 style={styles.sectionTitle}>Controls</h4>
        <div style={styles.help}>
          <div>🖱️ Left Click: Rotate</div>
          <div>🖱️ Right Click: Pan</div>
          <div>🖱️ Scroll: Zoom</div>
        </div>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '20px',
    backgroundColor: '#2a2a2a',
    borderRadius: '8px',
    color: '#ffffff',
  },
  title: {
    margin: '0 0 20px 0',
    fontSize: '18px',
    fontWeight: '600',
  },
  section: {
    marginBottom: '20px',
    paddingBottom: '20px',
    borderBottom: '1px solid #444444',
  },
  sectionTitle: {
    margin: '0 0 10px 0',
    fontSize: '14px',
    fontWeight: '500',
    color: '#cccccc',
  },
  infoRow: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '8px',
    fontSize: '13px',
  },
  label: {
    color: '#999999',
  },
  value: {
    color: '#ffffff',
    fontWeight: '500',
  },
  buttonGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  button: {
    padding: '10px 16px',
    backgroundColor: '#333333',
    color: '#ffffff',
    border: '2px solid #444444',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '500',
    transition: 'all 0.2s',
  },
  buttonActive: {
    backgroundColor: '#4CAF50',
    borderColor: '#4CAF50',
  },
  sampleControls: {
    display: 'flex',
    gap: '8px',
    alignItems: 'center',
  },
  smallButton: {
    padding: '8px 12px',
    backgroundColor: '#333333',
    color: '#ffffff',
    border: '1px solid #444444',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '12px',
    flex: '0 0 auto',
  },
  select: {
    flex: 1,
    padding: '8px',
    backgroundColor: '#333333',
    color: '#ffffff',
    border: '1px solid #444444',
    borderRadius: '4px',
    fontSize: '13px',
  },
  help: {
    fontSize: '12px',
    color: '#999999',
    lineHeight: '1.8',
  },
};

export default ControlPanel;

