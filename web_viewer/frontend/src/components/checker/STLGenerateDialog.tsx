/**
 * Modal dialog for configuring STL mesh generation settings.
 *
 * Shows detail level presets, reconstruction method selection,
 * and real-time estimates for file size and generation time.
 */
import { useState, useMemo } from 'react';

type DetailLevel = 'low' | 'medium' | 'high' | 'ultra';
type Method = 'poisson' | 'ball_pivoting' | 'convex_hull';

interface STLGenerateDialogProps {
  layerName: string;
  pcId: string;
  onGenerate: (pcId: string, method: string, depth: number) => void;
  onClose: () => void;
  isGenerating: boolean;
}

const DETAIL_PRESETS: Record<DetailLevel, { depth: number; label: string; desc: string }> = {
  low:    { depth: 6, label: 'Low',    desc: 'Fast, coarse surface' },
  medium: { depth: 7, label: 'Medium', desc: 'Balanced detail & speed' },
  high:   { depth: 8, label: 'High',   desc: 'Good detail for printing' },
  ultra:  { depth: 9, label: 'Ultra',  desc: 'Maximum detail, slower' },
};

const METHODS: Record<Method, { label: string; desc: string }> = {
  poisson:        { label: 'Poisson',        desc: 'Watertight surface ideal for 3D printing (recommended)' },
  ball_pivoting:  { label: 'Ball Pivoting',  desc: 'Mesh from local triangulation, may have holes' },
  convex_hull:    { label: 'Convex Hull',    desc: 'Simple convex envelope, fast but loses concave detail' },
};

/**
 * Rough heuristic estimates for Poisson reconstruction.
 * Faces scale ~4x per depth level for typical cranial geometry.
 */
function estimateFaces(depth: number, method: Method): number {
  if (method === 'convex_hull') return 2000;
  if (method === 'ball_pivoting') return 100_000;
  // Poisson: rough estimate based on depth
  const baseFaces = 15_000;
  return Math.round(baseFaces * Math.pow(4, depth - 6));
}

function estimateFileSize(faces: number): number {
  // Binary STL: 80-byte header + 4-byte count + 50 bytes per face
  return 84 + 50 * faces;
}

function estimateTime(depth: number, method: Method): [number, number] {
  if (method === 'convex_hull') return [1, 2];
  if (method === 'ball_pivoting') return [5, 15];
  // Poisson time ranges (seconds) — includes downsampling + normals + reconstruction
  const times: Record<number, [number, number]> = {
    6: [2, 5],
    7: [4, 10],
    8: [8, 20],
    9: [20, 60],
  };
  return times[depth] ?? [10, 30];
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `~${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `~${(n / 1_000).toFixed(0)}K`;
  return `~${n}`;
}

export function STLGenerateDialog({
  layerName,
  pcId,
  onGenerate,
  onClose,
  isGenerating,
}: STLGenerateDialogProps) {
  const [detailLevel, setDetailLevel] = useState<DetailLevel>('high');
  const [method, setMethod] = useState<Method>('poisson');

  const depth = DETAIL_PRESETS[detailLevel].depth;
  const depthApplies = method === 'poisson';

  const estimates = useMemo(() => {
    const faces = estimateFaces(depth, method);
    const size = estimateFileSize(faces);
    const [timeMin, timeMax] = estimateTime(depth, method);
    return { faces, size, timeMin, timeMax };
  }, [depth, method]);

  const handleGenerate = () => {
    onGenerate(pcId, method, depth);
  };

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <h3 style={styles.title}>Generate 3D Print Mesh</h3>

        <div style={styles.sourceRow}>
          <span style={styles.sourceLabel}>Source:</span>
          <span style={styles.sourceName}>{layerName}</span>
        </div>

        {/* Detail Level */}
        <div style={styles.section}>
          <label style={styles.sectionLabel}>Detail Level</label>
          <div style={styles.presetRow}>
            {(Object.keys(DETAIL_PRESETS) as DetailLevel[]).map((key) => {
              const preset = DETAIL_PRESETS[key];
              const active = detailLevel === key;
              const disabled = !depthApplies;
              return (
                <button
                  key={key}
                  style={{
                    ...styles.presetBtn,
                    ...(active ? styles.presetBtnActive : {}),
                    ...(disabled ? styles.presetBtnDisabled : {}),
                  }}
                  onClick={() => setDetailLevel(key)}
                  disabled={disabled}
                  title={preset.desc}
                >
                  {preset.label}
                </button>
              );
            })}
          </div>
          {depthApplies && (
            <div style={styles.depthNote}>
              Octree depth: {depth} — {DETAIL_PRESETS[detailLevel].desc}
            </div>
          )}
          {!depthApplies && (
            <div style={styles.depthNote}>
              Detail level only applies to Poisson reconstruction
            </div>
          )}
        </div>

        {/* Method */}
        <div style={styles.section}>
          <label style={styles.sectionLabel}>Reconstruction Method</label>
          <div style={styles.methodList}>
            {(Object.keys(METHODS) as Method[]).map((key) => {
              const m = METHODS[key];
              const active = method === key;
              return (
                <label key={key} style={styles.methodRow}>
                  <input
                    type="radio"
                    name="method"
                    checked={active}
                    onChange={() => setMethod(key)}
                    style={styles.radio}
                  />
                  <div>
                    <div style={{
                      ...styles.methodName,
                      ...(active ? styles.methodNameActive : {}),
                    }}>
                      {m.label}
                      {key === 'poisson' && (
                        <span style={styles.recommendedBadge}>Recommended</span>
                      )}
                    </div>
                    <div style={styles.methodDesc}>{m.desc}</div>
                  </div>
                </label>
              );
            })}
          </div>
        </div>

        {/* Estimates */}
        <div style={styles.estimatesBox}>
          <div style={styles.estimatesTitle}>Estimates</div>
          <div style={styles.estimatesGrid}>
            <div style={styles.estimateLabel}>Faces:</div>
            <div style={styles.estimateValue}>{formatNumber(estimates.faces)}</div>
            <div style={styles.estimateLabel}>File size:</div>
            <div style={styles.estimateValue}>{formatBytes(estimates.size)}</div>
            <div style={styles.estimateLabel}>Time:</div>
            <div style={styles.estimateValue}>
              ~{estimates.timeMin}-{estimates.timeMax}s
            </div>
          </div>
          <div style={styles.estimatesNote}>
            Estimates based on typical cranial implant geometry. Actual values may vary.
          </div>
        </div>

        {/* Actions */}
        <div style={styles.footer}>
          <button style={styles.cancelBtn} onClick={onClose} disabled={isGenerating}>
            Cancel
          </button>
          <button
            style={{
              ...styles.generateBtn,
              ...(isGenerating ? styles.generateBtnDisabled : {}),
            }}
            onClick={handleGenerate}
            disabled={isGenerating}
          >
            {isGenerating ? 'Generating...' : 'Generate STL'}
          </button>
        </div>
      </div>
    </div>
  );
}

/* ── Styles ───────────────────────────────────────────────── */

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(0,0,0,0.6)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  dialog: {
    background: '#1a1a2e',
    border: '1px solid #333',
    borderRadius: '8px',
    padding: '20px',
    width: '440px',
    maxWidth: '90vw',
    maxHeight: '85vh',
    overflow: 'auto',
    display: 'flex',
    flexDirection: 'column',
  },
  title: {
    margin: '0 0 12px',
    fontSize: '15px',
    color: '#fff',
  },
  sourceRow: {
    display: 'flex',
    gap: '6px',
    alignItems: 'center',
    marginBottom: '16px',
    padding: '8px 10px',
    background: 'rgba(255,255,255,0.03)',
    borderRadius: '4px',
  },
  sourceLabel: {
    fontSize: '11px',
    color: '#888',
    flexShrink: 0,
  },
  sourceName: {
    fontSize: '12px',
    color: '#ccc',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  section: {
    marginBottom: '16px',
  },
  sectionLabel: {
    display: 'block',
    fontSize: '11px',
    fontWeight: 600,
    color: '#aaa',
    textTransform: 'uppercase',
    letterSpacing: '0.4px',
    marginBottom: '8px',
  },
  presetRow: {
    display: 'flex',
    gap: '6px',
  },
  presetBtn: {
    flex: 1,
    padding: '8px 4px',
    fontSize: '12px',
    fontWeight: 600,
    background: 'rgba(255,255,255,0.04)',
    color: '#999',
    border: '1px solid #333',
    borderRadius: '4px',
    cursor: 'pointer',
    transition: 'all 0.15s',
  },
  presetBtnActive: {
    background: 'rgba(124,58,237,0.2)',
    color: '#c084fc',
    borderColor: '#7c3aed',
  },
  presetBtnDisabled: {
    opacity: 0.4,
    cursor: 'default',
  },
  depthNote: {
    fontSize: '10px',
    color: '#666',
    marginTop: '6px',
  },
  methodList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  methodRow: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '8px',
    padding: '8px 10px',
    borderRadius: '4px',
    background: 'rgba(255,255,255,0.02)',
    cursor: 'pointer',
  },
  radio: {
    marginTop: '2px',
    flexShrink: 0,
    accentColor: '#7c3aed',
  },
  methodName: {
    fontSize: '12px',
    color: '#ccc',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  methodNameActive: {
    color: '#fff',
  },
  recommendedBadge: {
    fontSize: '8px',
    color: '#34d399',
    background: 'rgba(16,185,129,0.12)',
    padding: '1px 5px',
    borderRadius: '2px',
    fontWeight: 700,
    textTransform: 'uppercase',
    letterSpacing: '0.3px',
  },
  methodDesc: {
    fontSize: '10px',
    color: '#666',
    marginTop: '2px',
  },
  estimatesBox: {
    background: 'rgba(255,255,255,0.03)',
    border: '1px solid #2a2a40',
    borderRadius: '6px',
    padding: '12px',
    marginBottom: '16px',
  },
  estimatesTitle: {
    fontSize: '11px',
    fontWeight: 600,
    color: '#aaa',
    textTransform: 'uppercase',
    letterSpacing: '0.4px',
    marginBottom: '8px',
  },
  estimatesGrid: {
    display: 'grid',
    gridTemplateColumns: 'auto 1fr',
    gap: '4px 12px',
    marginBottom: '8px',
  },
  estimateLabel: {
    fontSize: '12px',
    color: '#888',
  },
  estimateValue: {
    fontSize: '12px',
    color: '#e0e0ff',
    fontWeight: 600,
    fontVariantNumeric: 'tabular-nums',
  },
  estimatesNote: {
    fontSize: '9px',
    color: '#555',
    fontStyle: 'italic',
  },
  footer: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: '8px',
    paddingTop: '4px',
  },
  cancelBtn: {
    padding: '8px 16px',
    fontSize: '12px',
    background: 'transparent',
    color: '#999',
    border: '1px solid #444',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  generateBtn: {
    padding: '8px 20px',
    fontSize: '12px',
    fontWeight: 600,
    background: '#7c3aed',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  generateBtnDisabled: {
    opacity: 0.6,
    cursor: 'default',
  },
};
