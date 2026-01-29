import { useState } from 'react';
import { useImportSkullBreak } from '../../hooks/useScans';
import type { ImportResult } from '../../types/scan';

interface SkullBreakImporterProps {
  onClose: () => void;
}

export function SkullBreakImporter({ onClose }: SkullBreakImporterProps) {
  const [baseDir, setBaseDir] = useState(
    '/Users/michaltakac/projects/dimensionlab/pcdiff-implant/datasets/SkullBreak'
  );
  const [result, setResult] = useState<ImportResult | null>(null);
  const [error, setError] = useState('');

  const importMutation = useImportSkullBreak();

  const handleImport = async () => {
    setError('');
    setResult(null);
    try {
      const res = await importMutation.mutateAsync({
        base_dir: baseDir.trim(),
      });
      setResult(res);
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Import failed');
    }
  };

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <h3 style={styles.title}>Import SkullBreak Dataset</h3>
        <p style={styles.desc}>
          Scans the SkullBreak directory structure and registers all NRRD
          volumes and NPY point clouds into the database.
        </p>

        <div style={styles.field}>
          <label style={styles.label}>SkullBreak Directory</label>
          <input
            style={styles.input}
            type="text"
            value={baseDir}
            onChange={(e) => setBaseDir(e.target.value)}
          />
        </div>

        {error && <div style={styles.error}>{error}</div>}

        {result && (
          <div style={styles.result}>
            <div>Scans created: {result.scans_created}</div>
            <div>Point clouds created: {result.point_clouds_created}</div>
            <div>Skipped (existing): {result.skipped}</div>
            {result.errors.length > 0 && (
              <div style={styles.error}>Errors: {result.errors.length}</div>
            )}
          </div>
        )}

        <div style={styles.buttons}>
          <button style={styles.cancelBtn} onClick={onClose}>
            {result ? 'Close' : 'Cancel'}
          </button>
          {!result && (
            <button
              style={styles.submitBtn}
              onClick={handleImport}
              disabled={importMutation.isPending}
            >
              {importMutation.isPending ? 'Importing...' : 'Import Dataset'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

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
    padding: '24px',
    width: '480px',
    maxWidth: '90vw',
  },
  title: {
    margin: '0 0 8px',
    fontSize: '16px',
    color: '#fff',
  },
  desc: {
    margin: '0 0 16px',
    fontSize: '12px',
    color: '#888',
    lineHeight: 1.5,
  },
  field: {
    marginBottom: '12px',
  },
  label: {
    display: 'block',
    marginBottom: '4px',
    fontSize: '12px',
    color: '#999',
  },
  input: {
    width: '100%',
    padding: '8px',
    fontSize: '13px',
    background: '#16213e',
    color: '#e0e0e0',
    border: '1px solid #333',
    borderRadius: '4px',
    boxSizing: 'border-box' as const,
  },
  error: {
    color: '#e53e3e',
    fontSize: '12px',
    marginTop: '4px',
  },
  result: {
    padding: '12px',
    background: '#16213e',
    borderRadius: '4px',
    fontSize: '13px',
    color: '#a0aec0',
    marginBottom: '12px',
    lineHeight: 1.6,
  },
  buttons: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: '8px',
    marginTop: '16px',
  },
  cancelBtn: {
    padding: '8px 16px',
    fontSize: '13px',
    background: 'transparent',
    color: '#999',
    border: '1px solid #444',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  submitBtn: {
    padding: '8px 16px',
    fontSize: '13px',
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
};
