import { useState } from 'react';
import { useCreateScan } from '../../hooks/useScans';
import { useCreatePointCloud } from '../../hooks/usePointClouds';
import { useProjects } from '../../hooks/useProjects';
import { FilePicker } from '../common/FilePicker';

interface DataImportDialogProps {
  onClose: () => void;
  defaultProjectId?: string;
}

export function DataImportDialog({ onClose, defaultProjectId }: DataImportDialogProps) {
  const [filePath, setFilePath] = useState('');
  const [name, setName] = useState('');
  const [category, setCategory] = useState('');
  const [projectId, setProjectId] = useState(defaultProjectId || '');
  const [fileType, setFileType] = useState<'scan' | 'point_cloud'>('scan');
  const [error, setError] = useState('');
  const [showFilePicker, setShowFilePicker] = useState(false);

  const createScan = useCreateScan();
  const createPointCloud = useCreatePointCloud();
  const { data: projects = [] } = useProjects();

  const handleSubmit = async () => {
    setError('');
    if (!filePath.trim()) {
      setError('File path is required');
      return;
    }

    try {
      if (fileType === 'scan') {
        await createScan.mutateAsync({
          file_path: filePath.trim(),
          name: name.trim() || undefined,
          scan_category: category || undefined,
          project_id: projectId || undefined,
        });
      } else {
        await createPointCloud.mutateAsync({
          file_path: filePath.trim(),
          name: name.trim() || undefined,
          scan_category: category || undefined,
          project_id: projectId || undefined,
        });
      }
      onClose();
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Registration failed');
    }
  };

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <h3 style={styles.title}>Register Data File</h3>

        <div style={styles.field}>
          <label style={styles.label}>Type</label>
          <select
            style={styles.select}
            value={fileType}
            onChange={(e) => setFileType(e.target.value as 'scan' | 'point_cloud')}
          >
            <option value="scan">Volume (NRRD)</option>
            <option value="point_cloud">Point Cloud (NPY/PLY/STL)</option>
          </select>
        </div>

        <div style={styles.field}>
          <label style={styles.label}>File Path</label>
          <div style={styles.fileInputRow}>
            <input
              style={styles.filePathInput}
              type="text"
              value={filePath}
              onChange={(e) => setFilePath(e.target.value)}
              placeholder="Click Browse to select a file..."
              readOnly
            />
            <button
              style={styles.browseBtn}
              onClick={() => setShowFilePicker(true)}
              type="button"
            >
              Browse...
            </button>
          </div>
        </div>

        <div style={styles.field}>
          <label style={styles.label}>Name (optional)</label>
          <input
            style={styles.input}
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Display name"
          />
        </div>

        <div style={styles.field}>
          <label style={styles.label}>Category</label>
          <select
            style={styles.select}
            value={category}
            onChange={(e) => setCategory(e.target.value)}
          >
            <option value="">None</option>
            <option value="complete_skull">Complete Skull</option>
            <option value="defective_skull">Defective Skull</option>
            <option value="implant">Implant</option>
            <option value="other">Other</option>
          </select>
        </div>

        <div style={styles.field}>
          <label style={styles.label}>Project (optional)</label>
          <select
            style={styles.select}
            value={projectId}
            onChange={(e) => setProjectId(e.target.value)}
          >
            <option value="">No project</option>
            {projects.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </div>

        {error && <div style={styles.error}>{error}</div>}

        <div style={styles.buttons}>
          <button style={styles.cancelBtn} onClick={onClose}>
            Cancel
          </button>
          <button
            style={styles.submitBtn}
            onClick={handleSubmit}
            disabled={createScan.isPending || createPointCloud.isPending}
          >
            {createScan.isPending || createPointCloud.isPending
              ? 'Registering...'
              : 'Register'}
          </button>
        </div>
      </div>

      {showFilePicker && (
        <FilePicker
          title="Select Data File"
          onSelect={(path) => {
            setFilePath(path);
            setShowFilePicker(false);
            // Auto-detect name from filename
            if (!name) {
              const fileName = path.split('/').pop() || '';
              const baseName = fileName.replace(/\.[^.]+$/, '');
              setName(baseName);
            }
          }}
          onCancel={() => setShowFilePicker(false)}
        />
      )}
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
    width: '440px',
    maxWidth: '90vw',
  },
  title: {
    margin: '0 0 16px',
    fontSize: '16px',
    color: '#fff',
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
  fileInputRow: {
    display: 'flex',
    gap: '8px',
  },
  filePathInput: {
    flex: 1,
    padding: '8px',
    fontSize: '13px',
    background: '#16213e',
    color: '#e0e0e0',
    border: '1px solid #333',
    borderRadius: '4px',
    boxSizing: 'border-box' as const,
  },
  browseBtn: {
    padding: '8px 16px',
    fontSize: '13px',
    background: '#2a2a4e',
    color: '#e0e0e0',
    border: '1px solid #444',
    borderRadius: '4px',
    cursor: 'pointer',
    whiteSpace: 'nowrap',
  },
  select: {
    width: '100%',
    padding: '8px',
    fontSize: '13px',
    background: '#16213e',
    color: '#e0e0e0',
    border: '1px solid #333',
    borderRadius: '4px',
  },
  error: {
    color: '#e53e3e',
    fontSize: '12px',
    marginBottom: '12px',
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
