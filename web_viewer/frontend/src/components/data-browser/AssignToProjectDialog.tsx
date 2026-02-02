import { useState } from 'react';
import { useProjects } from '../../hooks/useProjects';
import { useUpdatePointCloud } from '../../hooks/usePointClouds';

interface AssignToProjectDialogProps {
  pointCloudId: string;
  pointCloudName: string;
  currentProjectId: string | null;
  onClose: () => void;
}

export function AssignToProjectDialog({
  pointCloudId,
  pointCloudName,
  currentProjectId,
  onClose,
}: AssignToProjectDialogProps) {
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(currentProjectId);
  const [error, setError] = useState('');

  const { data: projects = [] } = useProjects();
  const updatePointCloud = useUpdatePointCloud();

  const handleSubmit = async () => {
    setError('');
    try {
      await updatePointCloud.mutateAsync({
        id: pointCloudId,
        body: { project_id: selectedProjectId || undefined },
      });
      onClose();
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Failed to assign to project');
    }
  };

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <h3 style={styles.title}>Assign to Project</h3>
        
        <p style={styles.subtitle}>
          Assign "<strong>{pointCloudName}</strong>" to a project
        </p>

        <div style={styles.field}>
          <label style={styles.label}>Project</label>
          <select
            style={styles.select}
            value={selectedProjectId ?? ''}
            onChange={(e) => setSelectedProjectId(e.target.value || null)}
          >
            <option value="">None (unassigned)</option>
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
            disabled={updatePointCloud.isPending}
          >
            {updatePointCloud.isPending ? 'Assigning...' : 'Assign'}
          </button>
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'rgba(0, 0, 0, 0.7)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  dialog: {
    background: '#1e1e2e',
    borderRadius: '8px',
    padding: '24px',
    width: '380px',
    maxWidth: '90vw',
    border: '1px solid #333',
  },
  title: {
    margin: '0 0 8px 0',
    fontSize: '16px',
    fontWeight: 600,
    color: '#fff',
  },
  subtitle: {
    margin: '0 0 16px 0',
    fontSize: '13px',
    color: '#999',
  },
  field: {
    marginBottom: '16px',
  },
  label: {
    display: 'block',
    marginBottom: '6px',
    fontSize: '12px',
    color: '#888',
  },
  select: {
    width: '100%',
    padding: '8px 12px',
    background: '#2a2a3e',
    border: '1px solid #444',
    borderRadius: '4px',
    color: '#fff',
    fontSize: '13px',
  },
  error: {
    padding: '8px 12px',
    background: 'rgba(229, 62, 62, 0.15)',
    border: '1px solid rgba(229, 62, 62, 0.3)',
    borderRadius: '4px',
    color: '#fc8181',
    fontSize: '12px',
    marginBottom: '16px',
  },
  buttons: {
    display: 'flex',
    gap: '8px',
    justifyContent: 'flex-end',
  },
  cancelBtn: {
    padding: '8px 16px',
    background: 'transparent',
    border: '1px solid #444',
    borderRadius: '4px',
    color: '#999',
    fontSize: '13px',
    cursor: 'pointer',
  },
  submitBtn: {
    padding: '8px 16px',
    background: '#2563eb',
    border: 'none',
    borderRadius: '4px',
    color: '#fff',
    fontSize: '13px',
    cursor: 'pointer',
  },
};
