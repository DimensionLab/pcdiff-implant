/**
 * Import dialog for the Implant Checker.
 *
 * Two modes:
 * 1. "Browse" — pick existing point clouds from the database and
 *    assign them to the current workspace.
 * 2. "Register" — register a new file path as a point cloud,
 *    automatically associating it with the current workspace.
 */
import { useState } from 'react';
import { usePointClouds, useCreatePointCloud, useUpdatePointCloud } from '../../hooks/usePointClouds';
import type { PointCloud } from '../../types/point-cloud';

type Tab = 'browse' | 'register';

interface CheckerImportDialogProps {
  projectId: string;
  /** Point cloud IDs already loaded as layers — used to disable "Add" */
  loadedPcIds: Set<string>;
  onAddLayer: (pc: PointCloud) => void;
  onClose: () => void;
}

export function CheckerImportDialog({
  projectId,
  loadedPcIds,
  onAddLayer,
  onClose,
}: CheckerImportDialogProps) {
  const [tab, setTab] = useState<Tab>('browse');

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <h3 style={styles.title}>Add Point Clouds</h3>

        <div style={styles.tabs}>
          <button
            style={{ ...styles.tab, ...(tab === 'browse' ? styles.tabActive : {}) }}
            onClick={() => setTab('browse')}
          >
            Browse Existing
          </button>
          <button
            style={{ ...styles.tab, ...(tab === 'register' ? styles.tabActive : {}) }}
            onClick={() => setTab('register')}
          >
            Register New File
          </button>
        </div>

        {tab === 'browse' ? (
          <BrowseTab
            projectId={projectId}
            loadedPcIds={loadedPcIds}
            onAddLayer={onAddLayer}
          />
        ) : (
          <RegisterTab
            projectId={projectId}
            onAddLayer={onAddLayer}
            onClose={onClose}
          />
        )}

        <div style={styles.footer}>
          <button style={styles.closeBtn} onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

/* ── Browse existing point clouds ─────────────────────────── */

function BrowseTab({
  projectId,
  loadedPcIds,
  onAddLayer,
}: {
  projectId: string;
  loadedPcIds: Set<string>;
  onAddLayer: (pc: PointCloud) => void;
}) {
  const { data: allPointClouds = [], isLoading } = usePointClouds();
  const updatePc = useUpdatePointCloud();
  const [categoryFilter, setCategoryFilter] = useState('');

  const filtered = allPointClouds.filter((pc) => {
    if (categoryFilter && pc.scan_category !== categoryFilter) return false;
    return true;
  });

  const handleAdd = (pc: PointCloud) => {
    // Assign to workspace if not already
    if (pc.project_id !== projectId) {
      updatePc.mutate({ id: pc.id, body: { project_id: projectId } });
    }
    onAddLayer(pc);
  };

  if (isLoading) {
    return <div style={styles.status}>Loading point clouds...</div>;
  }

  return (
    <div style={styles.tabContent}>
      <div style={styles.filterRow}>
        <select
          style={styles.filterSelect}
          value={categoryFilter}
          onChange={(e) => setCategoryFilter(e.target.value)}
        >
          <option value="">All categories</option>
          <option value="complete_skull">Complete Skull</option>
          <option value="defective_skull">Defective Skull</option>
          <option value="implant">Implant</option>
          <option value="generated_implant">Generated Implant</option>
        </select>
      </div>

      {filtered.length === 0 ? (
        <div style={styles.status}>
          No point clouds found. Use the Data Viewer to import data first,
          or use "Register New File" to add a file path.
        </div>
      ) : (
        <div style={styles.list}>
          {filtered.map((pc) => {
            const alreadyLoaded = loadedPcIds.has(pc.id);
            return (
              <div key={pc.id} style={styles.pcRow}>
                <div style={styles.pcInfo}>
                  <div style={styles.pcName}>{pc.name}</div>
                  <div style={styles.pcMeta}>
                    {pc.scan_category && (
                      <span style={styles.badge}>{pc.scan_category}</span>
                    )}
                    {pc.num_points && (
                      <span style={styles.metaText}>
                        {pc.num_points.toLocaleString()} pts
                      </span>
                    )}
                  </div>
                </div>
                <button
                  style={{
                    ...styles.addBtn,
                    ...(alreadyLoaded ? styles.addBtnDisabled : {}),
                  }}
                  onClick={() => handleAdd(pc)}
                  disabled={alreadyLoaded}
                >
                  {alreadyLoaded ? 'Added' : 'Add'}
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ── Register new file ────────────────────────────────────── */

function RegisterTab({
  projectId,
  onAddLayer,
  onClose,
}: {
  projectId: string;
  onAddLayer: (pc: PointCloud) => void;
  onClose: () => void;
}) {
  const [filePath, setFilePath] = useState('');
  const [name, setName] = useState('');
  const [category, setCategory] = useState('');
  const [skullId, setSkullId] = useState('');
  const [error, setError] = useState('');

  const createPc = useCreatePointCloud();

  const handleSubmit = async () => {
    setError('');
    if (!filePath.trim()) {
      setError('File path is required');
      return;
    }

    try {
      const pc = await createPc.mutateAsync({
        file_path: filePath.trim(),
        name: name.trim() || undefined,
        scan_category: category || undefined,
        skull_id: skullId.trim() || undefined,
        project_id: projectId,
      });
      onAddLayer(pc);
      onClose();
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Registration failed');
    }
  };

  return (
    <div style={styles.tabContent}>
      <div style={styles.field}>
        <label style={styles.label}>File Path (absolute)</label>
        <input
          style={styles.input}
          type="text"
          value={filePath}
          onChange={(e) => setFilePath(e.target.value)}
          placeholder="/path/to/file.npy"
        />
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
          <option value="generated_implant">Generated Implant</option>
        </select>
      </div>

      <div style={styles.field}>
        <label style={styles.label}>Skull ID (optional, for auto-match)</label>
        <input
          style={styles.input}
          type="text"
          value={skullId}
          onChange={(e) => setSkullId(e.target.value)}
          placeholder="e.g. skull_001"
        />
      </div>

      {error && <div style={styles.error}>{error}</div>}

      <button
        style={styles.submitBtn}
        onClick={handleSubmit}
        disabled={createPc.isPending}
      >
        {createPc.isPending ? 'Registering...' : 'Register & Add'}
      </button>
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
    width: '500px',
    maxWidth: '90vw',
    maxHeight: '80vh',
    display: 'flex',
    flexDirection: 'column',
  },
  title: {
    margin: '0 0 12px',
    fontSize: '15px',
    color: '#fff',
  },
  tabs: {
    display: 'flex',
    borderBottom: '1px solid #333',
    marginBottom: '12px',
  },
  tab: {
    flex: 1,
    padding: '8px',
    fontSize: '12px',
    background: 'transparent',
    color: '#888',
    border: 'none',
    borderBottom: '2px solid transparent',
    cursor: 'pointer',
  },
  tabActive: {
    color: '#fff',
    borderBottomColor: '#2563eb',
  },
  tabContent: {
    flex: 1,
    overflow: 'auto',
    minHeight: '200px',
  },
  filterRow: {
    marginBottom: '8px',
  },
  filterSelect: {
    width: '100%',
    padding: '6px 8px',
    fontSize: '12px',
    background: '#111',
    color: '#ccc',
    border: '1px solid #333',
    borderRadius: '4px',
  },
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
  pcRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '6px 8px',
    borderRadius: '4px',
    background: 'rgba(255,255,255,0.03)',
  },
  pcInfo: {
    flex: 1,
    minWidth: 0,
  },
  pcName: {
    fontSize: '12px',
    color: '#ccc',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  pcMeta: {
    display: 'flex',
    gap: '6px',
    alignItems: 'center',
    marginTop: '2px',
  },
  badge: {
    fontSize: '9px',
    color: '#888',
    background: 'rgba(255,255,255,0.06)',
    padding: '1px 5px',
    borderRadius: '2px',
  },
  metaText: {
    fontSize: '9px',
    color: '#666',
  },
  addBtn: {
    background: '#1e3a5f',
    color: '#7dc4ff',
    border: '1px solid #2563eb',
    borderRadius: '3px',
    padding: '3px 10px',
    fontSize: '11px',
    cursor: 'pointer',
    flexShrink: 0,
  },
  addBtnDisabled: {
    background: '#1a2a1a',
    color: '#555',
    borderColor: '#333',
    cursor: 'default',
  },
  status: {
    fontSize: '12px',
    color: '#666',
    padding: '16px 4px',
  },
  field: {
    marginBottom: '10px',
  },
  label: {
    display: 'block',
    marginBottom: '4px',
    fontSize: '11px',
    color: '#999',
  },
  input: {
    width: '100%',
    padding: '7px 8px',
    fontSize: '12px',
    background: '#111',
    color: '#ccc',
    border: '1px solid #333',
    borderRadius: '4px',
    boxSizing: 'border-box' as const,
  },
  select: {
    width: '100%',
    padding: '7px 8px',
    fontSize: '12px',
    background: '#111',
    color: '#ccc',
    border: '1px solid #333',
    borderRadius: '4px',
  },
  error: {
    color: '#ef4444',
    fontSize: '12px',
    marginBottom: '8px',
  },
  submitBtn: {
    width: '100%',
    padding: '8px 12px',
    fontSize: '12px',
    fontWeight: 600,
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    marginTop: '4px',
  },
  footer: {
    display: 'flex',
    justifyContent: 'flex-end',
    marginTop: '12px',
    paddingTop: '12px',
    borderTop: '1px solid #222',
  },
  closeBtn: {
    padding: '6px 16px',
    fontSize: '12px',
    background: 'transparent',
    color: '#999',
    border: '1px solid #444',
    borderRadius: '4px',
    cursor: 'pointer',
  },
};
