/**
 * Workspace browser sidebar for the Implant Checker.
 *
 * Allows selecting/creating projects (workspaces), viewing their
 * point clouds, adding layers to the viewer, and auto-matching
 * skull/implant pairs by skull_id.
 */
import { useState } from 'react';
import { useProjects, useCreateProject, useProjectPointClouds, useAutoMatch } from '../../hooks/useProjects';
import { LayerManager } from './LayerManager';
import { CheckerImportDialog } from './CheckerImportDialog';
import type { PointCloud } from '../../types/point-cloud';
import type { CheckerLayer, SkullImplantPair } from '../../types/checker';

interface WorkspaceBrowserProps {
  selectedProjectId: string | null;
  onSelectProject: (projectId: string | null) => void;
  layers: CheckerLayer[];
  onAddLayer: (pc: PointCloud) => void;
  onToggleVisibility: (layerId: string) => void;
  onSetColor: (layerId: string, color: string) => void;
  onSetHeatmap: (layerId: string, useHeatmap: boolean) => void;
  onRemoveLayer: (layerId: string) => void;
  onAddFromAutoMatch: (pairs: SkullImplantPair[]) => void;
  onClearLayers: () => void;
  onGenerateSTL?: (pcId: string) => void;
  onDownloadSTL?: (pcId: string) => void;
  generatingSTLForId?: string | null;
}

export function WorkspaceBrowser({
  selectedProjectId,
  onSelectProject,
  layers,
  onAddLayer,
  onToggleVisibility,
  onSetColor,
  onSetHeatmap,
  onRemoveLayer,
  onAddFromAutoMatch,
  onClearLayers,
  onGenerateSTL,
  onDownloadSTL,
  generatingSTLForId,
}: WorkspaceBrowserProps) {
  const { data: projects = [] } = useProjects();
  const createProject = useCreateProject();
  const { data: pointClouds = [] } = useProjectPointClouds(selectedProjectId);
  const { data: autoMatchPairs } = useAutoMatch(selectedProjectId);
  const [newProjectName, setNewProjectName] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const [showImportDialog, setShowImportDialog] = useState(false);

  const handleCreateProject = () => {
    if (!newProjectName.trim()) return;
    createProject.mutate(
      { name: newProjectName.trim() },
      {
        onSuccess: (project) => {
          onSelectProject(project.id);
          setNewProjectName('');
          setShowCreate(false);
        },
      },
    );
  };

  const handleAutoMatch = () => {
    if (autoMatchPairs && autoMatchPairs.length > 0) {
      onAddFromAutoMatch(autoMatchPairs);
    }
  };

  const loadedIds = new Set(layers.map((l) => l.pointCloudId));

  return (
    <div style={styles.container}>
      {/* Workspace selector */}
      <section style={styles.section}>
        <h4 style={styles.sectionTitle}>Workspace</h4>
        <div style={styles.selectorRow}>
          <select
            style={styles.select}
            value={selectedProjectId ?? ''}
            onChange={(e) => {
              onSelectProject(e.target.value || null);
              onClearLayers();
            }}
          >
            <option value="">Select workspace...</option>
            {projects.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
          <button style={styles.addBtn} onClick={() => setShowCreate(!showCreate)}>
            +
          </button>
        </div>

        {showCreate && (
          <div style={styles.createRow}>
            <input
              type="text"
              placeholder="Workspace name"
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              style={styles.input}
              onKeyDown={(e) => e.key === 'Enter' && handleCreateProject()}
            />
            <button
              style={styles.createBtn}
              onClick={handleCreateProject}
              disabled={!newProjectName.trim() || createProject.isPending}
            >
              Create
            </button>
          </div>
        )}
      </section>

      {/* Point clouds in workspace */}
      {selectedProjectId && (
        <section style={styles.section}>
          <div style={styles.sectionHeader}>
            <h4 style={styles.sectionTitle}>Point Clouds</h4>
            <div style={{ display: 'flex', gap: '4px' }}>
              <button style={styles.addBtn} onClick={() => setShowImportDialog(true)}>
                + Add
              </button>
              {autoMatchPairs && autoMatchPairs.length > 0 && (
                <button style={styles.autoMatchBtn} onClick={handleAutoMatch}>
                  Auto-Match
                </button>
              )}
            </div>
          </div>

          {pointClouds.length === 0 ? (
            <div style={styles.emptyText}>
              No point clouds in this workspace.
            </div>
          ) : (
            <div style={styles.pcList}>
              {pointClouds.map((pc) => (
                <div key={pc.id} style={styles.pcRow}>
                  <div style={styles.pcInfo}>
                    <div style={styles.pcName}>{pc.name}</div>
                    {pc.scan_category && (
                      <span style={styles.badge}>{pc.scan_category}</span>
                    )}
                  </div>
                  <button
                    style={{
                      ...styles.addLayerBtn,
                      ...(loadedIds.has(pc.id) ? styles.addLayerBtnLoaded : {}),
                    }}
                    onClick={() => onAddLayer(pc)}
                    disabled={loadedIds.has(pc.id)}
                  >
                    {loadedIds.has(pc.id) ? 'Added' : 'Add'}
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {/* Layer manager */}
      <section style={styles.section}>
        <h4 style={styles.sectionTitle}>Layers</h4>
        <LayerManager
          layers={layers}
          onToggleVisibility={onToggleVisibility}
          onSetColor={onSetColor}
          onSetHeatmap={onSetHeatmap}
          onRemove={onRemoveLayer}
          onGenerateSTL={onGenerateSTL}
          onDownloadSTL={onDownloadSTL}
          generatingSTLForId={generatingSTLForId}
        />
      </section>

      {/* Import dialog */}
      {showImportDialog && selectedProjectId && (
        <CheckerImportDialog
          projectId={selectedProjectId}
          loadedPcIds={loadedIds}
          onAddLayer={onAddLayer}
          onClose={() => setShowImportDialog(false)}
        />
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '12px',
    display: 'flex',
    flexDirection: 'column',
    gap: '0',
  },
  section: {
    paddingBottom: '12px',
    marginBottom: '12px',
    borderBottom: '1px solid #222',
  },
  sectionHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px',
  },
  sectionTitle: {
    margin: '0 0 8px',
    fontSize: '11px',
    fontWeight: 600,
    color: '#aaa',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  selectorRow: {
    display: 'flex',
    gap: '4px',
  },
  select: {
    flex: 1,
    background: '#111',
    color: '#ccc',
    border: '1px solid #333',
    borderRadius: '4px',
    padding: '4px 8px',
    fontSize: '11px',
  },
  addBtn: {
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    padding: '4px 10px',
    fontSize: '12px',
    cursor: 'pointer',
    fontWeight: 700,
  },
  createRow: {
    display: 'flex',
    gap: '4px',
    marginTop: '6px',
  },
  input: {
    flex: 1,
    background: '#111',
    color: '#ccc',
    border: '1px solid #333',
    borderRadius: '4px',
    padding: '4px 8px',
    fontSize: '11px',
  },
  createBtn: {
    background: '#10b981',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    padding: '4px 8px',
    fontSize: '10px',
    cursor: 'pointer',
  },
  autoMatchBtn: {
    background: '#7c3aed',
    color: '#fff',
    border: 'none',
    borderRadius: '3px',
    padding: '2px 8px',
    fontSize: '10px',
    cursor: 'pointer',
  },
  pcList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
  pcRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '4px 6px',
    borderRadius: '4px',
    background: 'rgba(255,255,255,0.02)',
  },
  pcInfo: {
    flex: 1,
    minWidth: 0,
  },
  pcName: {
    fontSize: '11px',
    color: '#ccc',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  badge: {
    fontSize: '9px',
    color: '#888',
    background: 'rgba(255,255,255,0.05)',
    padding: '1px 4px',
    borderRadius: '2px',
  },
  addLayerBtn: {
    background: '#1e3a5f',
    color: '#7dc4ff',
    border: '1px solid #2563eb',
    borderRadius: '3px',
    padding: '2px 8px',
    fontSize: '10px',
    cursor: 'pointer',
    flexShrink: 0,
  },
  addLayerBtnLoaded: {
    background: '#1a2a1a',
    color: '#555',
    borderColor: '#333',
    cursor: 'default',
  },
  emptyText: {
    fontSize: '11px',
    color: '#555',
    padding: '4px 0',
  },
};
