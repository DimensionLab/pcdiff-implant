import { useState, useEffect } from 'react';
import { ScansList } from './ScansList';
import { PointCloudsList } from './PointCloudsList';
import { DataImportDialog } from './DataImportDialog';
import { SkullBreakImporter } from './SkullBreakImporter';
import { useProjects } from '../../hooks/useProjects';

interface DataBrowserProps {
  onSelectScan: (scanId: string) => void;
  onSelectPointCloud: (pcId: string) => void;
  selectedScanId: string | null;
  selectedPointCloudId: string | null;
  defaultProjectId?: string;
}

type Tab = 'scans' | 'point-clouds';

export function DataBrowser({
  onSelectScan,
  onSelectPointCloud,
  selectedScanId,
  selectedPointCloudId,
  defaultProjectId,
}: DataBrowserProps) {
  const [activeTab, setActiveTab] = useState<Tab>('scans');
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [showSkullBreakImport, setShowSkullBreakImport] = useState(false);
  const [categoryFilter, setCategoryFilter] = useState<string>('');
  
  const { data: projects = [] } = useProjects();
  const defaultProject = defaultProjectId ? projects.find(p => p.id === defaultProjectId) : null;

  // Pre-filter to defective skulls when coming from Implant Generator with a project
  useEffect(() => {
    if (defaultProjectId) {
      setCategoryFilter('defective_skull');
    }
  }, [defaultProjectId]);

  return (
    <div style={styles.container}>
      {/* Project context banner */}
      {defaultProject && (
        <div style={styles.projectBanner}>
          <span>📁 Assign data to: <strong>{defaultProject.name}</strong></span>
          <div style={styles.projectHint}>Click 📁 next to any item to assign it to this project</div>
        </div>
      )}

      <div style={styles.header}>
        <h3 style={styles.title}>Data Browser</h3>
        <div style={styles.actions}>
          <button
            style={styles.importBtn}
            onClick={() => setShowImportDialog(true)}
            title="Register file path"
          >
            + Add
          </button>
          <button
            style={styles.importBtn}
            onClick={() => setShowSkullBreakImport(true)}
            title="Import SkullBreak dataset"
          >
            Import
          </button>
        </div>
      </div>

      {/* Category filter */}
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

      {/* Tabs */}
      <div style={styles.tabs}>
        <button
          style={{
            ...styles.tab,
            ...(activeTab === 'scans' ? styles.tabActive : {}),
          }}
          onClick={() => setActiveTab('scans')}
        >
          Volumes
        </button>
        <button
          style={{
            ...styles.tab,
            ...(activeTab === 'point-clouds' ? styles.tabActive : {}),
          }}
          onClick={() => setActiveTab('point-clouds')}
        >
          Point Clouds
        </button>
      </div>

      {/* Content */}
      <div style={styles.content}>
        {activeTab === 'scans' && (
          <ScansList
            onSelect={onSelectScan}
            selectedId={selectedScanId}
            categoryFilter={categoryFilter}
            showAssignButton={true}
            defaultProjectId={defaultProjectId}
          />
        )}
        {activeTab === 'point-clouds' && (
          <PointCloudsList
            onSelect={onSelectPointCloud}
            selectedId={selectedPointCloudId}
            categoryFilter={categoryFilter}
            showAssignButton={true}
            defaultProjectId={defaultProjectId}
          />
        )}
      </div>

      {/* Dialogs */}
      {showImportDialog && (
        <DataImportDialog 
          onClose={() => setShowImportDialog(false)} 
          defaultProjectId={defaultProjectId}
        />
      )}
      {showSkullBreakImport && (
        <SkullBreakImporter onClose={() => setShowSkullBreakImport(false)} />
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    background: '#1a1a2e',
    color: '#e0e0e0',
  },
  projectBanner: {
    padding: '10px 16px',
    background: 'rgba(37, 99, 235, 0.15)',
    borderBottom: '1px solid rgba(37, 99, 235, 0.3)',
    fontSize: '12px',
    color: '#93c5fd',
  },
  projectHint: {
    fontSize: '11px',
    color: '#6b8fc7',
    marginTop: '4px',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '12px 16px 8px',
    borderBottom: '1px solid #333',
  },
  title: {
    margin: 0,
    fontSize: '14px',
    fontWeight: 600,
    color: '#fff',
  },
  actions: {
    display: 'flex',
    gap: '6px',
  },
  importBtn: {
    padding: '4px 10px',
    fontSize: '12px',
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  filterRow: {
    padding: '8px 16px',
  },
  filterSelect: {
    width: '100%',
    padding: '6px 8px',
    fontSize: '12px',
    background: '#16213e',
    color: '#e0e0e0',
    border: '1px solid #333',
    borderRadius: '4px',
  },
  tabs: {
    display: 'flex',
    borderBottom: '1px solid #333',
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
  content: {
    flex: 1,
    overflow: 'auto',
  },
};
