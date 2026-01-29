/**
 * Data Viewer page — the original main view of the application.
 *
 * Extracted from App.tsx so it can live under a route alongside the
 * Implant Checker page.
 */
import { useState } from 'react';
import { AppLayout } from '../components/layout/AppLayout';
import { DataBrowser } from '../components/data-browser/DataBrowser';
import { ViewerContainer } from '../components/viewer/ViewerContainer';
import { SDFColorBar } from '../components/viewer/SDFColorBar';
import { ColorProfileSelector } from '../components/color-profiles/ColorProfileSelector';
import { ColorProfileEditor } from '../components/color-profiles/ColorProfileEditor';
import { AuditLogPanel } from '../components/audit/AuditLogPanel';
import { useColorProfile } from '../hooks/useColorProfiles';
import type { ViewerMode } from '../types/viewer';

export function DataViewerPage() {
  // Data selection state
  const [selectedScanId, setSelectedScanId] = useState<string | null>(null);
  const [selectedPointCloudId, setSelectedPointCloudId] = useState<string | null>(null);

  // Viewer state
  const [viewerMode, setViewerMode] = useState<ViewerMode>('point_cloud');
  const [colorProfileId, setColorProfileId] = useState<string | null>(null);
  const [pointSize, setPointSize] = useState(0.01);
  const [showGrid, setShowGrid] = useState(false);
  const [showAxes, setShowAxes] = useState(false);
  const [showColorEditor, setShowColorEditor] = useState(false);

  // Auto-switch viewer mode when data is selected
  const handleSelectScan = (scanId: string) => {
    setSelectedScanId(scanId);
    if (!selectedPointCloudId) {
      setViewerMode('volume');
    }
  };

  const handleSelectPointCloud = (pcId: string) => {
    setSelectedPointCloudId(pcId);
    if (!selectedScanId) {
      setViewerMode('point_cloud');
    }
  };

  return (
    <AppLayout
      header={null}
      sidebar={
        <DataBrowser
          onSelectScan={handleSelectScan}
          onSelectPointCloud={handleSelectPointCloud}
          selectedScanId={selectedScanId}
          selectedPointCloudId={selectedPointCloudId}
        />
      }
      main={
        <ViewerContainer
          mode={viewerMode}
          selectedScanId={selectedScanId}
          selectedPointCloudId={selectedPointCloudId}
          colorProfileId={colorProfileId}
          pointSize={pointSize}
          showGrid={showGrid}
          showAxes={showAxes}
          onModeChange={setViewerMode}
        />
      }
      controls={
        <ControlsSidebar
          pointSize={pointSize}
          onPointSizeChange={setPointSize}
          showGrid={showGrid}
          onShowGridChange={setShowGrid}
          showAxes={showAxes}
          onShowAxesChange={setShowAxes}
          colorProfileId={colorProfileId}
          onColorProfileChange={setColorProfileId}
          onCreateProfile={() => setShowColorEditor(true)}
          showColorEditor={showColorEditor}
          onCloseColorEditor={() => setShowColorEditor(false)}
        />
      }
    />
  );
}

// ---------------------------------------------------------------------------
// Controls sidebar (moved from App.tsx)
// ---------------------------------------------------------------------------

interface ControlsSidebarProps {
  pointSize: number;
  onPointSizeChange: (size: number) => void;
  showGrid: boolean;
  onShowGridChange: (show: boolean) => void;
  showAxes: boolean;
  onShowAxesChange: (show: boolean) => void;
  colorProfileId: string | null;
  onColorProfileChange: (id: string | null) => void;
  onCreateProfile: () => void;
  showColorEditor: boolean;
  onCloseColorEditor: () => void;
}

function ControlsSidebar({
  pointSize,
  onPointSizeChange,
  showGrid,
  onShowGridChange,
  showAxes,
  onShowAxesChange,
  colorProfileId,
  onColorProfileChange,
  onCreateProfile,
  showColorEditor,
  onCloseColorEditor,
}: ControlsSidebarProps) {
  const { data: profile } = useColorProfile(colorProfileId);

  return (
    <div style={controlStyles.container}>
      {/* Point Cloud Display */}
      <section style={controlStyles.section}>
        <h4 style={controlStyles.sectionTitle}>Point Cloud Display</h4>

        <label style={controlStyles.label}>Point Size</label>
        <input
          type="range"
          min="0.001"
          max="0.05"
          step="0.001"
          value={pointSize}
          onChange={(e) => onPointSizeChange(parseFloat(e.target.value))}
          style={controlStyles.slider}
        />
        <span style={controlStyles.sliderValue}>{pointSize.toFixed(3)}</span>

        <div style={controlStyles.checkboxRow}>
          <label style={controlStyles.checkboxLabel}>
            <input
              type="checkbox"
              checked={showGrid}
              onChange={(e) => onShowGridChange(e.target.checked)}
            />
            Show Grid
          </label>
        </div>

        <div style={controlStyles.checkboxRow}>
          <label style={controlStyles.checkboxLabel}>
            <input
              type="checkbox"
              checked={showAxes}
              onChange={(e) => onShowAxesChange(e.target.checked)}
            />
            Show Axes
          </label>
        </div>
      </section>

      {/* SDF Colorization */}
      <section style={controlStyles.section}>
        <div style={controlStyles.sectionHeader}>
          <h4 style={controlStyles.sectionTitle}>SDF Colorization</h4>
          <button style={controlStyles.newBtn} onClick={onCreateProfile}>
            + New
          </button>
        </div>

        <ColorProfileSelector
          selectedId={colorProfileId}
          onSelect={onColorProfileChange}
        />

        {profile && (
          <div style={{ marginTop: '8px' }}>
            <SDFColorBar profile={profile} />
          </div>
        )}
      </section>

      {/* Audit Trail */}
      <section style={controlStyles.section}>
        <AuditLogPanel limit={10} />
      </section>

      {/* Help */}
      <section style={controlStyles.section}>
        <h4 style={controlStyles.sectionTitle}>Navigation</h4>
        <div style={controlStyles.help}>
          <div>Left Click: Rotate</div>
          <div>Right Click: Pan</div>
          <div>Scroll: Zoom</div>
        </div>
      </section>

      {/* Color Profile Editor Dialog */}
      {showColorEditor && (
        <ColorProfileEditor onClose={onCloseColorEditor} />
      )}
    </div>
  );
}

const controlStyles: Record<string, React.CSSProperties> = {
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
  label: {
    display: 'block',
    fontSize: '11px',
    color: '#888',
    marginBottom: '4px',
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
  newBtn: {
    padding: '2px 8px',
    fontSize: '10px',
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '3px',
    cursor: 'pointer',
  },
  help: {
    fontSize: '11px',
    color: '#666',
    lineHeight: '1.8',
  },
};
