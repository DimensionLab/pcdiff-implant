/**
 * Orchestrates which 3D viewer to display based on the viewer mode
 * and what data is selected.
 *
 * - Point Cloud mode: React Three Fiber (mounted only when a point
 *   cloud has been selected; once mounted it stays alive via CSS
 *   visibility to avoid R3F Canvas teardown crash)
 * - Volume mode: vtk.js
 * - Split mode: side-by-side
 */
import { useEffect, useState } from 'react';
import { ErrorBoundary } from '../common/ErrorBoundary';
import { PointCloudViewer } from './PointCloudViewer';
import { VtkViewport } from './VtkViewport';
import { ViewerToolbar } from './ViewerToolbar';
import type { ViewerMode } from '../../types/viewer';

interface ViewerContainerProps {
  mode: ViewerMode;
  selectedScanId: string | null;
  selectedPointCloudId: string | null;
  colorProfileId: string | null;
  pointSize: number;
  showGrid: boolean;
  showAxes: boolean;
  onModeChange: (mode: ViewerMode) => void;
}

export function ViewerContainer({
  mode,
  selectedScanId,
  selectedPointCloudId,
  colorProfileId,
  pointSize,
  showGrid,
  showAxes,
  onModeChange,
}: ViewerContainerProps) {
  const showVolume = mode === 'volume' || mode === 'split';
  const showPointCloud = mode === 'point_cloud' || mode === 'split';
  const isSplit = mode === 'split';

  // Track whether each viewer has ever been needed.  Once mounted we
  // keep it alive (hidden via CSS) to avoid React Three Fiber's
  // Canvas teardown crash ("traverse" on undefined).
  const [pcMounted, setPcMounted] = useState(false);
  const [vtkMounted, setVtkMounted] = useState(false);

  useEffect(() => {
    if (selectedPointCloudId && !pcMounted) setPcMounted(true);
  }, [selectedPointCloudId, pcMounted]);

  useEffect(() => {
    if (selectedScanId && !vtkMounted) setVtkMounted(true);
  }, [selectedScanId, vtkMounted]);

  const hasAnyViewer = pcMounted || vtkMounted;

  return (
    <div style={styles.container}>
      <ViewerToolbar
        mode={mode}
        onModeChange={onModeChange}
        hasScan={!!selectedScanId}
        hasPointCloud={!!selectedPointCloudId}
      />

      <div style={{
        ...styles.viewerArea,
        ...(isSplit ? { display: 'flex' } : {}),
      }}>
        {/* Volume viewer (vtk.js) — mount once a scan is selected */}
        {vtkMounted && (
          <div style={{
            ...(isSplit ? styles.splitPane : styles.overlayPane),
            visibility: showVolume ? 'visible' : 'hidden',
            pointerEvents: showVolume ? 'auto' : 'none',
          }}>
            <ErrorBoundary>
              <VtkViewport scanId={selectedScanId} />
            </ErrorBoundary>
          </div>
        )}

        {isSplit && <div style={styles.splitDivider} />}

        {/* Point cloud viewer (R3F) — mount once a PC is selected */}
        {pcMounted && (
          <div style={{
            ...(isSplit ? styles.splitPane : styles.overlayPane),
            visibility: showPointCloud ? 'visible' : 'hidden',
            pointerEvents: showPointCloud ? 'auto' : 'none',
          }}>
            <ErrorBoundary>
              <PointCloudViewer
                pointCloudId={selectedPointCloudId}
                colorProfileId={colorProfileId}
                pointSize={pointSize}
                showGrid={showGrid}
                showAxes={showAxes}
              />
            </ErrorBoundary>
          </div>
        )}

        {/* Empty-state prompt when nothing has been loaded yet */}
        {!hasAnyViewer && (
          <div style={styles.emptyState}>
            <div style={styles.emptyText}>
              Select a volume or point cloud from the Data Browser
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    background: '#0a0a1a',
  },
  viewerArea: {
    flex: 1,
    position: 'relative',
    overflow: 'hidden',
  },
  overlayPane: {
    position: 'absolute',
    inset: 0,
  },
  splitPane: {
    flex: 1,
    position: 'relative',
    overflow: 'hidden',
  },
  splitDivider: {
    width: '2px',
    background: '#333',
  },
  emptyState: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  emptyText: {
    color: '#555',
    fontSize: '14px',
  },
};
