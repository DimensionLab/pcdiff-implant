/**
 * Implant Checker — professional overlay viewer for defective-skull + implant NRRDs.
 *
 * Features:
 * - Automatic alignment: overlay is resampled server-side to match the base scan grid.
 * - Skull controls: visibility toggle, opacity, window/level with presets.
 * - Overlay controls: color picker, opacity, visibility toggle.
 * - Cross-section clipping planes along X/Y/Z.
 * - Camera view presets (anterior, posterior, left, right, superior, inferior).
 */
import { useState, useEffect, useMemo, useRef, useCallback, type CSSProperties } from 'react';
import { AppLayout } from '../components/layout/AppLayout';
import { VtkViewport, type VtkViewportHandle, type CameraPreset, type OverlayRenderMode } from '../components/viewer/VtkViewport';
import { useProjects } from '../hooks/useProjects';
import { useScans } from '../hooks/useScans';
import { useProjectJobs } from '../hooks/useGeneration';
import { JobDownloads } from '../components/generation/JobDownloads';
import { generationApi } from '../services/generation-api';

const STATUS_COLORS: Record<string, string> = {
  pending: '#f59e0b',
  running: '#3b82f6',
  completed: '#10b981',
  failed: '#ef4444',
  cancelled: '#6b7280',
};

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.replace('#', ''), 16);
  return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
}

const WL_PRESETS: Array<{ label: string; window: number; level: number }> = [
  { label: 'Bone', window: 175, level: 168 },
  { label: 'Soft tissue', window: 100, level: 128 },
  { label: 'Wide', window: 255, level: 128 },
  { label: 'Dense bone', window: 120, level: 200 },
];

const CAMERA_VIEWS: Array<{ label: string; preset: CameraPreset }> = [
  { label: 'A', preset: 'anterior' },
  { label: 'P', preset: 'posterior' },
  { label: 'L', preset: 'left' },
  { label: 'R', preset: 'right' },
  { label: 'S', preset: 'superior' },
  { label: 'I', preset: 'inferior' },
];

export function ImplantCheckerPage() {
  const viewportRef = useRef<VtkViewportHandle>(null);

  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [baseScanId, setBaseScanId] = useState<string | null>(null);
  const [overlayScanId, setOverlayScanId] = useState<string | null>(null);

  // Overlay controls
  const [overlayColor, setOverlayColor] = useState<[number, number, number]>([1.0, 0.2, 0.25]);
  const [overlayColorHex, setOverlayColorHex] = useState('#ff3340');
  const [overlayOpacity, setOverlayOpacity] = useState(0.75);
  const [overlayVisible, setOverlayVisible] = useState(true);

  // Skull controls
  const [baseVisible, setBaseVisible] = useState(true);
  const [baseOpacity, setBaseOpacity] = useState(1.0);
  const [windowWidth, setWindowWidth] = useState(175);
  const [windowLevel, setWindowLevel] = useState(168);

  // Render mode
  const [overlayRenderMode, setOverlayRenderMode] = useState<OverlayRenderMode>('volume');

  // Clipping plane
  const [clipAxis, setClipAxis] = useState<'x' | 'y' | 'z' | 'none'>('none');
  const [clipPosition, setClipPosition] = useState(0.5);

  const { data: projects = [] } = useProjects();
  const { data: projectScans = [] } = useScans(
    selectedProjectId ? { project_id: selectedProjectId } : undefined,
  );
  const { data: projectJobs = [] } = useProjectJobs(selectedProjectId);

  const completedJobs = useMemo(
    () => projectJobs.filter((j) => j.status === 'completed' && j.output_scan_id),
    [projectJobs],
  );

  const nrrdScans = useMemo(
    () => projectScans.filter((s) => s.file_format?.toLowerCase() === 'nrrd'),
    [projectScans],
  );
  const defectiveScans = useMemo(
    () => nrrdScans.filter((s) => s.scan_category === 'defective_skull'),
    [nrrdScans],
  );
  const implantScans = useMemo(
    () => nrrdScans.filter((s) => s.scan_category !== 'defective_skull'),
    [nrrdScans],
  );

  useEffect(() => {
    if (!selectedJobId) return;
    const job = completedJobs.find((j) => j.id === selectedJobId);
    if (!job) return;
    if (job.input_scan_id) setBaseScanId(job.input_scan_id);
    if (job.output_scan_id) setOverlayScanId(job.output_scan_id);
  }, [selectedJobId, completedJobs]);

  // Derive mesh download URL from the selected job for mesh render mode
  const overlayMeshUrl = useMemo(() => {
    if (overlayRenderMode !== 'mesh' || !selectedJobId) return null;
    return generationApi.downloadUrl(selectedJobId, 'stl');
  }, [overlayRenderMode, selectedJobId]);

  const handleManualBase = (id: string | null) => {
    setBaseScanId(id);
    setSelectedJobId(null);
  };
  const handleManualOverlay = (id: string | null) => {
    setOverlayScanId(id);
    setSelectedJobId(null);
  };

  // Propagate skull visibility/opacity/window-level changes to the viewport
  const handleBaseVisibleToggle = useCallback(() => {
    const next = !baseVisible;
    setBaseVisible(next);
    viewportRef.current?.setBaseVisible(next);
  }, [baseVisible]);

  const handleOverlayVisibleToggle = useCallback(() => {
    const next = !overlayVisible;
    setOverlayVisible(next);
    viewportRef.current?.setOverlayVisible(next);
  }, [overlayVisible]);

  const handleOverlayColorChange = useCallback((hex: string) => {
    setOverlayColorHex(hex);
    const rgb = hexToRgb(hex);
    setOverlayColor(rgb);
    viewportRef.current?.setOverlayColor(rgb[0], rgb[1], rgb[2]);
  }, []);

  const handleOverlayOpacityChange = useCallback((val: number) => {
    setOverlayOpacity(val);
    viewportRef.current?.setOverlayOpacity(val);
  }, []);

  const handleBaseOpacityChange = useCallback((val: number) => {
    setBaseOpacity(val);
    viewportRef.current?.setBaseOpacity(val);
  }, []);

  const handleWindowLevel = useCallback((w: number, l: number) => {
    setWindowWidth(w);
    setWindowLevel(l);
    viewportRef.current?.setBaseWindowLevel(w, l);
  }, []);

  const handleClipAxisChange = useCallback((axis: 'x' | 'y' | 'z' | 'none') => {
    setClipAxis(axis);
    if (axis === 'none') {
      viewportRef.current?.clearClippingPlanes();
    } else {
      viewportRef.current?.setClippingPlane(axis, clipPosition);
    }
  }, [clipPosition]);

  const handleClipPositionChange = useCallback((pos: number) => {
    setClipPosition(pos);
    if (clipAxis !== 'none') {
      viewportRef.current?.setClippingPlane(clipAxis, pos);
    }
  }, [clipAxis]);

  return (
    <AppLayout
      sidebar={
        <div style={styles.sidebar}>
          <section style={styles.section}>
            <h3 style={styles.sectionTitle}>Project</h3>
            <select
              value={selectedProjectId ?? ''}
              onChange={(e) => {
                setSelectedProjectId(e.target.value || null);
                setSelectedJobId(null);
                setBaseScanId(null);
                setOverlayScanId(null);
              }}
              style={styles.select}
            >
              <option value="">Select project…</option>
              {projects.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </section>

          {selectedProjectId && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Completed Generations</h3>
              {completedJobs.length === 0 ? (
                <p style={styles.emptyText}>
                  No completed cran-2 jobs in this project yet.
                </p>
              ) : (
                <div style={styles.jobList}>
                  {completedJobs.map((job) => (
                    <button
                      key={job.id}
                      onClick={() => setSelectedJobId(job.id)}
                      style={{
                        ...styles.jobItem,
                        background:
                          selectedJobId === job.id
                            ? 'rgba(59, 130, 246, 0.2)'
                            : 'transparent',
                      }}
                    >
                      <div style={styles.jobItemName}>{job.name}</div>
                      <div style={styles.jobItemMeta}>
                        <span
                          style={{
                            ...styles.statusBadge,
                            background: STATUS_COLORS[job.status] ?? '#6b7280',
                          }}
                        >
                          {job.status}
                        </span>
                        <span style={styles.jobItemTime}>
                          threshold {job.threshold.toFixed(2)}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
              {selectedJobId && (
                <JobDownloads jobId={selectedJobId} />
              )}
            </section>
          )}

          {selectedProjectId && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Defective Skull</h3>
              <select
                value={baseScanId ?? ''}
                onChange={(e) => handleManualBase(e.target.value || null)}
                style={styles.select}
              >
                <option value="">Select scan…</option>
                {defectiveScans.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </section>
          )}

          {selectedProjectId && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Implant Overlay</h3>
              <select
                value={overlayScanId ?? ''}
                onChange={(e) => handleManualOverlay(e.target.value || null)}
                style={styles.select}
              >
                <option value="">No overlay</option>
                {implantScans.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </section>
          )}
        </div>
      }
      main={
        <div style={styles.mainViewport}>
          {baseScanId ? (
            <VtkViewport
              ref={viewportRef}
              scanId={baseScanId}
              overlayScanId={overlayRenderMode === 'volume' ? overlayScanId : null}
              overlayColor={overlayColor}
              overlayOpacity={overlayOpacity}
              overlayRenderMode={overlayRenderMode}
              overlayMeshUrl={overlayMeshUrl}
            />
          ) : (
            <div style={styles.placeholder}>
              <h2 style={styles.placeholderTitle}>Implant Checker</h2>
              <p style={styles.placeholderText}>
                Select a project, then choose a completed cran-2 job or pick scans manually
                to visualize a defective skull with its implant overlay.
              </p>
            </div>
          )}
        </div>
      }
      controls={
        <div style={styles.controlsPanel}>
          {/* Camera views */}
          <section style={styles.section}>
            <h3 style={styles.sectionTitle}>Camera</h3>
            <div style={styles.viewGrid}>
              {CAMERA_VIEWS.map((v) => (
                <button
                  key={v.preset}
                  onClick={() => viewportRef.current?.setCameraPreset(v.preset)}
                  style={styles.viewBtn}
                  title={v.preset}
                >
                  {v.label}
                </button>
              ))}
              <button
                onClick={() => viewportRef.current?.resetCamera()}
                style={{ ...styles.viewBtn, gridColumn: 'span 2' }}
                title="Reset camera"
              >
                Reset
              </button>
            </div>
          </section>

          {/* Skull volume controls */}
          <section style={styles.section}>
            <div style={styles.sectionHeader}>
              <h3 style={styles.sectionTitle}>Skull Volume</h3>
              <button
                onClick={handleBaseVisibleToggle}
                style={{
                  ...styles.toggleBtn,
                  color: baseVisible ? '#10b981' : '#666',
                }}
                title={baseVisible ? 'Hide skull' : 'Show skull'}
              >
                {baseVisible ? '👁' : '👁‍🗨'}
              </button>
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>
                Opacity: {baseOpacity.toFixed(2)}
              </label>
              <input
                type="range"
                min={0.0}
                max={1.0}
                step={0.05}
                value={baseOpacity}
                onChange={(e) => handleBaseOpacityChange(Number(e.target.value))}
                style={styles.slider}
              />
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>Window / Level Presets</label>
              <div style={styles.presetRow}>
                {WL_PRESETS.map((p) => (
                  <button
                    key={p.label}
                    onClick={() => handleWindowLevel(p.window, p.level)}
                    style={{
                      ...styles.presetBtn,
                      background:
                        windowWidth === p.window && windowLevel === p.level
                          ? '#2563eb'
                          : '#222',
                    }}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>
                Window: {windowWidth}
              </label>
              <input
                type="range"
                min={10}
                max={255}
                step={5}
                value={windowWidth}
                onChange={(e) => handleWindowLevel(Number(e.target.value), windowLevel)}
                style={styles.slider}
              />
            </div>
            <div style={styles.formGroup}>
              <label style={styles.label}>
                Level: {windowLevel}
              </label>
              <input
                type="range"
                min={0}
                max={255}
                step={5}
                value={windowLevel}
                onChange={(e) => handleWindowLevel(windowWidth, Number(e.target.value))}
                style={styles.slider}
              />
            </div>
          </section>

          {/* Render mode */}
          <section style={styles.section}>
            <h3 style={styles.sectionTitle}>Render Mode</h3>
            <div style={styles.formGroup}>
              <div style={styles.clipAxisRow}>
                {(['volume', 'mesh'] as const).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setOverlayRenderMode(mode)}
                    style={{
                      ...styles.clipAxisBtn,
                      background: overlayRenderMode === mode ? '#2563eb' : '#222',
                      color: overlayRenderMode === mode ? '#fff' : '#999',
                    }}
                  >
                    {mode === 'volume' ? 'NRRD Volume' : 'STL Mesh'}
                  </button>
                ))}
              </div>
              {overlayRenderMode === 'mesh' && !selectedJobId && (
                <p style={styles.hintText}>Select a completed job to load the STL mesh.</p>
              )}
            </div>
          </section>

          {/* Implant overlay controls */}
          <section style={styles.section}>
            <div style={styles.sectionHeader}>
              <h3 style={styles.sectionTitle}>Implant Overlay</h3>
              <button
                onClick={handleOverlayVisibleToggle}
                style={{
                  ...styles.toggleBtn,
                  color: overlayVisible ? '#10b981' : '#666',
                }}
                title={overlayVisible ? 'Hide overlay' : 'Show overlay'}
              >
                {overlayVisible ? '👁' : '👁‍🗨'}
              </button>
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>Color</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <input
                  type="color"
                  value={overlayColorHex}
                  onChange={(e) => handleOverlayColorChange(e.target.value)}
                  style={styles.colorPicker}
                />
                <span style={{ fontSize: '11px', color: '#999', fontFamily: 'monospace' }}>
                  {overlayColorHex.toUpperCase()}
                </span>
              </div>
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>
                Opacity: {overlayOpacity.toFixed(2)}
              </label>
              <input
                type="range"
                min={0.1}
                max={1.0}
                step={0.05}
                value={overlayOpacity}
                onChange={(e) => handleOverlayOpacityChange(Number(e.target.value))}
                style={styles.slider}
              />
            </div>
          </section>

          {/* Cross-section clipping */}
          <section style={styles.section}>
            <h3 style={styles.sectionTitle}>Cross-Section</h3>
            <div style={styles.formGroup}>
              <label style={styles.label}>Clipping Axis</label>
              <div style={styles.clipAxisRow}>
                {(['none', 'x', 'y', 'z'] as const).map((a) => (
                  <button
                    key={a}
                    onClick={() => handleClipAxisChange(a)}
                    style={{
                      ...styles.clipAxisBtn,
                      background: clipAxis === a ? '#2563eb' : '#222',
                      color: clipAxis === a ? '#fff' : '#999',
                    }}
                  >
                    {a === 'none' ? 'Off' : a.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
            {clipAxis !== 'none' && (
              <div style={styles.formGroup}>
                <label style={styles.label}>
                  Position: {(clipPosition * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min={0.0}
                  max={1.0}
                  step={0.01}
                  value={clipPosition}
                  onChange={(e) => handleClipPositionChange(Number(e.target.value))}
                  style={styles.slider}
                />
              </div>
            )}
          </section>

          <section style={styles.section}>
            <h3 style={styles.sectionTitle}>About</h3>
            <p style={styles.aboutText}>
              The Implant Checker overlays a cran-2 generated implant on the defective skull
              using vtk.js volume rendering. The overlay is automatically resampled to match
              the skull's voxel grid for accurate alignment.
            </p>
            <p style={styles.aboutText}>
              Use Window/Level to adjust bone contrast. Cross-section clipping lets you
              inspect the implant fit along any axis. Camera presets provide standard
              anatomical viewing angles.
            </p>
          </section>
        </div>
      }
    />
  );
}

const styles: Record<string, CSSProperties> = {
  sidebar: { padding: '12px', display: 'flex', flexDirection: 'column' },
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
  select: {
    width: '100%',
    background: '#111',
    color: '#ccc',
    border: '1px solid #333',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
    boxSizing: 'border-box',
  },
  emptyText: {
    color: '#666',
    fontSize: '11px',
    fontStyle: 'italic',
    margin: 0,
  },
  jobList: { display: 'flex', flexDirection: 'column', gap: '4px' },
  jobItem: {
    padding: '8px',
    borderRadius: '4px',
    cursor: 'pointer',
    transition: 'background 0.2s',
    border: 'none',
    textAlign: 'left',
    color: 'inherit',
  },
  jobItemName: {
    fontSize: '12px',
    fontWeight: 500,
    color: '#fff',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  jobItemMeta: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginTop: '4px',
  },
  jobItemTime: { fontSize: '10px', color: '#666' },
  statusBadge: {
    padding: '2px 6px',
    borderRadius: '4px',
    fontSize: '9px',
    fontWeight: 600,
    textTransform: 'uppercase',
    color: '#fff',
  },
  mainViewport: { position: 'relative', width: '100%', height: '100%' },
  placeholder: {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '24px',
    textAlign: 'center',
  },
  placeholderTitle: {
    fontSize: '24px',
    fontWeight: 600,
    color: '#fff',
    marginBottom: '8px',
  },
  placeholderText: { fontSize: '14px', color: '#888', maxWidth: '400px' },
  controlsPanel: { display: 'flex', flexDirection: 'column' },
  formGroup: { display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '12px' },
  label: { fontSize: '12px', fontWeight: 500, color: '#aaa' },
  slider: { width: '100%', accentColor: '#2563eb' },
  colorPicker: {
    width: '40px',
    height: '28px',
    padding: 0,
    border: '1px solid #444',
    borderRadius: '4px',
    cursor: 'pointer',
    background: 'transparent',
  },
  toggleBtn: {
    background: 'transparent',
    border: 'none',
    cursor: 'pointer',
    fontSize: '16px',
    padding: '2px 4px',
  },
  viewGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap: '4px',
  },
  viewBtn: {
    padding: '6px 4px',
    background: '#222',
    color: '#ccc',
    border: '1px solid #333',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '11px',
    fontWeight: 600,
    textAlign: 'center',
  },
  presetRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '4px',
  },
  presetBtn: {
    padding: '4px 8px',
    borderRadius: '4px',
    border: '1px solid #333',
    color: '#ccc',
    cursor: 'pointer',
    fontSize: '10px',
    fontWeight: 500,
  },
  clipAxisRow: {
    display: 'flex',
    gap: '4px',
  },
  clipAxisBtn: {
    flex: 1,
    padding: '6px 4px',
    borderRadius: '4px',
    border: '1px solid #333',
    cursor: 'pointer',
    fontSize: '11px',
    fontWeight: 600,
    textAlign: 'center',
  },
  hintText: {
    fontSize: '10px',
    color: '#666',
    fontStyle: 'italic',
    margin: '4px 0 0',
  },
  aboutText: {
    fontSize: '11px',
    color: '#888',
    marginBottom: '8px',
    lineHeight: 1.5,
  },
};
