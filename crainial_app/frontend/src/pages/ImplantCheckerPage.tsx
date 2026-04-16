/**
 * Implant Checker — overlay an implant NRRD on a defective-skull NRRD.
 *
 * The cran-2 pipeline produces NRRD masks, so the checker is a thin volume
 * viewer: pick a project, optionally pick a completed cran-2 job (which
 * pairs an input skull with a generated implant), then visualize both
 * volumes blended in vtk.js. Manual scan pickers are exposed for cases
 * where you want to compare against an arbitrary implant NRRD.
 */
import { useState, useEffect, useMemo, type CSSProperties } from 'react';
import { AppLayout } from '../components/layout/AppLayout';
import { VtkViewport } from '../components/viewer/VtkViewport';
import { useProjects } from '../hooks/useProjects';
import { useScans } from '../hooks/useScans';
import { useProjectJobs } from '../hooks/useGeneration';

const STATUS_COLORS: Record<string, string> = {
  pending: '#f59e0b',
  running: '#3b82f6',
  completed: '#10b981',
  failed: '#ef4444',
  cancelled: '#6b7280',
};

const PRESET_COLORS: Array<{ label: string; rgb: [number, number, number] }> = [
  { label: 'Red', rgb: [1.0, 0.2, 0.25] },
  { label: 'Cyan', rgb: [0.2, 0.85, 1.0] },
  { label: 'Green', rgb: [0.25, 1.0, 0.4] },
  { label: 'Yellow', rgb: [1.0, 0.85, 0.2] },
  { label: 'Magenta', rgb: [1.0, 0.3, 0.9] },
];

export function ImplantCheckerPage() {
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [baseScanId, setBaseScanId] = useState<string | null>(null);
  const [overlayScanId, setOverlayScanId] = useState<string | null>(null);
  const [overlayColor, setOverlayColor] = useState<[number, number, number]>(PRESET_COLORS[0].rgb);
  const [overlayOpacity, setOverlayOpacity] = useState(0.75);

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

  // Selecting a job auto-fills both base and overlay scans.
  useEffect(() => {
    if (!selectedJobId) return;
    const job = completedJobs.find((j) => j.id === selectedJobId);
    if (!job) return;
    if (job.input_scan_id) setBaseScanId(job.input_scan_id);
    if (job.output_scan_id) setOverlayScanId(job.output_scan_id);
  }, [selectedJobId, completedJobs]);

  // Reset job-driven selection when the user manually overrides scans.
  const handleManualBase = (id: string | null) => {
    setBaseScanId(id);
    setSelectedJobId(null);
  };
  const handleManualOverlay = (id: string | null) => {
    setOverlayScanId(id);
    setSelectedJobId(null);
  };

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
              scanId={baseScanId}
              overlayScanId={overlayScanId}
              overlayColor={overlayColor}
              overlayOpacity={overlayOpacity}
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
          <section style={styles.section}>
            <h3 style={styles.sectionTitle}>Overlay Style</h3>

            <div style={styles.formGroup}>
              <label style={styles.label}>Color</label>
              <div style={styles.colorRow}>
                {PRESET_COLORS.map((c) => {
                  const active =
                    c.rgb[0] === overlayColor[0] &&
                    c.rgb[1] === overlayColor[1] &&
                    c.rgb[2] === overlayColor[2];
                  return (
                    <button
                      key={c.label}
                      onClick={() => setOverlayColor(c.rgb)}
                      title={c.label}
                      style={{
                        ...styles.colorSwatch,
                        background: `rgb(${c.rgb[0] * 255}, ${c.rgb[1] * 255}, ${c.rgb[2] * 255})`,
                        outline: active ? '2px solid #fff' : '1px solid #333',
                      }}
                    />
                  );
                })}
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
                onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                style={styles.slider}
              />
            </div>
          </section>

          <section style={styles.section}>
            <h3 style={styles.sectionTitle}>About</h3>
            <p style={styles.aboutText}>
              The checker overlays an implant NRRD on top of the defective skull NRRD using
              vtk.js volume rendering. Use the color and opacity controls to make the
              implant region stand out.
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
  colorRow: { display: 'flex', gap: '6px' },
  colorSwatch: {
    width: '28px',
    height: '28px',
    borderRadius: '4px',
    border: 'none',
    cursor: 'pointer',
  },
  aboutText: {
    fontSize: '11px',
    color: '#888',
    marginBottom: '8px',
    lineHeight: 1.5,
  },
};
