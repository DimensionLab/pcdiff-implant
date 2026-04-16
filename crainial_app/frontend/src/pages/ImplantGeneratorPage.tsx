/**
 * Implant Generator page — cran-2 cloud workspace.
 *
 * Flow:
 * - Pick a project and a defective-skull NRRD scan.
 * - Tune the cran-2 binarization threshold.
 * - Submit to the cran-2 RunPod endpoint; poll progress.
 * - Render the resulting implant NRRD overlaid on the input skull in vtk.js.
 */
import { useState, useCallback, useMemo, useEffect, type CSSProperties } from 'react';
import { useSearchParams } from 'react-router-dom';
import { AppLayout } from '../components/layout/AppLayout';
import { VtkViewport } from '../components/viewer/VtkViewport';
import {
  useGenerationJob,
  useProjectJobs,
  useCreateGenerationJob,
  useCancelJob,
} from '../hooks/useGeneration';
import { useProjects, useCreateProject } from '../hooks/useProjects';
import { useScans } from '../hooks/useScans';
import { useSettings } from '../hooks/useSettings';
import type { Scan } from '../types/scan';

const STATUS_COLORS: Record<string, string> = {
  pending: '#f59e0b',
  running: '#3b82f6',
  completed: '#10b981',
  failed: '#ef4444',
  cancelled: '#6b7280',
};

function StatusBadge({ status }: { status: string }) {
  return (
    <span style={{ ...styles.statusBadge, background: STATUS_COLORS[status] ?? '#6b7280' }}>
      {status}
    </span>
  );
}

function formatMs(ms?: number): string {
  if (!ms && ms !== 0) return '—';
  if (ms < 1000) return `${ms}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

export function ImplantGeneratorPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const jobIdFromUrl = searchParams.get('job');

  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [selectedScanId, setSelectedScanId] = useState<string | null>(null);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(jobIdFromUrl);
  const [threshold, setThreshold] = useState<number | null>(null);
  const [jobName, setJobName] = useState('');

  const [showCreateProject, setShowCreateProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');

  const { data: settings } = useSettings();
  const { data: projects = [] } = useProjects();
  const { data: projectScans = [] } = useScans(
    selectedProjectId ? { project_id: selectedProjectId, scan_category: 'defective_skull' } : undefined,
  );
  const { data: projectJobs = [] } = useProjectJobs(selectedProjectId);
  const { data: selectedJob } = useGenerationJob(selectedJobId);

  const createProject = useCreateProject();
  const createJob = useCreateGenerationJob();
  const cancelJob = useCancelJob();

  // vtk.js mounts NRRDs only — filter out anything else.
  const defectiveNrrds = useMemo<Scan[]>(
    () => projectScans.filter((s) => s.file_format?.toLowerCase() === 'nrrd'),
    [projectScans],
  );

  // Initialize threshold from settings the first time they arrive.
  useEffect(() => {
    if (threshold === null && settings?.cran2_threshold !== undefined) {
      setThreshold(settings.cran2_threshold);
    }
  }, [settings?.cran2_threshold, threshold]);

  // Keep ?job= URL param in sync with selection.
  useEffect(() => {
    const next = new URLSearchParams(searchParams);
    if (selectedJobId) next.set('job', selectedJobId);
    else next.delete('job');
    if (next.toString() !== searchParams.toString()) {
      setSearchParams(next, { replace: true });
    }
  }, [selectedJobId, searchParams, setSearchParams]);

  // When viewing a selected job, sync the project + input scan selection so the viewer
  // can show the input skull even before completion.
  useEffect(() => {
    if (!selectedJob) return;
    if (selectedJob.project_id && selectedJob.project_id !== selectedProjectId) {
      setSelectedProjectId(selectedJob.project_id);
    }
    if (selectedJob.input_scan_id && selectedJob.input_scan_id !== selectedScanId) {
      setSelectedScanId(selectedJob.input_scan_id);
    }
  }, [selectedJob, selectedProjectId, selectedScanId]);

  const cloudReady = !!(
    settings?.cloud_generation_enabled &&
    settings?.runpod_endpoint_id &&
    settings?.runpod_api_key_set
  );

  const handleCreateProject = useCallback(() => {
    const name = newProjectName.trim();
    if (!name) return;
    createProject.mutate(
      { name },
      {
        onSuccess: (project) => {
          setSelectedProjectId(project.id);
          setSelectedScanId(null);
          setSelectedJobId(null);
          setShowCreateProject(false);
          setNewProjectName('');
        },
      },
    );
  }, [newProjectName, createProject]);

  const handleGenerate = useCallback(() => {
    if (!selectedScanId || threshold === null) return;
    createJob.mutate(
      {
        input_scan_id: selectedScanId,
        project_id: selectedProjectId ?? undefined,
        threshold,
        name: jobName.trim() || undefined,
      },
      {
        onSuccess: (job) => {
          setSelectedJobId(job.id);
          setJobName('');
        },
      },
    );
  }, [selectedScanId, selectedProjectId, threshold, jobName, createJob]);

  const handleCancel = useCallback(() => {
    if (!selectedJobId) return;
    cancelJob.mutate(selectedJobId);
  }, [selectedJobId, cancelJob]);

  const isJobActive =
    selectedJob?.status === 'pending' || selectedJob?.status === 'running';

  // The viewer always shows the input skull. Once a completed job is selected,
  // overlay its implant NRRD on top.
  const viewerBaseScanId =
    selectedJob?.input_scan_id ?? selectedScanId ?? null;
  const viewerOverlayScanId =
    selectedJob?.status === 'completed' ? selectedJob.output_scan_id ?? null : null;

  return (
    <AppLayout
      sidebar={
        <div style={styles.sidebar}>
          <section style={styles.section}>
            <h3 style={styles.sectionTitle}>Project</h3>
            <div style={styles.selectRow}>
              <select
                value={selectedProjectId ?? ''}
                onChange={(e) => {
                  setSelectedProjectId(e.target.value || null);
                  setSelectedScanId(null);
                  setSelectedJobId(null);
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
              <button
                onClick={() => setShowCreateProject((v) => !v)}
                style={styles.addButton}
                title="New project"
              >
                +
              </button>
            </div>
            {showCreateProject && (
              <div style={styles.createRow}>
                <input
                  type="text"
                  value={newProjectName}
                  onChange={(e) => setNewProjectName(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleCreateProject()}
                  placeholder="New project name"
                  style={styles.input}
                />
                <button onClick={handleCreateProject} style={styles.createButton}>
                  Create
                </button>
              </div>
            )}
          </section>

          {selectedProjectId && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Defective Skull (NRRD)</h3>
              {defectiveNrrds.length === 0 ? (
                <p style={styles.emptyText}>
                  No defective-skull NRRDs in this project. Import one in the Data Viewer.
                </p>
              ) : (
                <>
                  <select
                    value={selectedScanId ?? ''}
                    onChange={(e) => {
                      setSelectedScanId(e.target.value || null);
                      setSelectedJobId(null);
                    }}
                    style={styles.select}
                  >
                    <option value="">Select scan…</option>
                    {defectiveNrrds.map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.name} {s.defect_type ? `· ${s.defect_type}` : '· (no defect_type)'}
                      </option>
                    ))}
                  </select>
                  {selectedScanId &&
                    !defectiveNrrds.find((s) => s.id === selectedScanId)?.defect_type && (
                      <p style={styles.warnText}>
                        This scan has no <code>defect_type</code>. cran-2 v3 needs one of
                        bilateral / frontoorbital / parietotemporal / random_1 / random_2.
                        Set it on the scan in the Data Viewer before generating.
                      </p>
                    )}
                </>
              )}
            </section>
          )}

          {selectedProjectId && projectJobs.length > 0 && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Recent Generations</h3>
              <div style={styles.jobList}>
                {projectJobs.slice(0, 12).map((job) => (
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
                      <StatusBadge status={job.status} />
                      <span style={styles.jobItemTime}>
                        {new Date(job.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </section>
          )}
        </div>
      }
      main={
        <div style={styles.mainViewport}>
          {viewerBaseScanId ? (
            <>
              <VtkViewport
                scanId={viewerBaseScanId}
                overlayScanId={viewerOverlayScanId}
                overlayColor={[1.0, 0.2, 0.25]}
                overlayOpacity={0.75}
              />
              {selectedJob && (
                <div style={styles.statusOverlay}>
                  <div style={styles.statusHeader}>
                    <span style={styles.statusJobName}>{selectedJob.name}</span>
                    <StatusBadge status={selectedJob.status} />
                  </div>
                  {isJobActive && (
                    <>
                      <div style={styles.progressBar}>
                        <div
                          style={{
                            ...styles.progressFill,
                            width: `${selectedJob.progress_percent}%`,
                          }}
                        />
                      </div>
                      <div style={styles.progressText}>
                        {selectedJob.progress_percent}% — {selectedJob.current_step ?? 'Queued…'}
                      </div>
                    </>
                  )}
                  {selectedJob.status === 'failed' && selectedJob.error_message && (
                    <div style={styles.errorText}>{selectedJob.error_message}</div>
                  )}
                  {selectedJob.status === 'completed' && (
                    <div style={styles.successText}>
                      Implant rendered as red overlay. Generation took{' '}
                      {formatMs(selectedJob.generation_time_ms)}.
                    </div>
                  )}
                </div>
              )}
            </>
          ) : (
            <div style={styles.placeholder}>
              <h2 style={styles.placeholderTitle}>cran-2 Implant Generator</h2>
              <p style={styles.placeholderText}>
                Select a project and defective-skull NRRD to generate a cranial implant.
              </p>
              {!cloudReady && (
                <p style={styles.warningText}>
                  Cloud generation is not configured. Open Settings to enable it and add your
                  RunPod credentials.
                </p>
              )}
            </div>
          )}
        </div>
      }
      controls={
        <div style={styles.controlsPanel}>
          <section style={styles.section}>
            <h3 style={styles.sectionTitle}>cran-2 Settings</h3>

            <div style={styles.formGroup}>
              <label style={styles.label}>Job Name (optional)</label>
              <input
                type="text"
                value={jobName}
                onChange={(e) => setJobName(e.target.value)}
                placeholder="e.g. Patient 12 — pass 1"
                style={styles.input}
                disabled={isJobActive}
              />
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>
                Threshold: {(threshold ?? 0.5).toFixed(2)}
              </label>
              <input
                type="range"
                min={0.05}
                max={0.95}
                step={0.05}
                value={threshold ?? 0.5}
                onChange={(e) => setThreshold(Number(e.target.value))}
                style={styles.slider}
                disabled={isJobActive}
              />
              <p style={styles.hint}>
                Cutoff applied to cran-2's predicted implant probability map.
              </p>
            </div>

            {!cloudReady && (
              <div style={styles.warningBox}>
                Cloud generation must be enabled in Settings (RunPod endpoint + API key).
              </div>
            )}

            <button
              onClick={handleGenerate}
              disabled={
                !selectedScanId ||
                threshold === null ||
                !cloudReady ||
                isJobActive ||
                createJob.isPending ||
                !defectiveNrrds.find((s) => s.id === selectedScanId)?.defect_type
              }
              style={{
                ...styles.generateButton,
                opacity:
                  !selectedScanId ||
                  !cloudReady ||
                  isJobActive ||
                  createJob.isPending ||
                  !defectiveNrrds.find((s) => s.id === selectedScanId)?.defect_type
                    ? 0.5
                    : 1,
              }}
            >
              {createJob.isPending ? 'Submitting…' : 'Generate Implant'}
            </button>

            {isJobActive && (
              <button
                onClick={handleCancel}
                disabled={cancelJob.isPending}
                style={styles.cancelButton}
              >
                {cancelJob.isPending ? 'Cancelling…' : 'Cancel Job'}
              </button>
            )}

            {createJob.isError && (
              <div style={styles.errorBox}>
                Failed to start job: {(createJob.error as Error).message}
              </div>
            )}
          </section>

          {selectedJob && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Job Details</h3>
              <DetailRow label="Status" value={<StatusBadge status={selectedJob.status} />} />
              <DetailRow label="Threshold" value={selectedJob.threshold.toFixed(2)} />
              {selectedJob.runpod_job_id && (
                <DetailRow
                  label="RunPod ID"
                  value={
                    <span style={styles.monoValue}>
                      {selectedJob.runpod_job_id.slice(0, 12)}…
                    </span>
                  }
                />
              )}
              {selectedJob.inference_time_ms !== undefined && (
                <DetailRow label="GPU time" value={formatMs(selectedJob.inference_time_ms)} />
              )}
              {selectedJob.generation_time_ms !== undefined && (
                <DetailRow label="Total time" value={formatMs(selectedJob.generation_time_ms)} />
              )}
              {selectedJob.output_scan_id && (
                <DetailRow
                  label="Implant scan"
                  value={
                    <span style={styles.monoValue}>
                      {selectedJob.output_scan_id.slice(0, 8)}…
                    </span>
                  }
                />
              )}
              {selectedJob.started_at && (
                <DetailRow
                  label="Started"
                  value={new Date(selectedJob.started_at).toLocaleTimeString()}
                />
              )}
              {selectedJob.completed_at && (
                <DetailRow
                  label="Completed"
                  value={new Date(selectedJob.completed_at).toLocaleTimeString()}
                />
              )}
            </section>
          )}

          {!selectedJob && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>About cran-2</h3>
              <p style={styles.aboutText}>
                cran-2 is DimensionLab's cranial-implant model. Generation runs on a RunPod
                serverless GPU; the resulting implant NRRD is downloaded from S3 and registered
                as a new scan.
              </p>
              <p style={styles.aboutText}>
                Lower thresholds keep more voxels (thicker implant); higher thresholds tighten
                the mask.
              </p>
            </section>
          )}
        </div>
      }
    />
  );
}

function DetailRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={styles.detailRow}>
      <span style={styles.detailLabel}>{label}</span>
      <span style={styles.detailValue}>{value}</span>
    </div>
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
  selectRow: { display: 'flex', gap: '4px' },
  select: {
    flex: 1,
    background: '#111',
    color: '#ccc',
    border: '1px solid #333',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
  },
  addButton: {
    padding: '6px 12px',
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    fontSize: '14px',
    fontWeight: 600,
    cursor: 'pointer',
  },
  createRow: { display: 'flex', gap: '4px', marginTop: '8px' },
  input: {
    flex: 1,
    background: '#111',
    color: '#ccc',
    border: '1px solid #333',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
    boxSizing: 'border-box',
    width: '100%',
  },
  createButton: {
    padding: '6px 12px',
    background: '#10b981',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    fontSize: '11px',
    fontWeight: 600,
    cursor: 'pointer',
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
  warningText: {
    marginTop: '16px',
    fontSize: '13px',
    color: '#f59e0b',
    maxWidth: '400px',
  },
  statusOverlay: {
    position: 'absolute',
    top: '16px',
    left: '16px',
    right: '16px',
    background: 'rgba(13, 13, 32, 0.92)',
    border: '1px solid #333',
    borderRadius: '8px',
    padding: '12px 16px',
    color: '#eee',
    backdropFilter: 'blur(4px)',
  },
  statusHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '12px',
    marginBottom: '8px',
  },
  statusJobName: {
    fontSize: '14px',
    fontWeight: 600,
    color: '#fff',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  progressBar: {
    width: '100%',
    height: '6px',
    background: '#222',
    borderRadius: '3px',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    background: '#3b82f6',
    transition: 'width 0.3s',
  },
  progressText: { fontSize: '12px', color: '#aaa', marginTop: '6px' },
  errorText: { fontSize: '12px', color: '#ef4444', marginTop: '4px' },
  successText: { fontSize: '12px', color: '#10b981', marginTop: '4px' },
  controlsPanel: { display: 'flex', flexDirection: 'column' },
  formGroup: { display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '12px' },
  label: { fontSize: '12px', fontWeight: 500, color: '#aaa' },
  slider: { width: '100%', accentColor: '#2563eb' },
  hint: { fontSize: '11px', color: '#666', margin: 0 },
  warnText: {
    marginTop: '8px',
    fontSize: '11px',
    color: '#f59e0b',
    lineHeight: 1.4,
  },
  warningBox: {
    padding: '8px 10px',
    background: 'rgba(245, 158, 11, 0.1)',
    border: '1px solid rgba(245, 158, 11, 0.3)',
    borderRadius: '6px',
    fontSize: '11px',
    color: '#f59e0b',
    marginBottom: '12px',
  },
  errorBox: {
    marginTop: '8px',
    padding: '8px 10px',
    background: 'rgba(239, 68, 68, 0.1)',
    border: '1px solid rgba(239, 68, 68, 0.3)',
    borderRadius: '6px',
    fontSize: '11px',
    color: '#ef4444',
  },
  generateButton: {
    width: '100%',
    padding: '10px 16px',
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '13px',
    fontWeight: 600,
    cursor: 'pointer',
  },
  cancelButton: {
    width: '100%',
    marginTop: '8px',
    padding: '8px 16px',
    background: '#374151',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '12px',
    cursor: 'pointer',
  },
  detailRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px',
    gap: '12px',
  },
  detailLabel: { fontSize: '11px', color: '#888' },
  detailValue: { fontSize: '12px', color: '#fff' },
  monoValue: { fontFamily: 'monospace', fontSize: '11px' },
  aboutText: {
    fontSize: '11px',
    color: '#888',
    marginBottom: '8px',
    lineHeight: 1.5,
  },
};
