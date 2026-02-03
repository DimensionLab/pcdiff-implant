/**
 * Implant Generator page — workspace for generating cranial implants.
 *
 * Users can:
 * - Select a defective skull from a workspace/project
 * - Configure generation settings (DDIM/DDPM, steps, ensemble count)
 * - Start generation and track progress
 * - View results and select the best output
 * - Navigate to Implant Checker to visualize and compare
 */
import { useState, useCallback, useMemo, type CSSProperties } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { AppLayout } from '../components/layout/AppLayout';
import {
  useGenerationJob,
  useProjectJobs,
  useCreateGenerationJob,
  useCancelJob,
  useSelectOutput,
  useDeleteUnselectedOutputs,
} from '../hooks/useGeneration';
import { useProjects, useCreateProject } from '../hooks/useProjects';
import { useSettings } from '../hooks/useSettings';
import { pointCloudApi } from '../services/point-cloud-api';
import type { GenerationJob, PcdiffModel } from '../types/generation';

export function ImplantGeneratorPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const jobIdFromUrl = searchParams.get('job');

  // Project/workspace state
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [selectedInputPcId, setSelectedInputPcId] = useState<string | null>(null);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(jobIdFromUrl);

  // Generation settings
  const [samplingMethod, setSamplingMethod] = useState<'ddim' | 'ddpm'>('ddim');
  const [samplingSteps, setSamplingSteps] = useState(50);
  const [numEnsemble, setNumEnsemble] = useState(5);
  const [jobName, setJobName] = useState('');
  const [useCloud, setUseCloud] = useState<boolean | null>(null); // null = use default from settings
  const [pcdiffModel, setPcdiffModel] = useState<PcdiffModel>('best');

  // Project creation
  const [showCreateProject, setShowCreateProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');

  // Settings for cloud generation
  const { data: appSettings } = useSettings();

  // Queries
  const { data: projects = [] } = useProjects();
  const { data: projectJobs = [] } = useProjectJobs(selectedProjectId);
  const { data: selectedJob } = useGenerationJob(selectedJobId);

  // Point clouds for selected project
  const { data: projectPointClouds = [] } = useQuery({
    queryKey: ['project-point-clouds', selectedProjectId],
    queryFn: () => pointCloudApi.list({ project_id: selectedProjectId! }),
    enabled: !!selectedProjectId,
  });

  // Filter for defective skulls only
  const defectiveSkulls = useMemo(() => {
    return projectPointClouds.filter(
      (pc) => pc.scan_category === 'defective_skull' && pc.file_format === 'npy',
    );
  }, [projectPointClouds]);

  // Mutations
  const createProject = useCreateProject();
  const createJob = useCreateGenerationJob();
  const cancelJob = useCancelJob();
  const selectOutput = useSelectOutput();
  const deleteUnselected = useDeleteUnselectedOutputs();

  // Handlers
  const handleCreateProject = useCallback(() => {
    if (!newProjectName.trim()) return;
    createProject.mutate(
      { name: newProjectName.trim() },
      {
        onSuccess: (project) => {
          setSelectedProjectId(project.id);
          setShowCreateProject(false);
          setNewProjectName('');
        },
      },
    );
  }, [newProjectName, createProject]);

  const handleStartGeneration = useCallback(() => {
    if (!selectedProjectId || !selectedInputPcId) return;

    createJob.mutate(
      {
        project_id: selectedProjectId,
        input_pc_id: selectedInputPcId,
        sampling_method: samplingMethod,
        sampling_steps: samplingSteps,
        num_ensemble: numEnsemble,
        name: jobName.trim() || undefined,
        use_cloud: useCloud ?? undefined, // null means use default from settings
        pcdiff_model: pcdiffModel,
      },
      {
        onSuccess: (job) => {
          setSelectedJobId(job.id);
          setSearchParams({ job: job.id });
        },
      },
    );
  }, [
    selectedProjectId,
    selectedInputPcId,
    samplingMethod,
    samplingSteps,
    numEnsemble,
    jobName,
    useCloud,
    pcdiffModel,
    createJob,
    setSearchParams,
  ]);

  // Determine if cloud will be used (for display purposes)
  const willUseCloud = useCloud ?? appSettings?.cloud_generation_enabled ?? false;
  const cloudConfigured = appSettings?.runpod_api_key_set && appSettings?.runpod_endpoint_id;

  const handleCancelJob = useCallback(() => {
    if (!selectedJobId) return;
    cancelJob.mutate(selectedJobId);
  }, [selectedJobId, cancelJob]);

  const handleSelectOutput = useCallback(
    (outputId: string) => {
      if (!selectedJobId) return;
      selectOutput.mutate({ jobId: selectedJobId, outputId });
    },
    [selectedJobId, selectOutput],
  );

  const handleDeleteUnselected = useCallback(() => {
    if (!selectedJobId) return;
    deleteUnselected.mutate(selectedJobId);
  }, [selectedJobId, deleteUnselected]);

  const handleViewInChecker = useCallback(() => {
    navigate('/checker');
  }, [navigate]);

  // Estimate generation time
  const estimatedTime = useMemo(() => {
    // Cloud GPU is ~10x faster than local CPU
    const cloudMultiplier = willUseCloud ? 0.1 : 1;
    // Rough estimates based on method and steps
    const baseTime = (samplingMethod === 'ddim' ? 30 : 120) * cloudMultiplier; // seconds
    const stepFactor = samplingSteps / (samplingMethod === 'ddim' ? 50 : 1000);
    const ensembleFactor = numEnsemble;
    const total = baseTime * stepFactor * ensembleFactor;
    if (total < 60) return `~${Math.round(total)}s`;
    return `~${Math.round(total / 60)}min`;
  }, [samplingMethod, samplingSteps, numEnsemble, willUseCloud]);

  // Render helpers
  const renderJobStatus = (job: GenerationJob) => {
    const statusColors: Record<string, string> = {
      pending: '#f59e0b',
      running: '#3b82f6',
      completed: '#10b981',
      failed: '#ef4444',
      cancelled: '#6b7280',
    };

    return (
      <span
        style={{
          ...styles.statusBadge,
          background: statusColors[job.status] ?? '#6b7280',
        }}
      >
        {job.status}
      </span>
    );
  };

  return (
    <AppLayout
      sidebar={
        <div style={styles.sidebar}>
          {/* Project Selection */}
          <section style={styles.section}>
            <h3 style={styles.sectionTitle}>Workspace</h3>
            <div style={styles.selectRow}>
              <select
                value={selectedProjectId ?? ''}
                onChange={(e) => {
                  setSelectedProjectId(e.target.value || null);
                  setSelectedInputPcId(null);
                  setSelectedJobId(null);
                }}
                style={styles.select}
              >
                <option value="">Select project...</option>
                {projects.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
              <button
                onClick={() => setShowCreateProject(!showCreateProject)}
                style={styles.addButton}
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

          {/* Defective Skull Selection */}
          {selectedProjectId && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Input Defective Skull</h3>
              {defectiveSkulls.length === 0 ? (
                <div style={styles.emptyState}>
                  <p style={styles.emptyText}>No defective skulls in this project</p>
                  <button
                    onClick={() => navigate(`/?project=${selectedProjectId}`)}
                    style={styles.importButton}
                  >
                    📁 Go to Data Viewer
                  </button>
                  <p style={styles.hintText}>
                    In Data Viewer, use "+ Add" to register a defective skull .npy file for this project.
                  </p>
                </div>
              ) : (
                <select
                  value={selectedInputPcId ?? ''}
                  onChange={(e) => setSelectedInputPcId(e.target.value || null)}
                  style={styles.select}
                >
                  <option value="">Select defective skull...</option>
                  {defectiveSkulls.map((pc) => (
                    <option key={pc.id} value={pc.id}>
                      {pc.name}
                    </option>
                  ))}
                </select>
              )}
            </section>
          )}

          {/* Previous Jobs */}
          {selectedProjectId && projectJobs.length > 0 && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Previous Generations</h3>
              <div style={styles.jobList}>
                {projectJobs.slice(0, 10).map((job) => (
                  <div
                    key={job.id}
                    onClick={() => {
                      setSelectedJobId(job.id);
                      setSearchParams({ job: job.id });
                    }}
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
                      {renderJobStatus(job)}
                      <span style={styles.jobItemTime}>
                        {new Date(job.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      }
      main={
        <div style={styles.main}>
          {!selectedProjectId && (
            <div style={styles.placeholder}>
              <h2 style={styles.placeholderTitle}>Implant Generator</h2>
              <p style={styles.placeholderText}>
                Select a workspace to get started with implant generation.
              </p>
            </div>
          )}

          {selectedProjectId && !selectedJobId && !selectedInputPcId && (
            <div style={styles.placeholder}>
              <h2 style={styles.placeholderTitle}>Select Input</h2>
              <p style={styles.placeholderText}>
                Choose a defective skull point cloud from the sidebar to configure
                generation settings.
              </p>
            </div>
          )}

          {/* Settings Form */}
          {selectedProjectId && selectedInputPcId && !selectedJobId && (
            <div style={styles.settingsPanel}>
              <h2 style={styles.settingsTitle}>Generation Settings</h2>

              <div style={styles.settingsForm}>
                <div style={styles.formGroup}>
                  <label style={styles.label}>Job Name (optional)</label>
                  <input
                    type="text"
                    value={jobName}
                    onChange={(e) => setJobName(e.target.value)}
                    placeholder="My Implant Generation"
                    style={styles.input}
                  />
                </div>

                <div style={styles.formGroup}>
                  <label style={styles.label}>Sampling Method</label>
                  <div style={styles.radioGroup}>
                    <label style={styles.radioLabel}>
                      <input
                        type="radio"
                        name="method"
                        checked={samplingMethod === 'ddim'}
                        onChange={() => {
                          setSamplingMethod('ddim');
                          setSamplingSteps(50);
                        }}
                      />
                      <span>DDIM (Fast)</span>
                    </label>
                    <label style={styles.radioLabel}>
                      <input
                        type="radio"
                        name="method"
                        checked={samplingMethod === 'ddpm'}
                        onChange={() => {
                          setSamplingMethod('ddpm');
                          setSamplingSteps(1000);
                        }}
                      />
                      <span>DDPM (High Quality)</span>
                    </label>
                  </div>
                </div>

                <div style={styles.formGroup}>
                  <label style={styles.label}>
                    Sampling Steps: {samplingSteps}
                  </label>
                  <input
                    type="range"
                    min={samplingMethod === 'ddim' ? 25 : 100}
                    max={samplingMethod === 'ddim' ? 100 : 1000}
                    step={samplingMethod === 'ddim' ? 5 : 50}
                    value={samplingSteps}
                    onChange={(e) => setSamplingSteps(Number(e.target.value))}
                    style={styles.slider}
                  />
                </div>

                <div style={styles.formGroup}>
                  <label style={styles.label}>Ensemble Count: {numEnsemble}</label>
                  <input
                    type="range"
                    min={1}
                    max={5}
                    value={numEnsemble}
                    onChange={(e) => setNumEnsemble(Number(e.target.value))}
                    style={styles.slider}
                  />
                  <p style={styles.hint}>
                    Generate multiple implants to compare and select the best one.
                  </p>
                </div>

                {/* Model Selection */}
                <div style={styles.formGroup}>
                  <label style={styles.label}>PCDiff Model</label>
                  <div style={styles.radioGroup}>
                    <label style={styles.radioLabel}>
                      <input
                        type="radio"
                        name="pcdiff_model"
                        checked={pcdiffModel === 'best'}
                        onChange={() => setPcdiffModel('best')}
                      />
                      <span>Best (Recommended)</span>
                    </label>
                    <label style={styles.radioLabel}>
                      <input
                        type="radio"
                        name="pcdiff_model"
                        checked={pcdiffModel === 'latest'}
                        onChange={() => setPcdiffModel('latest')}
                      />
                      <span>Latest (Experimental)</span>
                    </label>
                  </div>
                  <p style={styles.hint}>
                    "Best" uses the highest-performing checkpoint. "Latest" uses the most recently trained model.
                  </p>
                </div>

                {/* Cloud Generation Toggle */}
                <div style={styles.cloudToggle}>
                  <label style={styles.cloudToggleLabel}>
                    <input
                      type="checkbox"
                      checked={willUseCloud}
                      onChange={(e) => setUseCloud(e.target.checked)}
                      disabled={!cloudConfigured}
                      style={styles.checkbox}
                    />
                    <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      ☁️ Use Cloud GPU
                      {willUseCloud && <span style={styles.cloudBadge}>~10x faster</span>}
                    </span>
                  </label>
                  {!cloudConfigured && (
                    <p style={styles.cloudHint}>
                      Configure Runpod in Settings to enable cloud generation.
                    </p>
                  )}
                </div>

                <div style={styles.estimate}>
                  <span>Estimated time: </span>
                  <strong>{estimatedTime}</strong>
                  {willUseCloud && <span style={styles.cloudIndicator}> (Cloud GPU)</span>}
                </div>

                <button
                  onClick={handleStartGeneration}
                  disabled={createJob.isPending}
                  style={{
                    ...styles.generateButton,
                    background: willUseCloud ? '#8b5cf6' : '#2563eb',
                  }}
                >
                  {createJob.isPending ? 'Starting...' : willUseCloud ? '☁️ Generate on Cloud' : 'Generate Implant'}
                </button>
              </div>
            </div>
          )}

          {/* Job Progress View */}
          {selectedJob && (selectedJob.status === 'pending' || selectedJob.status === 'running') && (
            <div style={styles.progressPanel}>
              <h2 style={styles.progressTitle}>{selectedJob.name}</h2>
              {renderJobStatus(selectedJob)}

              <div style={styles.progressBar}>
                <div
                  style={{
                    ...styles.progressFill,
                    width: `${selectedJob.progress_percent}%`,
                  }}
                />
              </div>
              <div style={styles.progressText}>
                {selectedJob.progress_percent}% - {selectedJob.current_step ?? 'Starting...'}
              </div>

              <button onClick={handleCancelJob} style={styles.cancelButton}>
                Cancel
              </button>
            </div>
          )}

          {/* Results View */}
          {selectedJob && selectedJob.status === 'completed' && (
            <div style={styles.resultsPanel}>
              <h2 style={styles.resultsTitle}>{selectedJob.name}</h2>
              {renderJobStatus(selectedJob)}

              <div style={styles.statsRow}>
                <div style={styles.stat}>
                  <span style={styles.statValue}>
                    {selectedJob.output_pc_ids.length}
                  </span>
                  <span style={styles.statLabel}>Implants Generated</span>
                </div>
                <div style={styles.stat}>
                  <span style={styles.statValue}>
                    {selectedJob.generation_time_ms
                      ? `${(selectedJob.generation_time_ms / 1000).toFixed(1)}s`
                      : '-'}
                  </span>
                  <span style={styles.statLabel}>Generation Time</span>
                </div>
              </div>

              <h3 style={styles.subsectionTitle}>Ensemble Results</h3>
              <div style={styles.outputGrid}>
                {selectedJob.output_pc_ids.map((pcId, idx) => (
                  <div
                    key={pcId}
                    onClick={() => handleSelectOutput(pcId)}
                    style={{
                      ...styles.outputCard,
                      borderColor:
                        selectedJob.selected_output_id === pcId
                          ? '#10b981'
                          : '#333',
                    }}
                  >
                    <div style={styles.outputIndex}>#{idx + 1}</div>
                    {selectedJob.selected_output_id === pcId && (
                      <div style={styles.selectedBadge}>Selected</div>
                    )}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleSelectOutput(pcId);
                      }}
                      style={styles.selectButton}
                    >
                      {selectedJob.selected_output_id === pcId
                        ? 'Selected'
                        : 'Select'}
                    </button>
                  </div>
                ))}
              </div>

              <div style={styles.actionButtons}>
                <button onClick={handleViewInChecker} style={styles.viewButton}>
                  View in Implant Checker
                </button>
                {selectedJob.selected_output_id &&
                  selectedJob.output_pc_ids.length > 1 && (
                    <button
                      onClick={handleDeleteUnselected}
                      style={styles.deleteButton}
                    >
                      Delete Unselected
                    </button>
                  )}
              </div>
            </div>
          )}

          {/* Failed View */}
          {selectedJob && selectedJob.status === 'failed' && (
            <div style={styles.errorPanel}>
              <h2 style={styles.errorTitle}>Generation Failed</h2>
              <p style={styles.errorMessage}>{selectedJob.error_message}</p>
              <button
                onClick={() => setSelectedJobId(null)}
                style={styles.retryButton}
              >
                Try Again
              </button>
            </div>
          )}
        </div>
      }
      controls={
        <div style={styles.controls}>
          {selectedJob && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Job Details</h3>
              <div style={styles.detailRow}>
                <span style={styles.detailLabel}>Status</span>
                {renderJobStatus(selectedJob)}
              </div>
              <div style={styles.detailRow}>
                <span style={styles.detailLabel}>Method</span>
                <span style={styles.detailValue}>
                  {selectedJob.sampling_method.toUpperCase()}
                </span>
              </div>
              <div style={styles.detailRow}>
                <span style={styles.detailLabel}>Steps</span>
                <span style={styles.detailValue}>{selectedJob.sampling_steps}</span>
              </div>
              <div style={styles.detailRow}>
                <span style={styles.detailLabel}>Ensemble</span>
                <span style={styles.detailValue}>{selectedJob.num_ensemble}</span>
              </div>
              {selectedJob.started_at && (
                <div style={styles.detailRow}>
                  <span style={styles.detailLabel}>Started</span>
                  <span style={styles.detailValue}>
                    {new Date(selectedJob.started_at).toLocaleTimeString()}
                  </span>
                </div>
              )}
              {selectedJob.completed_at && (
                <div style={styles.detailRow}>
                  <span style={styles.detailLabel}>Completed</span>
                  <span style={styles.detailValue}>
                    {new Date(selectedJob.completed_at).toLocaleTimeString()}
                  </span>
                </div>
              )}
            </section>
          )}

          {!selectedJob && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>About</h3>
              <p style={styles.aboutText}>
                The Implant Generator uses a PCDiff diffusion model to generate
                patient-specific cranial implants from defective skull point clouds.
              </p>
              <p style={styles.aboutText}>
                <strong>DDIM</strong> is faster (~30s per sample) while{' '}
                <strong>DDPM</strong> produces higher quality results (~2min per
                sample).
              </p>
              <p style={styles.aboutText}>
                Ensemble generation creates multiple variations, allowing you to
                select the best result.
              </p>
            </section>
          )}
        </div>
      }
    />
  );
}

const styles: Record<string, CSSProperties> = {
  sidebar: {
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
  sectionTitle: {
    margin: '0 0 8px',
    fontSize: '11px',
    fontWeight: 600,
    color: '#aaa',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  selectRow: {
    display: 'flex',
    gap: '4px',
  },
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
  createRow: {
    display: 'flex',
    gap: '4px',
    marginTop: '8px',
  },
  input: {
    flex: 1,
    background: '#111',
    color: '#ccc',
    border: '1px solid #333',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '12px',
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
  emptyState: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  importButton: {
    padding: '8px 12px',
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '12px',
    fontWeight: 500,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '6px',
  },
  hintText: {
    color: '#555',
    fontSize: '10px',
    margin: 0,
    lineHeight: 1.4,
  },
  jobList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  jobItem: {
    padding: '8px',
    borderRadius: '4px',
    cursor: 'pointer',
    transition: 'background 0.2s',
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
  jobItemTime: {
    fontSize: '10px',
    color: '#666',
  },
  statusBadge: {
    padding: '2px 6px',
    borderRadius: '4px',
    fontSize: '9px',
    fontWeight: 600,
    textTransform: 'uppercase',
    color: '#fff',
  },
  main: {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '24px',
  },
  placeholder: {
    textAlign: 'center',
    maxWidth: '400px',
  },
  placeholderTitle: {
    fontSize: '24px',
    fontWeight: 600,
    color: '#fff',
    marginBottom: '8px',
  },
  placeholderText: {
    fontSize: '14px',
    color: '#888',
  },
  settingsPanel: {
    width: '100%',
    maxWidth: '500px',
    padding: '24px',
    background: '#111',
    borderRadius: '12px',
    border: '1px solid #333',
  },
  settingsTitle: {
    fontSize: '20px',
    fontWeight: 600,
    color: '#fff',
    marginBottom: '20px',
  },
  settingsForm: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  formGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  label: {
    fontSize: '12px',
    fontWeight: 500,
    color: '#aaa',
  },
  radioGroup: {
    display: 'flex',
    gap: '16px',
  },
  radioLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '13px',
    color: '#ccc',
    cursor: 'pointer',
  },
  slider: {
    width: '100%',
    accentColor: '#2563eb',
  },
  hint: {
    fontSize: '11px',
    color: '#666',
    margin: 0,
  },
  estimate: {
    padding: '12px',
    background: 'rgba(37, 99, 235, 0.1)',
    borderRadius: '6px',
    fontSize: '13px',
    color: '#ccc',
  },
  cloudToggle: {
    padding: '12px',
    background: 'rgba(139, 92, 246, 0.1)',
    borderRadius: '6px',
    border: '1px solid rgba(139, 92, 246, 0.2)',
  },
  cloudToggleLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '13px',
    color: '#ccc',
    cursor: 'pointer',
  },
  checkbox: {
    width: '16px',
    height: '16px',
    accentColor: '#8b5cf6',
  },
  cloudBadge: {
    padding: '2px 6px',
    background: 'rgba(139, 92, 246, 0.3)',
    borderRadius: '4px',
    fontSize: '10px',
    color: '#a78bfa',
    fontWeight: 600,
  },
  cloudHint: {
    fontSize: '10px',
    color: '#666',
    margin: '6px 0 0',
  },
  cloudIndicator: {
    color: '#a78bfa',
    fontSize: '12px',
  },
  generateButton: {
    padding: '12px 24px',
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '14px',
    fontWeight: 600,
    cursor: 'pointer',
    marginTop: '8px',
  },
  progressPanel: {
    textAlign: 'center',
    width: '100%',
    maxWidth: '500px',
  },
  progressTitle: {
    fontSize: '20px',
    fontWeight: 600,
    color: '#fff',
    marginBottom: '8px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    maxWidth: '100%',
  },
  progressBar: {
    width: '100%',
    height: '8px',
    background: '#333',
    borderRadius: '4px',
    marginTop: '16px',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    background: '#2563eb',
    borderRadius: '4px',
    transition: 'width 0.3s',
  },
  progressText: {
    fontSize: '13px',
    color: '#aaa',
    marginTop: '8px',
  },
  cancelButton: {
    marginTop: '16px',
    padding: '8px 16px',
    background: '#374151',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '12px',
    cursor: 'pointer',
  },
  resultsPanel: {
    width: '100%',
    maxWidth: '600px',
    textAlign: 'center',
  },
  resultsTitle: {
    fontSize: '24px',
    fontWeight: 600,
    color: '#fff',
    marginBottom: '8px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    maxWidth: '100%',
  },
  statsRow: {
    display: 'flex',
    justifyContent: 'center',
    gap: '32px',
    marginTop: '20px',
    marginBottom: '24px',
  },
  stat: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  statValue: {
    fontSize: '28px',
    fontWeight: 700,
    color: '#fff',
  },
  statLabel: {
    fontSize: '11px',
    color: '#888',
    marginTop: '4px',
  },
  subsectionTitle: {
    fontSize: '14px',
    fontWeight: 600,
    color: '#aaa',
    marginBottom: '12px',
  },
  outputGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(100px, 1fr))',
    gap: '12px',
    marginBottom: '24px',
  },
  outputCard: {
    padding: '16px',
    background: '#111',
    borderRadius: '8px',
    border: '2px solid #333',
    cursor: 'pointer',
    transition: 'border-color 0.2s',
    position: 'relative',
  },
  outputIndex: {
    fontSize: '18px',
    fontWeight: 700,
    color: '#fff',
    marginBottom: '8px',
  },
  selectedBadge: {
    position: 'absolute',
    top: '8px',
    right: '8px',
    padding: '2px 6px',
    background: '#10b981',
    borderRadius: '4px',
    fontSize: '9px',
    fontWeight: 600,
    color: '#fff',
  },
  selectButton: {
    padding: '6px 12px',
    background: 'transparent',
    color: '#aaa',
    border: '1px solid #444',
    borderRadius: '4px',
    fontSize: '11px',
    cursor: 'pointer',
  },
  actionButtons: {
    display: 'flex',
    justifyContent: 'center',
    gap: '12px',
  },
  viewButton: {
    padding: '12px 24px',
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '14px',
    fontWeight: 600,
    cursor: 'pointer',
  },
  deleteButton: {
    padding: '12px 24px',
    background: '#374151',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '14px',
    cursor: 'pointer',
  },
  errorPanel: {
    textAlign: 'center',
    maxWidth: '400px',
  },
  errorTitle: {
    fontSize: '20px',
    fontWeight: 600,
    color: '#ef4444',
    marginBottom: '8px',
  },
  errorMessage: {
    fontSize: '13px',
    color: '#aaa',
    marginBottom: '16px',
  },
  retryButton: {
    padding: '10px 20px',
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '13px',
    cursor: 'pointer',
  },
  controls: {
    padding: '12px',
  },
  detailRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px',
  },
  detailLabel: {
    fontSize: '11px',
    color: '#888',
  },
  detailValue: {
    fontSize: '12px',
    color: '#fff',
  },
  aboutText: {
    fontSize: '11px',
    color: '#888',
    marginBottom: '8px',
    lineHeight: '1.5',
  },
};
