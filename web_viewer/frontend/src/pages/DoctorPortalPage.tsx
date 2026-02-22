/**
 * Doctor Portal page - comprehensive patient and case management.
 *
 * This page provides:
 * - Patient list with search and CRUD
 * - Project/case management linked to patients
 * - Scans and implants overview
 * - AI-powered report generation
 */
import { useState, type CSSProperties } from 'react';
import { useNavigate } from 'react-router-dom';
import { AppLayout } from '../components/layout/AppLayout';
import { PatientList } from '../components/doctor-portal/PatientList';
import { PatientForm } from '../components/doctor-portal/PatientForm';
import { ProjectDetail } from '../components/doctor-portal/ProjectDetail';
import { ReportPanel } from '../components/doctor-portal/ReportPanel';
import { usePatient, usePatientProjects } from '../hooks/usePatients';
import { useProject, useCreateProject } from '../hooks/useProjects';
import type { Patient } from '../types/patient';
import type { Project, ProjectCreate } from '../types/project';

export function DoctorPortalPage() {
  const navigate = useNavigate();

  // Selection state
  const [selectedPatientId, setSelectedPatientId] = useState<string | null>(null);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);

  // Dialog state
  const [showPatientForm, setShowPatientForm] = useState(false);
  const [editingPatient, setEditingPatient] = useState<Patient | null>(null);
  const [showCreateProject, setShowCreateProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');

  // Data queries
  const { data: selectedPatient } = usePatient(selectedPatientId);
  const { data: patientProjects = [] } = usePatientProjects(selectedPatientId);
  const { data: selectedProject } = useProject(selectedProjectId);
  const createProject = useCreateProject();

  // Handlers
  const handleSelectPatient = (patient: Patient) => {
    setSelectedPatientId(patient.id);
    setSelectedProjectId(null);
  };

  const handleCreatePatient = () => {
    setEditingPatient(null);
    setShowPatientForm(true);
  };

  const handlePatientFormSuccess = (patient: Patient) => {
    setSelectedPatientId(patient.id);
  };

  const handleCreateProject = () => {
    if (!selectedPatientId || !newProjectName.trim()) return;

    const body: ProjectCreate = {
      name: newProjectName.trim(),
      patient_id: selectedPatientId,
    };

    createProject.mutate(body, {
      onSuccess: (project) => {
        setSelectedProjectId(project.id);
        setShowCreateProject(false);
        setNewProjectName('');
      },
    });
  };

  const handleOpenGenerator = () => {
    if (selectedProjectId) {
      navigate(`/generator?project=${selectedProjectId}`);
    }
  };

  return (
    <>
    <AppLayout
      sidebar={
        <PatientList
          selectedPatientId={selectedPatientId}
          onSelectPatient={handleSelectPatient}
          onCreatePatient={handleCreatePatient}
        />
      }
      main={
        <div style={styles.mainContent}>
          {!selectedPatientId && (
            <div style={styles.placeholder}>
              <h2 style={styles.placeholderTitle}>Doctor Portal</h2>
              <p style={styles.placeholderText}>
                Select a patient from the sidebar to view their cases and reports,
                or create a new patient to get started.
              </p>
            </div>
          )}

          {selectedPatientId && !selectedProjectId && (
            <div style={styles.patientView}>
              {/* Patient Header */}
              <div style={styles.patientHeader}>
                <div>
                  <h2 style={styles.patientName}>
                    {selectedPatient?.first_name || selectedPatient?.last_name
                      ? `${selectedPatient?.first_name || ''} ${selectedPatient?.last_name || ''}`.trim()
                      : selectedPatient?.patient_code || 'Loading...'}
                  </h2>
                  <div style={styles.patientMeta}>
                    <span style={styles.patientCode}>
                      Code: {selectedPatient?.patient_code}
                    </span>
                    {selectedPatient?.medical_record_number && (
                      <span style={styles.patientMrn}>
                        MRN: {selectedPatient.medical_record_number}
                      </span>
                    )}
                    {selectedPatient?.date_of_birth && (
                      <span>DOB: {selectedPatient.date_of_birth}</span>
                    )}
                  </div>
                </div>
                <button
                  style={styles.editPatientBtn}
                  onClick={() => {
                    setEditingPatient(selectedPatient || null);
                    setShowPatientForm(true);
                  }}
                >
                  Edit Patient
                </button>
              </div>

              {/* Projects Section */}
              <div style={styles.projectsSection}>
                <div style={styles.sectionHeader}>
                  <h3 style={styles.sectionTitle}>Cases / Projects</h3>
                  <button
                    style={styles.addProjectBtn}
                    onClick={() => setShowCreateProject(true)}
                  >
                    + New Case
                  </button>
                </div>

                {showCreateProject && (
                  <div style={styles.createProjectRow}>
                    <input
                      type="text"
                      value={newProjectName}
                      onChange={(e) => setNewProjectName(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleCreateProject()}
                      placeholder="Case name..."
                      style={styles.createProjectInput}
                      autoFocus
                    />
                    <button
                      style={styles.createProjectBtn}
                      onClick={handleCreateProject}
                      disabled={createProject.isPending}
                    >
                      Create
                    </button>
                    <button
                      style={styles.cancelBtn}
                      onClick={() => setShowCreateProject(false)}
                    >
                      Cancel
                    </button>
                  </div>
                )}

                {patientProjects.length === 0 && !showCreateProject && (
                  <div style={styles.emptyProjects}>
                    <p style={styles.emptyText}>No cases yet for this patient</p>
                    <button
                      style={styles.addProjectBtn}
                      onClick={() => setShowCreateProject(true)}
                    >
                      Create First Case
                    </button>
                  </div>
                )}

                <div style={styles.projectsList}>
                  {patientProjects.map((project) => (
                    <div
                      key={project.id}
                      onClick={() => setSelectedProjectId(project.id)}
                      style={styles.projectCard}
                    >
                      <div style={styles.projectName}>{project.name}</div>
                      <div style={styles.projectMeta}>
                        {project.reconstruction_type && (
                          <span style={styles.badge}>{project.reconstruction_type}</span>
                        )}
                        {project.implant_material && (
                          <span style={styles.materialBadge}>{project.implant_material}</span>
                        )}
                        <span style={styles.projectDate}>
                          {new Date(project.created_at).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {selectedProject && (
            <ProjectDetail
              project={selectedProject}
              onOpenGenerator={handleOpenGenerator}
            />
          )}
        </div>
      }
      controls={
        selectedProject ? (
          <div style={styles.controlsContainer}>
            <button
              style={styles.backBtn}
              onClick={() => setSelectedProjectId(null)}
            >
              ← Back to Patient
            </button>
            <ReportPanel project={selectedProject} />
          </div>
        ) : (
          <div style={styles.controlsPlaceholder}>
            <p style={styles.controlsHint}>
              Select a case to view reports and generate new ones
            </p>
          </div>
        )
      }
    />

    {/* Patient Form Dialog */}
    {showPatientForm && (
      <PatientForm
        patient={editingPatient}
        onClose={() => {
          setShowPatientForm(false);
          setEditingPatient(null);
        }}
        onSuccess={handlePatientFormSuccess}
      />
    )}
  </>
  );
}

const styles: Record<string, CSSProperties> = {
  mainContent: {
    height: '100%',
    overflow: 'auto',
    background: '#0a0a1a',
  },
  placeholder: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    padding: '48px',
    textAlign: 'center',
  },
  placeholderTitle: {
    margin: 0,
    fontSize: '28px',
    fontWeight: 600,
    color: '#fff',
  },
  placeholderText: {
    marginTop: '12px',
    fontSize: '15px',
    color: '#888',
    maxWidth: '400px',
  },
  patientView: {
    padding: '24px',
  },
  patientHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: '32px',
    paddingBottom: '24px',
    borderBottom: '1px solid #333',
  },
  patientName: {
    margin: 0,
    fontSize: '28px',
    fontWeight: 600,
    color: '#fff',
  },
  patientMeta: {
    display: 'flex',
    gap: '16px',
    marginTop: '8px',
    fontSize: '13px',
    color: '#888',
  },
  patientCode: {
    fontWeight: 500,
    color: '#aaa',
  },
  patientMrn: {
    color: '#666',
  },
  editPatientBtn: {
    padding: '8px 16px',
    fontSize: '12px',
    background: '#333',
    color: '#ccc',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
  },
  projectsSection: {
    marginTop: '24px',
  },
  sectionHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '16px',
  },
  sectionTitle: {
    margin: 0,
    fontSize: '14px',
    fontWeight: 600,
    color: '#aaa',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  addProjectBtn: {
    padding: '8px 16px',
    fontSize: '12px',
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
  },
  createProjectRow: {
    display: 'flex',
    gap: '8px',
    marginBottom: '16px',
  },
  createProjectInput: {
    flex: 1,
    padding: '10px 14px',
    fontSize: '13px',
    background: '#111',
    color: '#ccc',
    border: '1px solid #333',
    borderRadius: '6px',
  },
  createProjectBtn: {
    padding: '10px 20px',
    fontSize: '12px',
    background: '#10b981',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: 600,
  },
  cancelBtn: {
    padding: '10px 16px',
    fontSize: '12px',
    background: '#333',
    color: '#ccc',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
  },
  emptyProjects: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '16px',
    padding: '48px',
    background: '#16213e',
    borderRadius: '12px',
  },
  emptyText: {
    margin: 0,
    fontSize: '14px',
    color: '#666',
  },
  projectsList: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
    gap: '16px',
  },
  projectCard: {
    padding: '20px',
    background: '#16213e',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'background 0.15s, transform 0.15s',
  },
  projectName: {
    fontSize: '16px',
    fontWeight: 600,
    color: '#fff',
    marginBottom: '8px',
  },
  projectMeta: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '8px',
    alignItems: 'center',
  },
  badge: {
    padding: '3px 10px',
    fontSize: '11px',
    fontWeight: 600,
    textTransform: 'uppercase',
    background: 'rgba(59, 130, 246, 0.2)',
    color: '#60a5fa',
    borderRadius: '4px',
  },
  materialBadge: {
    padding: '3px 10px',
    fontSize: '11px',
    fontWeight: 600,
    textTransform: 'uppercase',
    background: 'rgba(16, 185, 129, 0.2)',
    color: '#34d399',
    borderRadius: '4px',
  },
  projectDate: {
    fontSize: '11px',
    color: '#666',
  },
  controlsContainer: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    padding: '12px',
    gap: '12px',
  },
  backBtn: {
    padding: '8px 12px',
    fontSize: '12px',
    background: '#333',
    color: '#ccc',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    textAlign: 'left',
  },
  controlsPlaceholder: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    padding: '24px',
  },
  controlsHint: {
    fontSize: '12px',
    color: '#666',
    textAlign: 'center',
    fontStyle: 'italic',
  },
};
