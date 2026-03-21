/**
 * Patient list component with search and selection.
 */
import { useState, type CSSProperties } from 'react';
import { usePatients, useDeletePatient } from '../../hooks/usePatients';
import type { Patient } from '../../types/patient';

interface PatientListProps {
  selectedPatientId: string | null;
  onSelectPatient: (patient: Patient) => void;
  onCreatePatient: () => void;
}

export function PatientList({
  selectedPatientId,
  onSelectPatient,
  onCreatePatient,
}: PatientListProps) {
  const [search, setSearch] = useState('');
  const { data: patients = [], isLoading } = usePatients({ search: search || undefined });
  const deletePatient = useDeletePatient();

  const handleDelete = (e: React.MouseEvent, patient: Patient) => {
    e.stopPropagation();
    if (confirm(`Delete patient ${patient.patient_code}?`)) {
      deletePatient.mutate(patient.id);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>Patients</h3>
        <button style={styles.addBtn} onClick={onCreatePatient}>
          + New Patient
        </button>
      </div>

      <div style={styles.searchRow}>
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search patients..."
          style={styles.searchInput}
        />
      </div>

      <div style={styles.list}>
        {isLoading && <div style={styles.loading}>Loading...</div>}
        {!isLoading && patients.length === 0 && (
          <div style={styles.empty}>No patients found</div>
        )}
        {patients.map((patient) => (
          <div
            key={patient.id}
            onClick={() => onSelectPatient(patient)}
            style={{
              ...styles.patientItem,
              ...(selectedPatientId === patient.id ? styles.patientItemSelected : {}),
            }}
          >
            <div style={styles.patientMain}>
              <div style={styles.patientCode}>{patient.patient_code}</div>
              <div style={styles.patientName}>
                {patient.first_name || patient.last_name
                  ? `${patient.first_name || ''} ${patient.last_name || ''}`.trim()
                  : 'No name'}
              </div>
            </div>
            <div style={styles.patientMeta}>
              {patient.medical_record_number && (
                <span style={styles.mrn}>MRN: {patient.medical_record_number}</span>
              )}
              <button
                onClick={(e) => handleDelete(e, patient)}
                style={styles.deleteBtn}
                title="Delete patient"
              >
                ×
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '12px 16px',
    borderBottom: '1px solid #333',
  },
  title: {
    margin: 0,
    fontSize: '14px',
    fontWeight: 600,
    color: '#fff',
  },
  addBtn: {
    padding: '6px 12px',
    fontSize: '12px',
    background: '#10b981',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  searchRow: {
    padding: '8px 16px',
  },
  searchInput: {
    width: '100%',
    padding: '8px 12px',
    fontSize: '13px',
    background: '#111',
    color: '#ccc',
    border: '1px solid #333',
    borderRadius: '6px',
  },
  list: {
    flex: 1,
    overflow: 'auto',
    padding: '8px',
  },
  loading: {
    padding: '16px',
    textAlign: 'center',
    color: '#666',
  },
  empty: {
    padding: '16px',
    textAlign: 'center',
    color: '#666',
    fontStyle: 'italic',
  },
  patientItem: {
    padding: '12px',
    marginBottom: '4px',
    background: '#16213e',
    borderRadius: '6px',
    cursor: 'pointer',
    transition: 'background 0.15s',
  },
  patientItemSelected: {
    background: 'rgba(59, 130, 246, 0.3)',
    border: '1px solid rgba(59, 130, 246, 0.5)',
  },
  patientMain: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  patientCode: {
    fontSize: '13px',
    fontWeight: 600,
    color: '#fff',
  },
  patientName: {
    fontSize: '12px',
    color: '#aaa',
  },
  patientMeta: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: '8px',
  },
  mrn: {
    fontSize: '10px',
    color: '#666',
  },
  deleteBtn: {
    width: '20px',
    height: '20px',
    padding: 0,
    fontSize: '16px',
    lineHeight: '18px',
    background: 'transparent',
    color: '#666',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
};
