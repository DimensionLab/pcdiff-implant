/**
 * Patient create/edit form component.
 */
import { useState, useEffect, type CSSProperties } from 'react';
import { useCreatePatient, useUpdatePatient } from '../../hooks/usePatients';
import type { Patient, PatientCreate } from '../../types/patient';

interface PatientFormProps {
  patient?: Patient | null;
  onClose: () => void;
  onSuccess: (patient: Patient) => void;
}

export function PatientForm({ patient, onClose, onSuccess }: PatientFormProps) {
  const isEditing = !!patient;
  const createPatient = useCreatePatient();
  const updatePatient = useUpdatePatient();

  const [formData, setFormData] = useState<PatientCreate>({
    patient_code: '',
    first_name: '',
    last_name: '',
    date_of_birth: '',
    sex: '',
    email: '',
    phone: '',
    medical_record_number: '',
    insurance_provider: '',
    insurance_policy_number: '',
    notes: '',
  });

  useEffect(() => {
    if (patient) {
      setFormData({
        patient_code: patient.patient_code,
        first_name: patient.first_name || '',
        last_name: patient.last_name || '',
        date_of_birth: patient.date_of_birth || '',
        sex: patient.sex || '',
        email: patient.email || '',
        phone: patient.phone || '',
        medical_record_number: patient.medical_record_number || '',
        insurance_provider: patient.insurance_provider || '',
        insurance_policy_number: patient.insurance_policy_number || '',
        notes: patient.notes || '',
      });
    }
  }, [patient]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.patient_code.trim()) {
      alert('Patient code is required');
      return;
    }

    if (isEditing) {
      updatePatient.mutate(
        { id: patient.id, body: formData },
        {
          onSuccess: (updatedPatient) => {
            onSuccess(updatedPatient);
            onClose();
          },
        }
      );
    } else {
      createPatient.mutate(formData, {
        onSuccess: (newPatient) => {
          onSuccess(newPatient);
          onClose();
        },
      });
    }
  };

  const isPending = createPatient.isPending || updatePatient.isPending;

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <div style={styles.header}>
          <h3 style={styles.title}>{isEditing ? 'Edit Patient' : 'New Patient'}</h3>
          <button style={styles.closeBtn} onClick={onClose}>×</button>
        </div>

        <form onSubmit={handleSubmit} style={styles.form}>
          <div style={styles.row}>
            <div style={styles.field}>
              <label style={styles.label}>Patient Code *</label>
              <input
                type="text"
                value={formData.patient_code}
                onChange={(e) => setFormData({ ...formData, patient_code: e.target.value })}
                placeholder="PAT-2026-001"
                style={styles.input}
                required
              />
            </div>
            <div style={styles.field}>
              <label style={styles.label}>MRN</label>
              <input
                type="text"
                value={formData.medical_record_number}
                onChange={(e) => setFormData({ ...formData, medical_record_number: e.target.value })}
                placeholder="Hospital MRN"
                style={styles.input}
              />
            </div>
          </div>

          <div style={styles.row}>
            <div style={styles.field}>
              <label style={styles.label}>First Name</label>
              <input
                type="text"
                value={formData.first_name}
                onChange={(e) => setFormData({ ...formData, first_name: e.target.value })}
                style={styles.input}
              />
            </div>
            <div style={styles.field}>
              <label style={styles.label}>Last Name</label>
              <input
                type="text"
                value={formData.last_name}
                onChange={(e) => setFormData({ ...formData, last_name: e.target.value })}
                style={styles.input}
              />
            </div>
          </div>

          <div style={styles.row}>
            <div style={styles.field}>
              <label style={styles.label}>Date of Birth</label>
              <input
                type="date"
                value={formData.date_of_birth}
                onChange={(e) => setFormData({ ...formData, date_of_birth: e.target.value })}
                style={styles.input}
              />
            </div>
            <div style={styles.field}>
              <label style={styles.label}>Sex</label>
              <select
                value={formData.sex}
                onChange={(e) => setFormData({ ...formData, sex: e.target.value })}
                style={styles.input}
              >
                <option value="">Select...</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
                <option value="unknown">Unknown</option>
              </select>
            </div>
          </div>

          <div style={styles.row}>
            <div style={styles.field}>
              <label style={styles.label}>Email</label>
              <input
                type="email"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                style={styles.input}
              />
            </div>
            <div style={styles.field}>
              <label style={styles.label}>Phone</label>
              <input
                type="tel"
                value={formData.phone}
                onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                style={styles.input}
              />
            </div>
          </div>

          <div style={styles.row}>
            <div style={styles.field}>
              <label style={styles.label}>Insurance Provider</label>
              <input
                type="text"
                value={formData.insurance_provider}
                onChange={(e) => setFormData({ ...formData, insurance_provider: e.target.value })}
                style={styles.input}
              />
            </div>
            <div style={styles.field}>
              <label style={styles.label}>Policy Number</label>
              <input
                type="text"
                value={formData.insurance_policy_number}
                onChange={(e) => setFormData({ ...formData, insurance_policy_number: e.target.value })}
                style={styles.input}
              />
            </div>
          </div>

          <div style={styles.field}>
            <label style={styles.label}>Notes</label>
            <textarea
              value={formData.notes}
              onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
              style={styles.textarea}
              rows={3}
            />
          </div>

          <div style={styles.actions}>
            <button type="button" style={styles.cancelBtn} onClick={onClose}>
              Cancel
            </button>
            <button type="submit" style={styles.submitBtn} disabled={isPending}>
              {isPending ? 'Saving...' : isEditing ? 'Update' : 'Create'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  overlay: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(0, 0, 0, 0.7)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  dialog: {
    width: '500px',
    maxHeight: '90vh',
    overflow: 'auto',
    background: '#1a1a2e',
    borderRadius: '12px',
    border: '1px solid #333',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '16px 20px',
    borderBottom: '1px solid #333',
  },
  title: {
    margin: 0,
    fontSize: '16px',
    fontWeight: 600,
    color: '#fff',
  },
  closeBtn: {
    width: '28px',
    height: '28px',
    padding: 0,
    fontSize: '20px',
    lineHeight: '26px',
    background: 'transparent',
    color: '#666',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  form: {
    padding: '20px',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  row: {
    display: 'flex',
    gap: '16px',
  },
  field: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  label: {
    fontSize: '12px',
    fontWeight: 500,
    color: '#aaa',
  },
  input: {
    padding: '10px 12px',
    fontSize: '13px',
    background: '#111',
    color: '#ccc',
    border: '1px solid #333',
    borderRadius: '6px',
  },
  textarea: {
    padding: '10px 12px',
    fontSize: '13px',
    background: '#111',
    color: '#ccc',
    border: '1px solid #333',
    borderRadius: '6px',
    resize: 'vertical',
  },
  actions: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: '12px',
    marginTop: '8px',
  },
  cancelBtn: {
    padding: '10px 20px',
    fontSize: '13px',
    background: '#333',
    color: '#ccc',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
  },
  submitBtn: {
    padding: '10px 24px',
    fontSize: '13px',
    background: '#10b981',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: 600,
  },
};
