/**
 * Project detail component showing case metadata, scans, and implants.
 */
import { useState, type CSSProperties } from 'react';
import { useQuery } from '@tanstack/react-query';
import { pointCloudApi } from '../../services/point-cloud-api';
import { scanApi } from '../../services/scan-api';
import { useUpdateProject } from '../../hooks/useProjects';
import type { Project, ProjectUpdate } from '../../types/project';

// Material options for cranioplasty implants
const MATERIALS = ['PEEK', 'Titanium', 'MEDPOR', 'PMMA', 'Other'];
const RECONSTRUCTION_TYPES = ['Cranioplasty', 'Maxillofacial', 'Orbital', 'Other'];
const REGIONS = [
  { code: 'US', name: 'United States' },
  { code: 'EU', name: 'European Union' },
  { code: 'SK', name: 'Slovakia' },
  { code: 'GB', name: 'United Kingdom' },
  { code: 'CA', name: 'Canada' },
  { code: 'AU', name: 'Australia' },
];

interface ProjectDetailProps {
  project: Project;
  onOpenGenerator: () => void;
}

export function ProjectDetail({ project, onOpenGenerator }: ProjectDetailProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editData, setEditData] = useState<ProjectUpdate>({
    reconstruction_type: project.reconstruction_type || '',
    implant_material: project.implant_material || '',
    notes: project.notes || '',
    region_code: project.region_code || '',
  });

  // Fetch related data
  const { data: scans = [] } = useQuery({
    queryKey: ['project-scans', project.id],
    queryFn: () => scanApi.list({ project_id: project.id }),
  });

  const { data: pointClouds = [] } = useQuery({
    queryKey: ['project-point-clouds', project.id],
    queryFn: () => pointCloudApi.list({ project_id: project.id }),
  });

  // Filter for implants
  const implants = pointClouds.filter(
    (pc) => pc.scan_category === 'implant' || pc.scan_category === 'generated_implant'
  );

  // Update mutation
  const updateProject = useUpdateProject();

  const handleSave = () => {
    updateProject.mutate(
      { id: project.id, body: editData },
      { onSuccess: () => setIsEditing(false) }
    );
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div>
          <h2 style={styles.title}>{project.name}</h2>
          <p style={styles.description}>{project.description || 'No description'}</p>
        </div>
        <div style={styles.headerActions}>
          <button style={styles.generateBtn} onClick={onOpenGenerator}>
            Generate Implant
          </button>
        </div>
      </div>

      {/* Case Metadata */}
      <section style={styles.section}>
        <div style={styles.sectionHeader}>
          <h3 style={styles.sectionTitle}>Case Information</h3>
          {!isEditing ? (
            <button style={styles.editBtn} onClick={() => setIsEditing(true)}>
              Edit
            </button>
          ) : (
            <div style={styles.editActions}>
              <button style={styles.cancelBtn} onClick={() => setIsEditing(false)}>
                Cancel
              </button>
              <button style={styles.saveBtn} onClick={handleSave}>
                Save
              </button>
            </div>
          )}
        </div>

        <div style={styles.metadataGrid}>
          <div style={styles.metaField}>
            <span style={styles.metaLabel}>Reconstruction Type</span>
            {isEditing ? (
              <select
                value={editData.reconstruction_type || ''}
                onChange={(e) => setEditData({ ...editData, reconstruction_type: e.target.value })}
                style={styles.select}
              >
                <option value="">Select...</option>
                {RECONSTRUCTION_TYPES.map((type) => (
                  <option key={type} value={type}>{type}</option>
                ))}
              </select>
            ) : (
              <span style={styles.metaValue}>{project.reconstruction_type || 'Not specified'}</span>
            )}
          </div>

          <div style={styles.metaField}>
            <span style={styles.metaLabel}>Implant Material</span>
            {isEditing ? (
              <select
                value={editData.implant_material || ''}
                onChange={(e) => setEditData({ ...editData, implant_material: e.target.value })}
                style={styles.select}
              >
                <option value="">Select...</option>
                {MATERIALS.map((mat) => (
                  <option key={mat} value={mat}>{mat}</option>
                ))}
              </select>
            ) : (
              <span style={styles.metaValue}>{project.implant_material || 'Not specified'}</span>
            )}
          </div>

          <div style={styles.metaField}>
            <span style={styles.metaLabel}>Region</span>
            {isEditing ? (
              <select
                value={editData.region_code || ''}
                onChange={(e) => setEditData({ ...editData, region_code: e.target.value })}
                style={styles.select}
              >
                <option value="">Select...</option>
                {REGIONS.map((r) => (
                  <option key={r.code} value={r.code}>{r.name}</option>
                ))}
              </select>
            ) : (
              <span style={styles.metaValue}>
                {REGIONS.find((r) => r.code === project.region_code)?.name || project.region_code || 'Not specified'}
              </span>
            )}
          </div>

          <div style={styles.metaField}>
            <span style={styles.metaLabel}>Created</span>
            <span style={styles.metaValue}>
              {new Date(project.created_at).toLocaleDateString()}
            </span>
          </div>
        </div>

        {isEditing ? (
          <div style={styles.notesField}>
            <span style={styles.metaLabel}>Notes</span>
            <textarea
              value={editData.notes || ''}
              onChange={(e) => setEditData({ ...editData, notes: e.target.value })}
              style={styles.textarea}
              rows={3}
              placeholder="Add case notes..."
            />
          </div>
        ) : project.notes ? (
          <div style={styles.notesField}>
            <span style={styles.metaLabel}>Notes</span>
            <p style={styles.notesText}>{project.notes}</p>
          </div>
        ) : null}
      </section>

      {/* CT Scans */}
      <section style={styles.section}>
        <h3 style={styles.sectionTitle}>CT Scans ({scans.length})</h3>
        {scans.length === 0 ? (
          <p style={styles.emptyText}>No scans linked to this project</p>
        ) : (
          <div style={styles.itemList}>
            {scans.map((scan) => (
              <div key={scan.id} style={styles.item}>
                <div style={styles.itemName}>{scan.name}</div>
                <div style={styles.itemMeta}>
                  <span style={styles.badge}>{scan.scan_category || 'Unknown'}</span>
                  {scan.volume_dims_x && (
                    <span style={styles.dims}>
                      {scan.volume_dims_x}×{scan.volume_dims_y}×{scan.volume_dims_z}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Generated Implants */}
      <section style={styles.section}>
        <h3 style={styles.sectionTitle}>Generated Implants ({implants.length})</h3>
        {implants.length === 0 ? (
          <div style={styles.emptyState}>
            <p style={styles.emptyText}>No implants generated yet</p>
            <button style={styles.generateBtn} onClick={onOpenGenerator}>
              Generate Implant
            </button>
          </div>
        ) : (
          <div style={styles.itemList}>
            {implants.map((implant) => (
              <div key={implant.id} style={styles.item}>
                <div style={styles.itemName}>{implant.name}</div>
                <div style={styles.itemMeta}>
                  <span style={styles.badge}>{implant.scan_category}</span>
                  <span style={styles.format}>{implant.file_format.toUpperCase()}</span>
                  {implant.num_points && (
                    <span style={styles.points}>{implant.num_points.toLocaleString()} pts</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  container: {
    padding: '24px',
    overflow: 'auto',
    height: '100%',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: '24px',
  },
  title: {
    margin: 0,
    fontSize: '24px',
    fontWeight: 600,
    color: '#fff',
  },
  description: {
    margin: '4px 0 0',
    fontSize: '14px',
    color: '#888',
  },
  headerActions: {
    display: 'flex',
    gap: '8px',
  },
  generateBtn: {
    padding: '10px 20px',
    fontSize: '13px',
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: 500,
  },
  section: {
    marginBottom: '24px',
    padding: '20px',
    background: '#16213e',
    borderRadius: '8px',
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
  editBtn: {
    padding: '4px 12px',
    fontSize: '12px',
    background: '#333',
    color: '#ccc',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  editActions: {
    display: 'flex',
    gap: '8px',
  },
  cancelBtn: {
    padding: '4px 12px',
    fontSize: '12px',
    background: '#333',
    color: '#ccc',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  saveBtn: {
    padding: '4px 12px',
    fontSize: '12px',
    background: '#10b981',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  metadataGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '16px',
  },
  metaField: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  metaLabel: {
    fontSize: '11px',
    color: '#666',
    textTransform: 'uppercase',
  },
  metaValue: {
    fontSize: '14px',
    color: '#fff',
  },
  select: {
    padding: '8px 12px',
    fontSize: '13px',
    background: '#111',
    color: '#ccc',
    border: '1px solid #333',
    borderRadius: '6px',
  },
  notesField: {
    marginTop: '16px',
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  notesText: {
    margin: 0,
    fontSize: '13px',
    color: '#ccc',
    lineHeight: 1.5,
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
  emptyText: {
    margin: 0,
    fontSize: '13px',
    color: '#666',
    fontStyle: 'italic',
  },
  emptyState: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '12px',
    padding: '16px',
  },
  itemList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  item: {
    padding: '12px',
    background: '#111',
    borderRadius: '6px',
  },
  itemName: {
    fontSize: '13px',
    fontWeight: 500,
    color: '#fff',
    marginBottom: '4px',
  },
  itemMeta: {
    display: 'flex',
    gap: '8px',
    alignItems: 'center',
  },
  badge: {
    padding: '2px 8px',
    fontSize: '10px',
    fontWeight: 600,
    textTransform: 'uppercase',
    background: 'rgba(59, 130, 246, 0.2)',
    color: '#60a5fa',
    borderRadius: '4px',
  },
  dims: {
    fontSize: '11px',
    color: '#666',
  },
  format: {
    fontSize: '10px',
    color: '#888',
    fontWeight: 600,
  },
  points: {
    fontSize: '11px',
    color: '#666',
  },
};
