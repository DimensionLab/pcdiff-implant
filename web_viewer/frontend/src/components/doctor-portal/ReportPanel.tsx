/**
 * Report panel component for generating and viewing case reports.
 */
import { useState, type CSSProperties } from 'react';
import {
  useCaseReports,
  useCaseReport,
  useGenerateCaseReport,
  useDeleteCaseReport,
} from '../../hooks/useCaseReports';
import { caseReportApi } from '../../services/case-report-api';
import type { Project } from '../../types/project';

// Available AI models for report generation
const AI_MODELS = [
  { id: 'anthropic/claude-4.5-sonnet', name: 'Claude 4.5 Sonnet (Best Quality)' },
  { id: 'anthropic/claude-4.5-haiku', name: 'Claude 4.5 Haiku (Faster)' },
  { id: 'openai/gpt-5.2', name: 'GPT-5.2' },
];

interface ReportPanelProps {
  project: Project;
}

export function ReportPanel({ project }: ReportPanelProps) {
  const [selectedReportId, setSelectedReportId] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState(AI_MODELS[0].id);
  const [showPreview, setShowPreview] = useState(false);

  const { data: reports = [], isLoading } = useCaseReports(project.id);
  const { data: selectedReport } = useCaseReport(selectedReportId);
  const generateReport = useGenerateCaseReport();
  const deleteReport = useDeleteCaseReport();

  const handleGenerate = () => {
    generateReport.mutate(
      {
        body: {
          project_id: project.id,
          region_code: project.region_code || undefined,
        },
        model: selectedModel,
      },
      {
        onSuccess: (report) => {
          setSelectedReportId(report.id);
          setShowPreview(true);
        },
      }
    );
  };

  const handleDelete = (e: React.MouseEvent, reportId: string) => {
    e.stopPropagation();
    if (confirm('Delete this report?')) {
      deleteReport.mutate(reportId, {
        onSuccess: () => {
          if (selectedReportId === reportId) {
            setSelectedReportId(null);
            setShowPreview(false);
          }
        },
      });
    }
  };

  const handleDownloadPdf = (reportId: string) => {
    window.open(caseReportApi.getPdfUrl(reportId), '_blank');
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>Case Reports</h3>
      </div>

      {/* Generate Section */}
      <div style={styles.generateSection}>
        <div style={styles.modelSelect}>
          <label style={styles.label}>AI Model</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            style={styles.select}
          >
            {AI_MODELS.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
        </div>
        <button
          style={styles.generateBtn}
          onClick={handleGenerate}
          disabled={generateReport.isPending}
        >
          {generateReport.isPending ? 'Generating...' : 'Generate Report'}
        </button>
        {generateReport.isPending && (
          <p style={styles.hint}>This may take 30-60 seconds...</p>
        )}
      </div>

      {/* Reports List */}
      <div style={styles.reportsList}>
        {isLoading && <div style={styles.loading}>Loading reports...</div>}
        {!isLoading && reports.length === 0 && (
          <div style={styles.empty}>No reports generated yet</div>
        )}
        {reports.map((report) => (
          <div
            key={report.id}
            onClick={() => {
              setSelectedReportId(report.id);
              setShowPreview(true);
            }}
            style={{
              ...styles.reportItem,
              ...(selectedReportId === report.id ? styles.reportItemSelected : {}),
            }}
          >
            <div style={styles.reportMain}>
              <div style={styles.reportTitle}>{report.title}</div>
              <div style={styles.reportMeta}>
                <span>{new Date(report.generated_at).toLocaleDateString()}</span>
                <span style={styles.modelBadge}>{report.ai_model}</span>
              </div>
            </div>
            <div style={styles.reportActions}>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleDownloadPdf(report.id);
                }}
                style={styles.downloadBtn}
                title="Download PDF"
              >
                PDF
              </button>
              <button
                onClick={(e) => handleDelete(e, report.id)}
                style={styles.deleteBtn}
                title="Delete report"
              >
                ×
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Report Preview Modal */}
      {showPreview && selectedReport && (
        <div style={styles.previewOverlay} onClick={() => setShowPreview(false)}>
          <div style={styles.previewDialog} onClick={(e) => e.stopPropagation()}>
            <div style={styles.previewHeader}>
              <h3 style={styles.previewTitle}>{selectedReport.title}</h3>
              <div style={styles.previewActions}>
                <button
                  style={styles.pdfBtn}
                  onClick={() => handleDownloadPdf(selectedReport.id)}
                >
                  Download PDF
                </button>
                <button
                  style={styles.closeBtn}
                  onClick={() => setShowPreview(false)}
                >
                  ×
                </button>
              </div>
            </div>
            <iframe
              src={caseReportApi.getHtmlUrl(selectedReport.id)}
              style={styles.previewFrame}
              title="Report Preview"
            />
          </div>
        </div>
      )}
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    background: '#16213e',
    borderRadius: '8px',
    overflow: 'hidden',
  },
  header: {
    padding: '16px',
    borderBottom: '1px solid #333',
  },
  title: {
    margin: 0,
    fontSize: '14px',
    fontWeight: 600,
    color: '#aaa',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  generateSection: {
    padding: '16px',
    borderBottom: '1px solid #333',
  },
  modelSelect: {
    marginBottom: '12px',
  },
  label: {
    display: 'block',
    marginBottom: '6px',
    fontSize: '11px',
    color: '#666',
    textTransform: 'uppercase',
  },
  select: {
    width: '100%',
    padding: '10px 12px',
    fontSize: '13px',
    background: '#111',
    color: '#ccc',
    border: '1px solid #333',
    borderRadius: '6px',
  },
  generateBtn: {
    width: '100%',
    padding: '12px',
    fontSize: '13px',
    fontWeight: 600,
    background: '#8b5cf6',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
  },
  hint: {
    margin: '8px 0 0',
    fontSize: '11px',
    color: '#666',
    textAlign: 'center',
  },
  reportsList: {
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
  reportItem: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '12px',
    marginBottom: '4px',
    background: '#111',
    borderRadius: '6px',
    cursor: 'pointer',
  },
  reportItemSelected: {
    background: 'rgba(139, 92, 246, 0.2)',
    border: '1px solid rgba(139, 92, 246, 0.4)',
  },
  reportMain: {
    flex: 1,
    overflow: 'hidden',
  },
  reportTitle: {
    fontSize: '13px',
    fontWeight: 500,
    color: '#fff',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  reportMeta: {
    display: 'flex',
    gap: '8px',
    marginTop: '4px',
    fontSize: '11px',
    color: '#666',
  },
  modelBadge: {
    padding: '1px 6px',
    background: 'rgba(139, 92, 246, 0.2)',
    borderRadius: '3px',
    fontSize: '10px',
    color: '#a78bfa',
  },
  reportActions: {
    display: 'flex',
    gap: '4px',
    marginLeft: '8px',
  },
  downloadBtn: {
    padding: '4px 8px',
    fontSize: '10px',
    fontWeight: 600,
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  deleteBtn: {
    width: '24px',
    height: '24px',
    padding: 0,
    fontSize: '16px',
    lineHeight: '22px',
    background: 'transparent',
    color: '#666',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  previewOverlay: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(0, 0, 0, 0.8)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  previewDialog: {
    width: '90%',
    maxWidth: '900px',
    height: '90%',
    background: '#fff',
    borderRadius: '12px',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
  },
  previewHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '16px 20px',
    background: '#f5f5f5',
    borderBottom: '1px solid #ddd',
  },
  previewTitle: {
    margin: 0,
    fontSize: '16px',
    fontWeight: 600,
    color: '#333',
  },
  previewActions: {
    display: 'flex',
    gap: '8px',
  },
  pdfBtn: {
    padding: '8px 16px',
    fontSize: '12px',
    fontWeight: 600,
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
  },
  closeBtn: {
    width: '32px',
    height: '32px',
    padding: 0,
    fontSize: '20px',
    lineHeight: '30px',
    background: '#eee',
    color: '#666',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
  },
  previewFrame: {
    flex: 1,
    width: '100%',
    border: 'none',
  },
};
