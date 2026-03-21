import { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import type { ConvertedFile } from '../types';

interface FileDownloadProps {
  resultId: string | null;
}

export const FileDownload = ({ resultId }: FileDownloadProps) => {
  const [files, setFiles] = useState<ConvertedFile[]>([]);
  const [loading, setLoading] = useState(false);
  const [converting, setConverting] = useState(false);

  useEffect(() => {
    if (!resultId) {
      setFiles([]);
      return;
    }

    const fetchFiles = async () => {
      try {
        setLoading(true);
        const response = await apiService.getResultFiles(resultId);
        setFiles(response.files);
      } catch (error) {
        console.error('Failed to fetch files:', error);
        setFiles([]);
      } finally {
        setLoading(false);
      }
    };

    fetchFiles();
  }, [resultId]);

  const handleConvert = async () => {
    if (!resultId) return;

    try {
      setConverting(true);
      await apiService.convertResult(resultId, false, true);
      
      // Wait a bit and refresh files
      setTimeout(async () => {
        const response = await apiService.getResultFiles(resultId);
        setFiles(response.files);
        setConverting(false);
      }, 2000);
    } catch (error) {
      console.error('Conversion failed:', error);
      setConverting(false);
    }
  };

  const handleDownload = (file: ConvertedFile) => {
    const link = document.createElement('a');
    link.href = file.path;
    link.download = file.name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (!resultId) {
    return (
      <div style={styles.container}>
        <h3 style={styles.title}>Downloads</h3>
        <div style={styles.empty}>Select a result to download files</div>
      </div>
    );
  }

  if (loading) {
    return (
      <div style={styles.container}>
        <h3 style={styles.title}>Downloads</h3>
        <div style={styles.empty}>Loading...</div>
      </div>
    );
  }

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const plyFiles = files.filter(f => f.type === 'ply');
  const stlFiles = files.filter(f => f.type === 'stl');

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Downloads</h3>
      
      {files.length === 0 ? (
        <div style={styles.emptyState}>
          <div style={styles.empty}>No converted files available</div>
          <button
            style={styles.convertButton}
            onClick={handleConvert}
            disabled={converting}
          >
            {converting ? 'Converting...' : 'Convert Now'}
          </button>
        </div>
      ) : (
        <>
          {/* PLY Files */}
          {plyFiles.length > 0 && (
            <div style={styles.section}>
              <h4 style={styles.sectionTitle}>PLY Files (Visualization)</h4>
              <div style={styles.fileList}>
                {plyFiles.map((file) => (
                  <div key={file.name} style={styles.fileItem}>
                    <div style={styles.fileInfo}>
                      <div style={styles.fileName}>{file.name}</div>
                      <div style={styles.fileSize}>{formatFileSize(file.size)}</div>
                    </div>
                    <button
                      style={styles.downloadButton}
                      onClick={() => handleDownload(file)}
                    >
                      Download
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* STL Files */}
          {stlFiles.length > 0 && (
            <div style={styles.section}>
              <h4 style={styles.sectionTitle}>STL Files (3D Printing)</h4>
              <div style={styles.fileList}>
                {stlFiles.map((file) => (
                  <div key={file.name} style={styles.fileItem}>
                    <div style={styles.fileInfo}>
                      <div style={styles.fileName}>{file.name}</div>
                      <div style={styles.fileSize}>{formatFileSize(file.size)}</div>
                    </div>
                    <button
                      style={styles.downloadButton}
                      onClick={() => handleDownload(file)}
                    >
                      Download
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          <button
            style={styles.convertButton}
            onClick={handleConvert}
            disabled={converting}
          >
            {converting ? 'Converting...' : 'Reconvert Files'}
          </button>
        </>
      )}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '20px',
    backgroundColor: '#2a2a2a',
    borderRadius: '8px',
    color: '#ffffff',
  },
  title: {
    margin: '0 0 20px 0',
    fontSize: '18px',
    fontWeight: '600',
  },
  section: {
    marginBottom: '20px',
  },
  sectionTitle: {
    margin: '0 0 10px 0',
    fontSize: '14px',
    fontWeight: '500',
    color: '#cccccc',
  },
  fileList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  fileItem: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '10px',
    backgroundColor: '#333333',
    borderRadius: '6px',
    border: '1px solid #444444',
  },
  fileInfo: {
    flex: 1,
  },
  fileName: {
    fontSize: '13px',
    fontWeight: '500',
    marginBottom: '4px',
  },
  fileSize: {
    fontSize: '11px',
    color: '#999999',
  },
  downloadButton: {
    padding: '6px 12px',
    backgroundColor: '#4CAF50',
    color: '#ffffff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '12px',
    fontWeight: '500',
    transition: 'background-color 0.2s',
  },
  convertButton: {
    width: '100%',
    padding: '10px',
    backgroundColor: '#2196F3',
    color: '#ffffff',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '500',
    marginTop: '10px',
  },
  empty: {
    color: '#999999',
    textAlign: 'center',
    padding: '20px 0',
    fontSize: '13px',
  },
  emptyState: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
  },
};

export default FileDownload;

