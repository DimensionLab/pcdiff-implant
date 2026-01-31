import { useState, useEffect } from 'react';
import { filesystemApi, FileEntry, CommonPath } from '../../services/filesystem-api';

interface FilePickerProps {
  onSelect: (path: string) => void;
  onCancel: () => void;
  title?: string;
  filterExtensions?: boolean;
}

export function FilePicker({
  onSelect,
  onCancel,
  title = 'Select File',
  filterExtensions = true,
}: FilePickerProps) {
  const [currentPath, setCurrentPath] = useState('');
  const [entries, setEntries] = useState<FileEntry[]>([]);
  const [parentPath, setParentPath] = useState<string | null>(null);
  const [commonPaths, setCommonPaths] = useState<CommonPath[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  useEffect(() => {
    loadCommonPaths();
    loadInitialDirectory();
  }, []);

  const loadCommonPaths = async () => {
    try {
      const paths = await filesystemApi.getCommonPaths();
      setCommonPaths(paths);
    } catch (e) {
      console.error('Failed to load common paths:', e);
    }
  };

  const loadInitialDirectory = async () => {
    try {
      const { path } = await filesystemApi.getHome();
      await browseDirectory(path);
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Failed to load home directory');
      setLoading(false);
    }
  };

  const browseDirectory = async (path: string) => {
    setLoading(true);
    setError('');
    setSelectedFile(null);
    try {
      const listing = await filesystemApi.browse({
        path,
        filter_extensions: filterExtensions,
      });
      setCurrentPath(listing.path);
      setParentPath(listing.parent);
      setEntries(listing.entries);
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Failed to browse directory');
    } finally {
      setLoading(false);
    }
  };

  const handleEntryClick = (entry: FileEntry) => {
    if (entry.is_dir) {
      browseDirectory(entry.path);
    } else {
      setSelectedFile(entry.path);
    }
  };

  const handleEntryDoubleClick = (entry: FileEntry) => {
    if (entry.is_dir) {
      browseDirectory(entry.path);
    } else {
      onSelect(entry.path);
    }
  };

  const handleSelect = () => {
    if (selectedFile) {
      onSelect(selectedFile);
    }
  };

  const formatSize = (bytes: number | null) => {
    if (bytes === null) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div style={styles.overlay} onClick={onCancel}>
      <div style={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <div style={styles.header}>
          <h3 style={styles.title}>{title}</h3>
          <button style={styles.closeBtn} onClick={onCancel}>
            x
          </button>
        </div>

        <div style={styles.quickNav}>
          {commonPaths.map((cp) => (
            <button
              key={cp.path}
              style={styles.quickNavBtn}
              onClick={() => browseDirectory(cp.path)}
            >
              {cp.name}
            </button>
          ))}
        </div>

        <div style={styles.pathBar}>
          {parentPath && (
            <button
              style={styles.upBtn}
              onClick={() => browseDirectory(parentPath)}
            >
              ..
            </button>
          )}
          <span style={styles.pathText}>{currentPath}</span>
        </div>

        <div style={styles.fileList}>
          {loading ? (
            <div style={styles.loadingText}>Loading...</div>
          ) : error ? (
            <div style={styles.errorText}>{error}</div>
          ) : entries.length === 0 ? (
            <div style={styles.emptyText}>No files found</div>
          ) : (
            entries.map((entry) => (
              <div
                key={entry.path}
                style={{
                  ...styles.fileEntry,
                  ...(selectedFile === entry.path ? styles.fileEntrySelected : {}),
                }}
                onClick={() => handleEntryClick(entry)}
                onDoubleClick={() => handleEntryDoubleClick(entry)}
              >
                <span style={styles.fileIcon}>{entry.is_dir ? '📁' : '📄'}</span>
                <span style={styles.fileName}>{entry.name}</span>
                {!entry.is_dir && (
                  <span style={styles.fileSize}>{formatSize(entry.size)}</span>
                )}
              </div>
            ))
          )}
        </div>

        <div style={styles.footer}>
          <span style={styles.selectedPath}>
            {selectedFile ? selectedFile.split('/').pop() : 'No file selected'}
          </span>
          <div style={styles.buttons}>
            <button style={styles.cancelBtn} onClick={onCancel}>
              Cancel
            </button>
            <button
              style={{
                ...styles.selectBtn,
                ...(selectedFile ? {} : styles.selectBtnDisabled),
              }}
              onClick={handleSelect}
              disabled={!selectedFile}
            >
              Select
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(0,0,0,0.7)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1100,
  },
  dialog: {
    background: '#1a1a2e',
    border: '1px solid #333',
    borderRadius: '8px',
    width: '600px',
    maxWidth: '90vw',
    maxHeight: '80vh',
    display: 'flex',
    flexDirection: 'column',
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
    color: '#fff',
  },
  closeBtn: {
    background: 'transparent',
    border: 'none',
    color: '#999',
    fontSize: '18px',
    cursor: 'pointer',
    padding: '4px 8px',
  },
  quickNav: {
    display: 'flex',
    gap: '8px',
    padding: '12px 20px',
    borderBottom: '1px solid #333',
    flexWrap: 'wrap',
  },
  quickNavBtn: {
    padding: '4px 12px',
    fontSize: '12px',
    background: '#16213e',
    color: '#aaa',
    border: '1px solid #333',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  pathBar: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '10px 20px',
    background: '#16213e',
    borderBottom: '1px solid #333',
  },
  upBtn: {
    padding: '4px 10px',
    fontSize: '12px',
    background: '#2a2a4e',
    color: '#aaa',
    border: '1px solid #444',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  pathText: {
    fontSize: '12px',
    color: '#888',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  fileList: {
    flex: 1,
    overflow: 'auto',
    minHeight: '300px',
    maxHeight: '400px',
  },
  loadingText: {
    padding: '40px',
    textAlign: 'center',
    color: '#888',
  },
  errorText: {
    padding: '40px',
    textAlign: 'center',
    color: '#e53e3e',
  },
  emptyText: {
    padding: '40px',
    textAlign: 'center',
    color: '#666',
  },
  fileEntry: {
    display: 'flex',
    alignItems: 'center',
    padding: '8px 20px',
    cursor: 'pointer',
    borderBottom: '1px solid #2a2a3e',
  },
  fileEntrySelected: {
    background: '#2563eb33',
  },
  fileIcon: {
    marginRight: '10px',
    fontSize: '14px',
  },
  fileName: {
    flex: 1,
    fontSize: '13px',
    color: '#e0e0e0',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  fileSize: {
    fontSize: '11px',
    color: '#666',
    marginLeft: '12px',
  },
  footer: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '16px 20px',
    borderTop: '1px solid #333',
  },
  selectedPath: {
    fontSize: '12px',
    color: '#888',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    maxWidth: '250px',
  },
  buttons: {
    display: 'flex',
    gap: '8px',
  },
  cancelBtn: {
    padding: '8px 16px',
    fontSize: '13px',
    background: 'transparent',
    color: '#999',
    border: '1px solid #444',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  selectBtn: {
    padding: '8px 16px',
    fontSize: '13px',
    background: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  selectBtnDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
};
