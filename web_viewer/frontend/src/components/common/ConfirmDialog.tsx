interface ConfirmDialogProps {
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  destructive?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({
  title,
  message,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  destructive = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  return (
    <div style={styles.overlay} onClick={onCancel}>
      <div style={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <h3 style={styles.title}>{title}</h3>
        <p style={styles.message}>{message}</p>
        <div style={styles.actions}>
          <button style={styles.cancelBtn} onClick={onCancel}>
            {cancelLabel}
          </button>
          <button
            style={{
              ...styles.confirmBtn,
              background: destructive ? '#dc2626' : '#2563eb',
            }}
            onClick={onConfirm}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(0,0,0,0.6)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  dialog: {
    background: '#1a1a2e',
    borderRadius: '8px',
    padding: '20px',
    width: '360px',
    border: '1px solid #333',
  },
  title: {
    margin: '0 0 8px',
    fontSize: '16px',
    color: '#eee',
  },
  message: {
    margin: '0 0 20px',
    fontSize: '13px',
    color: '#888',
    lineHeight: '1.5',
  },
  actions: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: '8px',
  },
  cancelBtn: {
    padding: '6px 16px',
    background: 'transparent',
    border: '1px solid #444',
    borderRadius: '4px',
    color: '#888',
    cursor: 'pointer',
    fontSize: '12px',
  },
  confirmBtn: {
    padding: '6px 16px',
    border: 'none',
    borderRadius: '4px',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '12px',
  },
};
