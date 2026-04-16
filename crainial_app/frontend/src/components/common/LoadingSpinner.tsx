interface LoadingSpinnerProps {
  message?: string;
  size?: number;
}

export function LoadingSpinner({ message = 'Loading...', size = 24 }: LoadingSpinnerProps) {
  return (
    <div style={styles.container}>
      <div
        style={{
          ...styles.spinner,
          width: size,
          height: size,
          borderWidth: Math.max(2, size / 8),
        }}
      />
      {message && <div style={styles.message}>{message}</div>}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '12px',
    padding: '20px',
  },
  spinner: {
    borderStyle: 'solid',
    borderColor: '#333',
    borderTopColor: '#2563eb',
    borderRadius: '50%',
    animation: 'spin 0.8s linear infinite',
  },
  message: {
    fontSize: '12px',
    color: '#888',
  },
};
