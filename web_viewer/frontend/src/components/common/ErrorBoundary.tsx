import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;

      return (
        <div style={styles.container}>
          <div style={styles.icon}>!</div>
          <h3 style={styles.title}>Something went wrong</h3>
          <p style={styles.message}>{this.state.error?.message}</p>
          <button
            style={styles.button}
            onClick={() => this.setState({ hasError: false, error: null })}
          >
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '40px',
    gap: '12px',
  },
  icon: {
    width: '40px',
    height: '40px',
    borderRadius: '50%',
    background: '#dc2626',
    color: '#fff',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '20px',
    fontWeight: 'bold',
  },
  title: {
    margin: 0,
    color: '#eee',
    fontSize: '16px',
  },
  message: {
    margin: 0,
    color: '#888',
    fontSize: '12px',
    maxWidth: '400px',
    textAlign: 'center',
  },
  button: {
    padding: '6px 16px',
    background: '#2563eb',
    border: 'none',
    borderRadius: '4px',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '12px',
    marginTop: '8px',
  },
};
