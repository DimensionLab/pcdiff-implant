/**
 * Main application layout with three-panel design:
 * - Left sidebar: Data Browser (scans + point clouds)
 * - Center: 3D Viewer (volume or point cloud)
 * - Right sidebar: Controls (viewer settings, color profiles)
 */
import type { ReactNode } from 'react';

interface AppLayoutProps {
  header?: ReactNode;
  sidebar: ReactNode;
  main: ReactNode;
  controls: ReactNode;
}

export function AppLayout({ header, sidebar, main, controls }: AppLayoutProps) {
  return (
    <div style={styles.root}>
      {header && <header style={styles.header}>{header}</header>}
      <div style={styles.body}>
        <aside style={styles.sidebar}>{sidebar}</aside>
        <main style={styles.main}>{main}</main>
        <aside style={styles.controls}>{controls}</aside>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    overflow: 'hidden',
    background: '#0a0a1a',
    color: '#eee',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '8px 16px',
    background: '#111128',
    borderBottom: '1px solid #222',
    minHeight: '40px',
    flexShrink: 0,
  },
  body: {
    display: 'flex',
    flex: 1,
    overflow: 'hidden',
  },
  sidebar: {
    width: '280px',
    flexShrink: 0,
    borderRight: '1px solid #222',
    overflowY: 'auto',
    background: '#0d0d20',
  },
  main: {
    flex: 1,
    overflow: 'hidden',
    position: 'relative',
  },
  controls: {
    width: '260px',
    flexShrink: 0,
    borderLeft: '1px solid #222',
    overflowY: 'auto',
    background: '#0d0d20',
    padding: '12px',
  },
};
