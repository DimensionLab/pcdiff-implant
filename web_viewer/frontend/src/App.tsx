import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Routes, Route, Link, useLocation } from 'react-router-dom';
import { ErrorBoundary } from './components/common/ErrorBoundary';
import { NotificationBell } from './components/common/NotificationBell';
import { NotificationProvider } from './context/NotificationContext';
import { DataViewerPage } from './pages/DataViewerPage';
import { DoctorPortalPage } from './pages/DoctorPortalPage';
import { ImplantCheckerPage } from './pages/ImplantCheckerPage';
import { ImplantGeneratorPage } from './pages/ImplantGeneratorPage';
import { SettingsPage } from './pages/SettingsPage';
import './App.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30_000,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <NotificationProvider>
        <ErrorBoundary>
          <div style={rootStyles.app}>
            <NavHeader />
            <div style={rootStyles.pageContent}>
              <Routes>
                <Route path="/" element={<DataViewerPage />} />
                <Route path="/portal" element={<DoctorPortalPage />} />
                <Route path="/generator" element={<ImplantGeneratorPage />} />
                <Route path="/checker" element={<ImplantCheckerPage />} />
                <Route path="/settings" element={<SettingsPage />} />
              </Routes>
            </div>
          </div>
        </ErrorBoundary>
      </NotificationProvider>
    </QueryClientProvider>
  );
}

function NavHeader() {
  const location = useLocation();
  const currentPath = location.pathname;

  const navItems = [
    { path: '/portal', label: 'Doctor Portal' },
    { path: '/', label: 'Data Viewer' },
    { path: '/generator', label: 'Implant Generator' },
    { path: '/checker', label: 'Implant Checker' },
  ];

  return (
    <header style={headerStyles.header}>
      <div style={headerStyles.left}>
        <span style={headerStyles.title}>DimensionLab CrAInial</span>
        <nav style={headerStyles.nav}>
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              style={{
                ...headerStyles.navLink,
                ...(currentPath === item.path ? headerStyles.navLinkActive : {}),
              }}
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
      <div style={headerStyles.right}>
        <NotificationBell />
        <Link
          to="/settings"
          style={{
            ...headerStyles.settingsLink,
            ...(currentPath === '/settings' ? headerStyles.settingsLinkActive : {}),
          }}
          title="Settings"
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        </Link>
        <span style={headerStyles.version}>v2.0.0</span>
      </div>
    </header>
  );
}

const rootStyles: Record<string, React.CSSProperties> = {
  app: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    overflow: 'hidden',
    background: '#0a0a1a',
    color: '#eee',
  },
  pageContent: {
    flex: 1,
    overflow: 'hidden',
  },
};

const headerStyles: Record<string, React.CSSProperties> = {
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
  left: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
  },
  right: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  title: {
    fontSize: '15px',
    fontWeight: 700,
    color: '#fff',
  },
  nav: {
    display: 'flex',
    gap: '4px',
  },
  navLink: {
    padding: '4px 10px',
    fontSize: '12px',
    color: '#888',
    textDecoration: 'none',
    borderRadius: '4px',
    transition: 'color 0.15s, background 0.15s',
  },
  navLinkActive: {
    color: '#fff',
    background: 'rgba(255,255,255,0.1)',
  },
  version: {
    fontSize: '10px',
    color: '#555',
  },
  settingsLink: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '32px',
    height: '32px',
    color: '#888',
    borderRadius: '6px',
    transition: 'color 0.15s, background 0.15s',
  },
  settingsLinkActive: {
    color: '#fff',
    background: 'rgba(255,255,255,0.1)',
  },
};

export default App;
