import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Routes, Route, Link, useLocation } from 'react-router-dom';
import { ErrorBoundary } from './components/common/ErrorBoundary';
import { DataViewerPage } from './pages/DataViewerPage';
import { ImplantCheckerPage } from './pages/ImplantCheckerPage';
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
      <ErrorBoundary>
        <div style={rootStyles.app}>
          <NavHeader />
          <div style={rootStyles.pageContent}>
            <Routes>
              <Route path="/" element={<DataViewerPage />} />
              <Route path="/checker" element={<ImplantCheckerPage />} />
            </Routes>
          </div>
        </div>
      </ErrorBoundary>
    </QueryClientProvider>
  );
}

function NavHeader() {
  const location = useLocation();
  const isChecker = location.pathname === '/checker';

  return (
    <header style={headerStyles.header}>
      <div style={headerStyles.left}>
        <span style={headerStyles.title}>DimensionLab CrAInial</span>
        <nav style={headerStyles.nav}>
          <Link
            to="/"
            style={{
              ...headerStyles.navLink,
              ...(isChecker ? {} : headerStyles.navLinkActive),
            }}
          >
            Data Viewer
          </Link>
          <Link
            to="/checker"
            style={{
              ...headerStyles.navLink,
              ...(isChecker ? headerStyles.navLinkActive : {}),
            }}
          >
            Implant Checker
          </Link>
        </nav>
      </div>
      <span style={headerStyles.version}>v2.0.0</span>
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
};

export default App;
