import { useQuery } from '@tanstack/react-query';
import { generationApi } from '../../services/generation-api';
import type { CSSProperties } from 'react';

const FORMAT_LABELS: Record<string, string> = {
  nrrd: 'NRRD',
  stl: 'STL',
  ply: 'PLY',
  npy: 'NPY',
};

interface JobDownloadsProps {
  jobId: string;
}

export function JobDownloads({ jobId }: JobDownloadsProps) {
  const { data, isLoading } = useQuery({
    queryKey: ['job-artifacts', jobId],
    queryFn: () => generationApi.listArtifacts(jobId),
    staleTime: 60_000,
  });

  if (isLoading) return <div style={styles.loading}>Loading artifacts…</div>;

  const artifacts = data?.artifacts ?? [];
  if (artifacts.length === 0) return <div style={styles.empty}>No downloads available</div>;

  return (
    <div style={styles.container}>
      <div style={styles.label}>Download</div>
      <div style={styles.btnRow}>
        {artifacts.map((a) => (
          <a
            key={a.format}
            href={generationApi.downloadUrl(jobId, a.format)}
            download
            style={styles.btn}
            title={`Download ${FORMAT_LABELS[a.format] || a.format}`}
          >
            {FORMAT_LABELS[a.format] || a.format}
          </a>
        ))}
      </div>
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
    marginTop: '6px',
  },
  label: {
    fontSize: '10px',
    color: '#888',
    textTransform: 'uppercase',
    letterSpacing: '0.3px',
  },
  btnRow: {
    display: 'flex',
    gap: '4px',
    flexWrap: 'wrap',
  },
  btn: {
    padding: '4px 8px',
    fontSize: '10px',
    fontWeight: 600,
    background: '#1e3a5f',
    color: '#93c5fd',
    borderRadius: '3px',
    border: '1px solid #2563eb44',
    textDecoration: 'none',
    cursor: 'pointer',
    textAlign: 'center',
  },
  loading: {
    fontSize: '10px',
    color: '#666',
    marginTop: '4px',
  },
  empty: {
    fontSize: '10px',
    color: '#555',
    marginTop: '4px',
  },
};
