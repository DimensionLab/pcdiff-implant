/**
 * Settings page for the cran-2 CrAInial application.
 *
 * Configures:
 * - Inference device (informational only — heavy work runs on Runpod)
 * - cran-2 binarization threshold
 * - Cloud generation (Runpod serverless GPU + AWS S3)
 */
import { useEffect, useState, type CSSProperties } from 'react';
import { useSettings, useSystemInfo, useUpdateSettings } from '../hooks/useSettings';
import type { SettingsUpdate } from '../types/settings';

export function SettingsPage() {
  const { data: settings, isLoading: settingsLoading } = useSettings();
  const { data: systemInfo, isLoading: systemInfoLoading } = useSystemInfo();
  const updateSettings = useUpdateSettings();

  const [localSettings, setLocalSettings] = useState<SettingsUpdate>({});
  const [hasChanges, setHasChanges] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);

  useEffect(() => {
    if (settings) {
      setLocalSettings({
        inference_device: settings.inference_device,
        cran2_threshold: settings.cran2_threshold,
        cloud_generation_enabled: settings.cloud_generation_enabled,
        runpod_endpoint_id: settings.runpod_endpoint_id,
        aws_s3_bucket: settings.aws_s3_bucket,
        aws_s3_region: settings.aws_s3_region,
      });
      setHasChanges(false);
    }
  }, [settings]);

  const handleChange = (key: keyof SettingsUpdate, value: unknown) => {
    setLocalSettings((prev) => ({ ...prev, [key]: value }));
    setHasChanges(true);
  };

  const handleSave = () => {
    updateSettings.mutate(localSettings, {
      onSuccess: () => {
        setHasChanges(false);
        setLocalSettings((prev) => ({ ...prev, runpod_api_key: undefined }));
      },
    });
  };

  const handleReset = () => {
    if (settings) {
      setLocalSettings({
        inference_device: settings.inference_device,
        cran2_threshold: settings.cran2_threshold,
        cloud_generation_enabled: settings.cloud_generation_enabled,
        runpod_endpoint_id: settings.runpod_endpoint_id,
        aws_s3_bucket: settings.aws_s3_bucket,
        aws_s3_region: settings.aws_s3_region,
      });
      setHasChanges(false);
    }
  };

  if (settingsLoading || systemInfoLoading) {
    return (
      <div style={styles.container}>
        <div style={styles.loading}>Loading settings...</div>
      </div>
    );
  }

  const threshold = localSettings.cran2_threshold ?? 0.5;

  return (
    <div style={styles.container}>
      <div style={styles.content}>
        <header style={styles.header}>
          <h1 style={styles.title}>Settings</h1>
          <p style={styles.subtitle}>Configure cran-2 inference and cloud generation</p>
        </header>

        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>System Information</h2>
          <div style={styles.card}>
            <div style={styles.infoGrid}>
              <div style={styles.infoItem}>
                <span style={styles.infoLabel}>Platform</span>
                <span style={styles.infoValue}>{systemInfo?.platform ?? 'Unknown'}</span>
              </div>
              <div style={styles.infoItem}>
                <span style={styles.infoLabel}>Python Version</span>
                <span style={styles.infoValue}>{systemInfo?.python_version ?? 'Unknown'}</span>
              </div>
              <div style={styles.infoItem}>
                <span style={styles.infoLabel}>PyTorch Version</span>
                <span style={styles.infoValue}>{systemInfo?.torch_version ?? 'Unknown'}</span>
              </div>
              <div style={styles.infoItem}>
                <span style={styles.infoLabel}>CUDA Available</span>
                <span
                  style={{
                    ...styles.infoValue,
                    color: systemInfo?.cuda_available ? '#10b981' : '#888',
                  }}
                >
                  {systemInfo?.cuda_available ? 'Yes' : 'No'}
                </span>
              </div>
              <div style={styles.infoItem}>
                <span style={styles.infoLabel}>MPS (Apple Silicon) Available</span>
                <span
                  style={{
                    ...styles.infoValue,
                    color: systemInfo?.mps_available ? '#10b981' : '#888',
                  }}
                >
                  {systemInfo?.mps_available ? 'Yes' : 'No'}
                </span>
              </div>
              {systemInfo?.device_name && (
                <div style={styles.infoItem}>
                  <span style={styles.infoLabel}>GPU Device</span>
                  <span style={styles.infoValue}>{systemInfo.device_name}</span>
                </div>
              )}
              <div style={styles.infoItem}>
                <span style={styles.infoLabel}>cran-2 Endpoint</span>
                <span
                  style={{
                    ...styles.infoValue,
                    color: systemInfo?.cloud_configured ? '#10b981' : '#f59e0b',
                  }}
                >
                  {systemInfo?.cloud_configured ? 'Configured' : 'Not configured'}
                </span>
              </div>
            </div>
          </div>
        </section>

        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>cran-2 Inference</h2>
          <div style={styles.card}>
            <div style={styles.formGroup}>
              <label style={styles.label}>
                Binarization Threshold: {threshold.toFixed(2)}
              </label>
              <input
                type="range"
                style={styles.slider}
                min={0.05}
                max={0.95}
                step={0.05}
                value={threshold}
                onChange={(e) => handleChange('cran2_threshold', Number(e.target.value))}
              />
              <p style={styles.hint}>
                Cutoff applied to cran-2's predicted implant probability map. Lower thresholds keep
                more voxels (thicker implant); higher thresholds tighten the mask. Default: 0.5.
              </p>
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>Local Preview Device</label>
              <select
                style={styles.select}
                value={localSettings.inference_device ?? 'auto'}
                onChange={(e) =>
                  handleChange('inference_device', e.target.value as 'auto' | 'cuda' | 'mps' | 'cpu')
                }
              >
                <option value="auto">Auto</option>
                <option value="cuda" disabled={!systemInfo?.cuda_available}>
                  CUDA{!systemInfo?.cuda_available ? ' - Not Available' : ''}
                </option>
                <option value="mps" disabled={!systemInfo?.mps_available}>
                  MPS{!systemInfo?.mps_available ? ' - Not Available' : ''}
                </option>
                <option value="cpu">CPU</option>
              </select>
              <p style={styles.hint}>
                Used only for local 3D preview rendering — cran-2 inference itself runs on Runpod.
              </p>
            </div>
          </div>
        </section>

        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              Cloud Generation (Runpod GPU)
              {systemInfo?.cloud_configured && (
                <span style={styles.configuredBadge}>Configured</span>
              )}
            </span>
          </h2>
          <div style={styles.card}>
            <div style={styles.formGroup}>
              <label style={styles.toggleLabel}>
                <input
                  type="checkbox"
                  checked={localSettings.cloud_generation_enabled ?? false}
                  onChange={(e) => handleChange('cloud_generation_enabled', e.target.checked)}
                  style={styles.checkbox}
                />
                <span>Enable cran-2 Cloud Generation</span>
              </label>
              <p style={styles.hint}>
                Submit defective-skull NRRD volumes to your cran-2 Runpod endpoint. The implant
                NRRD is downloaded back into the local scan library.
              </p>
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>Runpod Endpoint ID</label>
              <input
                type="text"
                style={styles.input}
                value={localSettings.runpod_endpoint_id ?? ''}
                onChange={(e) => handleChange('runpod_endpoint_id', e.target.value)}
                placeholder="e.g., wferq1g3i1hhqd"
              />
              <p style={styles.hint}>The cran-2 serverless endpoint ID from the Runpod console.</p>
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>
                Runpod API Key
                {settings?.runpod_api_key_set && !localSettings.runpod_api_key && (
                  <span style={styles.keySetBadge}>Set</span>
                )}
              </label>
              <div style={styles.apiKeyRow}>
                <input
                  type={showApiKey ? 'text' : 'password'}
                  style={{ ...styles.input, flex: 1 }}
                  value={localSettings.runpod_api_key ?? ''}
                  onChange={(e) => handleChange('runpod_api_key', e.target.value)}
                  placeholder={settings?.runpod_api_key_set ? '••••••••••••••••' : 'Enter API key'}
                />
                <button
                  type="button"
                  style={styles.showButton}
                  onClick={() => setShowApiKey(!showApiKey)}
                >
                  {showApiKey ? 'Hide' : 'Show'}
                </button>
              </div>
              <p style={styles.hint}>
                Stored encrypted server-side. Can also be supplied via the RUNPOD_API_KEY env var.
              </p>
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>AWS S3 Bucket</label>
              <input
                type="text"
                style={styles.input}
                value={localSettings.aws_s3_bucket ?? ''}
                onChange={(e) => handleChange('aws_s3_bucket', e.target.value)}
                placeholder="test-crainial"
              />
              <p style={styles.hint}>S3 bucket cran-2 writes implant NRRDs to.</p>
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>AWS S3 Region</label>
              <select
                style={styles.select}
                value={localSettings.aws_s3_region ?? 'eu-central-1'}
                onChange={(e) => handleChange('aws_s3_region', e.target.value)}
              >
                <option value="us-east-1">US East (N. Virginia)</option>
                <option value="us-west-2">US West (Oregon)</option>
                <option value="eu-west-1">EU (Ireland)</option>
                <option value="eu-central-1">EU (Frankfurt)</option>
                <option value="ap-northeast-1">Asia Pacific (Tokyo)</option>
              </select>
            </div>
          </div>
        </section>

        <div style={styles.actions}>
          <button
            style={{ ...styles.button, ...styles.buttonSecondary }}
            onClick={handleReset}
            disabled={!hasChanges}
          >
            Reset
          </button>
          <button
            style={{
              ...styles.button,
              ...styles.buttonPrimary,
              opacity: hasChanges ? 1 : 0.5,
            }}
            onClick={handleSave}
            disabled={!hasChanges || updateSettings.isPending}
          >
            {updateSettings.isPending ? 'Saving...' : 'Save Changes'}
          </button>
        </div>

        {updateSettings.isError && (
          <div style={styles.error}>
            Failed to save settings: {(updateSettings.error as Error).message}
          </div>
        )}
      </div>
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  container: {
    height: '100%',
    overflow: 'auto',
    background: '#0a0a1a',
  },
  content: {
    maxWidth: '800px',
    margin: '0 auto',
    padding: '24px',
  },
  loading: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    color: '#888',
    fontSize: '14px',
  },
  header: {
    marginBottom: '32px',
  },
  title: {
    margin: 0,
    fontSize: '28px',
    fontWeight: 700,
    color: '#fff',
  },
  subtitle: {
    margin: '8px 0 0',
    fontSize: '14px',
    color: '#888',
  },
  section: {
    marginBottom: '32px',
  },
  sectionTitle: {
    margin: '0 0 16px',
    fontSize: '16px',
    fontWeight: 600,
    color: '#ccc',
  },
  card: {
    background: '#111128',
    borderRadius: '12px',
    border: '1px solid #222',
    padding: '20px',
  },
  infoGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
    gap: '16px',
  },
  infoItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  infoLabel: {
    fontSize: '11px',
    color: '#666',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  infoValue: {
    fontSize: '14px',
    color: '#fff',
    fontWeight: 500,
  },
  formGroup: {
    marginBottom: '20px',
  },
  label: {
    display: 'block',
    marginBottom: '8px',
    fontSize: '13px',
    fontWeight: 500,
    color: '#ccc',
  },
  select: {
    width: '100%',
    padding: '10px 12px',
    background: '#0a0a1a',
    color: '#fff',
    border: '1px solid #333',
    borderRadius: '6px',
    fontSize: '14px',
  },
  input: {
    width: '100%',
    padding: '10px 12px',
    background: '#0a0a1a',
    color: '#fff',
    border: '1px solid #333',
    borderRadius: '6px',
    fontSize: '14px',
    boxSizing: 'border-box',
  },
  slider: {
    width: '100%',
    accentColor: '#2563eb',
  },
  hint: {
    margin: '8px 0 0',
    fontSize: '12px',
    color: '#666',
  },
  actions: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: '12px',
    marginTop: '32px',
  },
  button: {
    padding: '12px 24px',
    borderRadius: '6px',
    fontSize: '14px',
    fontWeight: 600,
    cursor: 'pointer',
    border: 'none',
    transition: 'opacity 0.2s',
  },
  buttonPrimary: {
    background: '#2563eb',
    color: '#fff',
  },
  buttonSecondary: {
    background: '#333',
    color: '#ccc',
  },
  error: {
    marginTop: '16px',
    padding: '12px 16px',
    background: 'rgba(239, 68, 68, 0.1)',
    border: '1px solid rgba(239, 68, 68, 0.3)',
    borderRadius: '6px',
    color: '#ef4444',
    fontSize: '13px',
  },
  toggleLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    fontSize: '14px',
    color: '#fff',
    cursor: 'pointer',
  },
  checkbox: {
    width: '18px',
    height: '18px',
    accentColor: '#2563eb',
  },
  configuredBadge: {
    padding: '2px 8px',
    background: 'rgba(16, 185, 129, 0.2)',
    border: '1px solid rgba(16, 185, 129, 0.4)',
    borderRadius: '4px',
    fontSize: '11px',
    color: '#10b981',
    fontWeight: 500,
  },
  keySetBadge: {
    marginLeft: '8px',
    padding: '2px 6px',
    background: 'rgba(16, 185, 129, 0.2)',
    borderRadius: '4px',
    fontSize: '11px',
    color: '#10b981',
  },
  apiKeyRow: {
    display: 'flex',
    gap: '8px',
  },
  showButton: {
    padding: '10px 16px',
    background: '#333',
    color: '#ccc',
    border: 'none',
    borderRadius: '6px',
    fontSize: '13px',
    cursor: 'pointer',
  },
};
