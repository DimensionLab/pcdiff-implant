/**
 * Settings page for CrAInial application.
 *
 * Allows users to configure:
 * - Inference device (CUDA, MPS, CPU)
 * - Default generation parameters
 * - Model paths
 */
import { useState, useEffect, type CSSProperties } from 'react';
import { useSettings, useSystemInfo, useUpdateSettings } from '../hooks/useSettings';
import type { SettingsUpdate } from '../types/settings';

export function SettingsPage() {
  const { data: settings, isLoading: settingsLoading } = useSettings();
  const { data: systemInfo, isLoading: systemInfoLoading } = useSystemInfo();
  const updateSettings = useUpdateSettings();

  // Local form state
  const [localSettings, setLocalSettings] = useState<SettingsUpdate>({});
  const [hasChanges, setHasChanges] = useState(false);

  // Sync local state when settings load
  useEffect(() => {
    if (settings) {
      setLocalSettings({
        inference_device: settings.inference_device,
        default_sampling_method: settings.default_sampling_method,
        default_sampling_steps: settings.default_sampling_steps,
        default_ensemble_count: settings.default_ensemble_count,
        pcdiff_model_path: settings.pcdiff_model_path,
        voxelization_model_path: settings.voxelization_model_path,
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
      },
    });
  };

  const handleReset = () => {
    if (settings) {
      setLocalSettings({
        inference_device: settings.inference_device,
        default_sampling_method: settings.default_sampling_method,
        default_sampling_steps: settings.default_sampling_steps,
        default_ensemble_count: settings.default_ensemble_count,
        pcdiff_model_path: settings.pcdiff_model_path,
        voxelization_model_path: settings.voxelization_model_path,
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

  const getDeviceDescription = (device: string) => {
    switch (device) {
      case 'auto':
        return 'Automatically select the best available device';
      case 'cuda':
        return 'NVIDIA GPU (fastest, requires CUDA)';
      case 'mps':
        return 'Apple Silicon GPU (Metal Performance Shaders)';
      case 'cpu':
        return 'CPU only (slowest, always available)';
      default:
        return '';
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.content}>
        <header style={styles.header}>
          <h1 style={styles.title}>Settings</h1>
          <p style={styles.subtitle}>Configure CrAInial application behavior</p>
        </header>

        {/* System Information */}
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
                <span style={styles.infoLabel}>Active Backend</span>
                <span
                  style={{
                    ...styles.infoValue,
                    color: systemInfo?.backend_type === 'cuda' ? '#10b981' : '#f59e0b',
                  }}
                >
                  {systemInfo?.backend_type?.toUpperCase() ?? 'Unknown'}
                </span>
              </div>
            </div>
          </div>
        </section>

        {/* Inference Settings */}
        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>Inference Settings</h2>
          <div style={styles.card}>
            <div style={styles.formGroup}>
              <label style={styles.label}>Inference Device</label>
              <select
                style={styles.select}
                value={localSettings.inference_device ?? 'auto'}
                onChange={(e) =>
                  handleChange('inference_device', e.target.value as 'auto' | 'cuda' | 'mps' | 'cpu')
                }
              >
                <option value="auto">Auto (Recommended)</option>
                <option value="cuda" disabled={!systemInfo?.cuda_available}>
                  CUDA (NVIDIA GPU){!systemInfo?.cuda_available ? ' - Not Available' : ''}
                </option>
                <option value="mps" disabled={!systemInfo?.mps_available}>
                  MPS (Apple Silicon){!systemInfo?.mps_available ? ' - Not Available' : ''}
                </option>
                <option value="cpu">CPU</option>
              </select>
              <p style={styles.hint}>{getDeviceDescription(localSettings.inference_device ?? 'auto')}</p>
            </div>
          </div>
        </section>

        {/* Default Generation Parameters */}
        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>Default Generation Parameters</h2>
          <div style={styles.card}>
            <div style={styles.formGroup}>
              <label style={styles.label}>Sampling Method</label>
              <div style={styles.radioGroup}>
                <label style={styles.radioLabel}>
                  <input
                    type="radio"
                    name="sampling_method"
                    checked={localSettings.default_sampling_method === 'ddim'}
                    onChange={() => handleChange('default_sampling_method', 'ddim')}
                  />
                  <span>DDIM (Fast)</span>
                </label>
                <label style={styles.radioLabel}>
                  <input
                    type="radio"
                    name="sampling_method"
                    checked={localSettings.default_sampling_method === 'ddpm'}
                    onChange={() => handleChange('default_sampling_method', 'ddpm')}
                  />
                  <span>DDPM (High Quality)</span>
                </label>
              </div>
              <p style={styles.hint}>
                DDIM is ~20x faster with minimal quality loss. DDPM produces highest quality results.
              </p>
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>
                Default Sampling Steps: {localSettings.default_sampling_steps}
              </label>
              <input
                type="range"
                style={styles.slider}
                min={localSettings.default_sampling_method === 'ddim' ? 25 : 100}
                max={localSettings.default_sampling_method === 'ddim' ? 100 : 1000}
                step={localSettings.default_sampling_method === 'ddim' ? 5 : 50}
                value={localSettings.default_sampling_steps ?? 50}
                onChange={(e) => handleChange('default_sampling_steps', Number(e.target.value))}
              />
              <p style={styles.hint}>
                More steps = higher quality but slower. Recommended: 50 for DDIM, 1000 for DDPM.
              </p>
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>
                Default Ensemble Count: {localSettings.default_ensemble_count}
              </label>
              <input
                type="range"
                style={styles.slider}
                min={1}
                max={10}
                value={localSettings.default_ensemble_count ?? 5}
                onChange={(e) => handleChange('default_ensemble_count', Number(e.target.value))}
              />
              <p style={styles.hint}>
                Generate multiple implants to compare and select the best result.
              </p>
            </div>
          </div>
        </section>

        {/* Advanced Settings */}
        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>Advanced Settings</h2>
          <div style={styles.card}>
            <div style={styles.formGroup}>
              <label style={styles.label}>PCDiff Model Path</label>
              <input
                type="text"
                style={styles.input}
                value={localSettings.pcdiff_model_path ?? ''}
                onChange={(e) => handleChange('pcdiff_model_path', e.target.value)}
                placeholder="pcdiff/checkpoints/pcdiff_model_best.pth"
              />
              <p style={styles.hint}>Relative path to the PCDiff model checkpoint from project root.</p>
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>Voxelization Model Path</label>
              <input
                type="text"
                style={styles.input}
                value={localSettings.voxelization_model_path ?? ''}
                onChange={(e) => handleChange('voxelization_model_path', e.target.value)}
                placeholder="voxelization/checkpoints/model_best.pt"
              />
              <p style={styles.hint}>Relative path to the voxelization model checkpoint from project root.</p>
            </div>
          </div>
        </section>

        {/* Save/Reset Buttons */}
        <div style={styles.actions}>
          <button
            style={{
              ...styles.button,
              ...styles.buttonSecondary,
            }}
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
  radioGroup: {
    display: 'flex',
    gap: '24px',
  },
  radioLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '14px',
    color: '#ccc',
    cursor: 'pointer',
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
};
