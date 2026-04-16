/**
 * Editor for creating a new color profile.
 * Provides a form with color stop inputs and live gradient preview.
 */
import { useState } from 'react';
import { useCreateColorProfile } from '../../hooks/useColorProfiles';
import type { ColorStop } from '../../types/color-profile';
import { GradientBar } from './GradientBar';

interface ColorProfileEditorProps {
  onClose: () => void;
}

const DEFAULT_STOPS: ColorStop[] = [
  { value: 0, color: '#0000ff' },
  { value: 0.5, color: '#ffffff' },
  { value: 1, color: '#ff0000' },
];

export function ColorProfileEditor({ onClose }: ColorProfileEditorProps) {
  const [name, setName] = useState('');
  const [mapType, setMapType] = useState<'diverging' | 'sequential' | 'categorical'>('diverging');
  const [stops, setStops] = useState<ColorStop[]>(DEFAULT_STOPS);
  const [rangeMin, setRangeMin] = useState(-5);
  const [rangeMax, setRangeMax] = useState(5);

  const createMutation = useCreateColorProfile();

  const updateStop = (index: number, field: 'value' | 'color', val: string) => {
    const updated = [...stops];
    if (field === 'value') {
      updated[index] = { ...updated[index], value: parseFloat(val) || 0 };
    } else {
      updated[index] = { ...updated[index], color: val };
    }
    setStops(updated);
  };

  const addStop = () => {
    setStops([...stops, { value: 1, color: '#888888' }]);
  };

  const removeStop = (index: number) => {
    if (stops.length <= 2) return;
    setStops(stops.filter((_, i) => i !== index));
  };

  const handleSave = () => {
    if (!name.trim()) return;
    const sorted = [...stops].sort((a, b) => a.value - b.value);
    createMutation.mutate(
      {
        name: name.trim(),
        color_map_type: mapType,
        color_stops: JSON.stringify(sorted),
        sdf_range_min: rangeMin,
        sdf_range_max: rangeMax,
      },
      { onSuccess: onClose },
    );
  };

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <h3 style={styles.title}>New Color Profile</h3>

        <label style={styles.label}>Name</label>
        <input
          style={styles.input}
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="My Color Profile"
        />

        <label style={styles.label}>Type</label>
        <select
          style={styles.input}
          value={mapType}
          onChange={(e) => setMapType(e.target.value as typeof mapType)}
        >
          <option value="diverging">Diverging</option>
          <option value="sequential">Sequential</option>
          <option value="categorical">Categorical</option>
        </select>

        <label style={styles.label}>SDF Range</label>
        <div style={{ display: 'flex', gap: '8px' }}>
          <input
            style={{ ...styles.input, flex: 1 }}
            type="number"
            step="0.5"
            value={rangeMin}
            onChange={(e) => setRangeMin(parseFloat(e.target.value) || 0)}
          />
          <span style={{ color: '#666', alignSelf: 'center' }}>to</span>
          <input
            style={{ ...styles.input, flex: 1 }}
            type="number"
            step="0.5"
            value={rangeMax}
            onChange={(e) => setRangeMax(parseFloat(e.target.value) || 0)}
          />
        </div>

        <label style={styles.label}>Color Stops</label>
        {stops.map((stop, i) => (
          <div key={i} style={styles.stopRow}>
            <input
              type="color"
              value={stop.color}
              onChange={(e) => updateStop(i, 'color', e.target.value)}
              style={styles.colorInput}
            />
            <input
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={stop.value}
              onChange={(e) => updateStop(i, 'value', e.target.value)}
              style={{ ...styles.input, width: '70px' }}
            />
            <button
              style={styles.removeBtn}
              onClick={() => removeStop(i)}
              disabled={stops.length <= 2}
            >
              x
            </button>
          </div>
        ))}
        <button style={styles.addBtn} onClick={addStop}>
          + Add Stop
        </button>

        <label style={styles.label}>Preview</label>
        <GradientBar stops={[...stops].sort((a, b) => a.value - b.value)} height={20} />

        <div style={styles.actions}>
          <button style={styles.cancelBtn} onClick={onClose}>
            Cancel
          </button>
          <button
            style={styles.saveBtn}
            onClick={handleSave}
            disabled={!name.trim() || createMutation.isPending}
          >
            {createMutation.isPending ? 'Saving...' : 'Save'}
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
    width: '380px',
    maxHeight: '80vh',
    overflowY: 'auto',
    border: '1px solid #333',
  },
  title: {
    margin: '0 0 16px',
    fontSize: '16px',
    color: '#eee',
  },
  label: {
    display: 'block',
    fontSize: '10px',
    color: '#888',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    marginTop: '12px',
    marginBottom: '4px',
  },
  input: {
    width: '100%',
    padding: '6px 8px',
    background: '#111128',
    border: '1px solid #333',
    borderRadius: '4px',
    color: '#ccc',
    fontSize: '12px',
    boxSizing: 'border-box',
  },
  stopRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    marginBottom: '4px',
  },
  colorInput: {
    width: '32px',
    height: '28px',
    padding: '0',
    border: '1px solid #333',
    borderRadius: '3px',
    cursor: 'pointer',
    background: 'transparent',
  },
  removeBtn: {
    padding: '4px 8px',
    background: 'transparent',
    border: '1px solid #444',
    borderRadius: '3px',
    color: '#888',
    cursor: 'pointer',
    fontSize: '11px',
  },
  addBtn: {
    padding: '4px 8px',
    background: 'transparent',
    border: '1px solid #333',
    borderRadius: '3px',
    color: '#888',
    cursor: 'pointer',
    fontSize: '11px',
    marginTop: '4px',
  },
  actions: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: '8px',
    marginTop: '20px',
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
  saveBtn: {
    padding: '6px 16px',
    background: '#2563eb',
    border: 'none',
    borderRadius: '4px',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '12px',
  },
};
