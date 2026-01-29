/**
 * Color legend bar for SDF visualization.
 * Shows the gradient and min/max SDF values.
 */
import type { ColorProfile } from '../../types/color-profile';
import { parseColorStops } from '../../types/color-profile';

interface SDFColorBarProps {
  profile: ColorProfile | null;
}

export function SDFColorBar({ profile }: SDFColorBarProps) {
  if (!profile) return null;

  const stops = parseColorStops(profile);
  if (stops.length === 0) return null;

  const gradientCss = stops
    .map((s) => `${s.color} ${s.value * 100}%`)
    .join(', ');

  return (
    <div style={styles.container}>
      <div style={styles.label}>{profile.name}</div>
      <div
        style={{
          ...styles.bar,
          background: `linear-gradient(to right, ${gradientCss})`,
        }}
      />
      <div style={styles.range}>
        <span>{profile.sdf_range_min.toFixed(1)}</span>
        <span>SDF</span>
        <span>{profile.sdf_range_max.toFixed(1)}</span>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '8px 12px',
    background: '#1a1a2e',
    borderRadius: '6px',
    border: '1px solid #333',
  },
  label: {
    fontSize: '11px',
    color: '#888',
    marginBottom: '6px',
  },
  bar: {
    height: '12px',
    borderRadius: '3px',
    marginBottom: '4px',
  },
  range: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '10px',
    color: '#666',
  },
};
