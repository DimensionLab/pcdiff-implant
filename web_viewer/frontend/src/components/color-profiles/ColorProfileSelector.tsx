/**
 * Dropdown selector for choosing a color profile.
 * Shows gradient preview for each profile.
 */
import { useColorProfiles } from '../../hooks/useColorProfiles';
import { parseColorStops } from '../../types/color-profile';
import { GradientBar } from './GradientBar';

interface ColorProfileSelectorProps {
  selectedId: string | null;
  onSelect: (profileId: string | null) => void;
}

export function ColorProfileSelector({ selectedId, onSelect }: ColorProfileSelectorProps) {
  const { data: profiles, isLoading } = useColorProfiles();

  if (isLoading) {
    return <div style={styles.loading}>Loading profiles...</div>;
  }

  return (
    <div style={styles.container}>
      <label style={styles.label}>Color Profile</label>

      {/* None option */}
      <button
        style={{
          ...styles.item,
          ...(selectedId === null ? styles.itemActive : {}),
        }}
        onClick={() => onSelect(null)}
      >
        <span style={styles.name}>None (uniform color)</span>
      </button>

      {profiles?.map((profile) => {
        const stops = parseColorStops(profile);
        const isSelected = selectedId === profile.id;

        return (
          <button
            key={profile.id}
            style={{
              ...styles.item,
              ...(isSelected ? styles.itemActive : {}),
            }}
            onClick={() => onSelect(profile.id)}
          >
            <span style={styles.name}>
              {profile.name}
              {profile.is_default && <span style={styles.defaultBadge}>default</span>}
            </span>
            <GradientBar stops={stops} height={10} />
            <span style={styles.range}>
              {profile.sdf_range_min.toFixed(1)} to {profile.sdf_range_max.toFixed(1)}
            </span>
          </button>
        );
      })}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
  label: {
    fontSize: '10px',
    color: '#888',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    marginBottom: '4px',
  },
  loading: {
    fontSize: '11px',
    color: '#666',
    padding: '8px 0',
  },
  item: {
    display: 'flex',
    flexDirection: 'column',
    gap: '3px',
    padding: '6px 8px',
    background: 'transparent',
    border: '1px solid transparent',
    borderRadius: '4px',
    cursor: 'pointer',
    textAlign: 'left',
    color: '#ccc',
    width: '100%',
  },
  itemActive: {
    background: '#1a1a3e',
    borderColor: '#2563eb',
  },
  name: {
    fontSize: '11px',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  defaultBadge: {
    fontSize: '9px',
    color: '#888',
    background: '#222',
    padding: '1px 4px',
    borderRadius: '3px',
  },
  range: {
    fontSize: '9px',
    color: '#666',
  },
};
