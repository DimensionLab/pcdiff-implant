export interface ColorStop {
  value: number;  // 0.0 to 1.0 (normalised position)
  color: string;  // hex "#rrggbb"
}

export interface ColorProfile {
  id: string;
  name: string;
  description: string | null;
  color_map_type: 'diverging' | 'sequential' | 'categorical';
  color_stops: string;  // JSON stringified ColorStop[]
  sdf_range_min: number;
  sdf_range_max: number;
  is_default: boolean;
  created_at: string;
  updated_at: string;
  created_by: string;
}

export interface ColorProfileCreate {
  name: string;
  description?: string;
  color_map_type: string;
  color_stops: string;
  sdf_range_min?: number;
  sdf_range_max?: number;
  is_default?: boolean;
}

/** Parsed color stops for rendering */
export function parseColorStops(profile: ColorProfile): ColorStop[] {
  try {
    return JSON.parse(profile.color_stops) as ColorStop[];
  } catch {
    return [];
  }
}

/** Interpolate between color stops at normalised position t (0-1) */
export function interpolateColor(stops: ColorStop[], t: number): [number, number, number] {
  if (stops.length === 0) return [1, 1, 1];
  if (stops.length === 1) return hexToRgb01(stops[0].color);

  const clamped = Math.max(0, Math.min(1, t));

  // Find the two stops to interpolate between
  let lower = stops[0];
  let upper = stops[stops.length - 1];

  for (let i = 0; i < stops.length - 1; i++) {
    if (clamped >= stops[i].value && clamped <= stops[i + 1].value) {
      lower = stops[i];
      upper = stops[i + 1];
      break;
    }
  }

  const range = upper.value - lower.value;
  const factor = range > 0 ? (clamped - lower.value) / range : 0;

  const [lr, lg, lb] = hexToRgb01(lower.color);
  const [ur, ug, ub] = hexToRgb01(upper.color);

  return [
    lr + (ur - lr) * factor,
    lg + (ug - lg) * factor,
    lb + (ub - lb) * factor,
  ];
}

function hexToRgb01(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  return [
    parseInt(h.substring(0, 2), 16) / 255,
    parseInt(h.substring(2, 4), 16) / 255,
    parseInt(h.substring(4, 6), 16) / 255,
  ];
}
