import { useMemo } from 'react';
import { usePointCloudSDF } from './usePointClouds';
import { useColorProfile } from './useColorProfiles';
import type { ColorProfile } from '../types/color-profile';
import { parseColorStops, interpolateColor } from '../types/color-profile';

/**
 * Fetches SDF values for a point cloud and maps them to RGB colors
 * using the selected color profile. Color mapping runs client-side
 * for real-time profile switching.
 */
export function useSDFColors(
  pointCloudId: string | null,
  colorProfileId: string | null,
) {
  const { data: sdfData, isLoading: sdfLoading } = usePointCloudSDF(pointCloudId);
  const { data: profile, isLoading: profileLoading } = useColorProfile(colorProfileId);

  const colors = useMemo(() => {
    if (!sdfData || !profile) return null;
    return computeColors(sdfData, profile);
  }, [sdfData, profile]);

  return {
    colors,
    loading: sdfLoading || profileLoading,
    hasSDF: !!sdfData,
  };
}

function computeColors(sdf: Float32Array, profile: ColorProfile): Float32Array {
  const stops = parseColorStops(profile);
  if (stops.length === 0) {
    // Fallback: all white
    return new Float32Array(sdf.length * 3).fill(1.0);
  }

  const rangeMin = profile.sdf_range_min;
  const rangeMax = profile.sdf_range_max;
  const rangeSpan = rangeMax - rangeMin;

  const rgb = new Float32Array(sdf.length * 3);

  for (let i = 0; i < sdf.length; i++) {
    // Normalise SDF value to [0, 1]
    const t = rangeSpan > 0 ? (sdf[i] - rangeMin) / rangeSpan : 0.5;
    const [r, g, b] = interpolateColor(stops, t);
    rgb[i * 3] = r;
    rgb[i * 3 + 1] = g;
    rgb[i * 3 + 2] = b;
  }

  return rgb;
}
