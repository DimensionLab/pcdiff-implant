/**
 * Renders a horizontal gradient bar from color stops.
 * Used as a preview in the color profile selector and editor.
 */
import type { ColorStop } from '../../types/color-profile';

interface GradientBarProps {
  stops: ColorStop[];
  height?: number;
}

export function GradientBar({ stops, height = 16 }: GradientBarProps) {
  if (stops.length === 0) return null;

  const gradientCss = stops
    .map((s) => `${s.color} ${s.value * 100}%`)
    .join(', ');

  return (
    <div
      style={{
        height,
        borderRadius: '3px',
        background: `linear-gradient(to right, ${gradientCss})`,
        border: '1px solid #333',
      }}
    />
  );
}
