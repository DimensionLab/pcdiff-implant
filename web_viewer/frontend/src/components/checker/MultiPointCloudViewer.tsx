/**
 * Multi-layer point cloud viewer for the Implant Checker.
 *
 * Renders multiple point clouds in a single React Three Fiber Canvas
 * with a shared coordinate space. Unlike PointCloudViewer (which centers
 * each cloud individually), all clouds share a single centering offset
 * so skull and implant remain correctly aligned.
 */
import { useEffect, useMemo, useRef } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import * as THREE from 'three';
import { useMultiPointCloudData } from '../../hooks/useMultiPointCloudData';
import { MeshLayer } from './MeshLayer';
import type { CheckerLayer } from '../../types/checker';

interface MultiPointCloudViewerProps {
  layers: CheckerLayer[];
  pointSize: number;
  showGrid: boolean;
  showAxes: boolean;
  /** Map of layerId -> Float32Array of RGB vertex colors (N*3) for heatmap */
  heatmapColors: Map<string, Float32Array>;
  /** Map of layerId -> ArrayBuffer of STL binary data for mesh layers */
  meshData: Map<string, ArrayBuffer>;
}

export function MultiPointCloudViewer({
  layers,
  pointSize,
  showGrid,
  showAxes,
  heatmapColors,
  meshData,
}: MultiPointCloudViewerProps) {
  // Fetch NPY data for point cloud layers in parallel
  const pointLayers = layers.filter((l) => l.layerType === 'points');
  const pcIds = pointLayers.map((l) => l.pointCloudId);
  const queries = useMultiPointCloudData(pcIds);

  // Parse all loaded data and compute shared bounding box
  const { parsedLayers, sharedOffset, boundingRadius } = useMemo(() => {
    const parsed: Map<string, Float32Array> = new Map();

    for (let i = 0; i < pointLayers.length; i++) {
      const q = queries[i];
      if (q?.data) {
        parsed.set(pointLayers[i].id, parseNpyPositions(q.data));
      }
    }

    if (parsed.size === 0) {
      return { parsedLayers: parsed, sharedOffset: [0, 0, 0] as number[], boundingRadius: 1 };
    }

    // Compute shared centroid across all point clouds
    let totalX = 0, totalY = 0, totalZ = 0, totalN = 0;
    for (const pts of parsed.values()) {
      const n = pts.length / 3;
      for (let j = 0; j < n; j++) {
        totalX += pts[j * 3];
        totalY += pts[j * 3 + 1];
        totalZ += pts[j * 3 + 2];
      }
      totalN += n;
    }

    const cx = totalN > 0 ? totalX / totalN : 0;
    const cy = totalN > 0 ? totalY / totalN : 0;
    const cz = totalN > 0 ? totalZ / totalN : 0;

    // Compute max distance from shared centroid
    let maxR2 = 0;
    for (const pts of parsed.values()) {
      const n = pts.length / 3;
      for (let j = 0; j < n; j++) {
        const dx = pts[j * 3] - cx;
        const dy = pts[j * 3 + 1] - cy;
        const dz = pts[j * 3 + 2] - cz;
        const r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > maxR2) maxR2 = r2;
      }
    }

    return {
      parsedLayers: parsed,
      sharedOffset: [cx, cy, cz],
      boundingRadius: Math.sqrt(maxR2) || 1,
    };
  }, [pointLayers, queries]);

  const r = boundingRadius;
  const anyLoading = queries.some((q) => q.isLoading);
  const hasData = parsedLayers.size > 0;

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas
        camera={{ position: [0, 0, 3], fov: 50, near: 0.01, far: 10000 }}
        style={{ background: '#0a0a1a' }}
      >
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />

        {/* Point cloud layers */}
        {layers.filter((l) => l.layerType === 'points').map((layer) => {
          const positions = parsedLayers.get(layer.id);
          if (!positions || !layer.visible) return null;
          const hColors = layer.useHeatmap ? heatmapColors.get(layer.id) ?? null : null;
          return (
            <PointCloudLayer
              key={layer.id}
              positions={positions}
              offset={sharedOffset}
              color={layer.color}
              heatmapColors={hColors}
              pointSize={pointSize}
            />
          );
        })}

        {/* Mesh layers */}
        {layers.filter((l) => l.layerType === 'mesh').map((layer) => {
          const stlBuf = meshData.get(layer.id);
          if (!stlBuf || !layer.visible) return null;
          return (
            <MeshLayer
              key={layer.id}
              stlBuffer={stlBuf}
              offset={sharedOffset}
              color={layer.color}
              opacity={layer.opacity}
            />
          );
        })}

        {hasData && <SharedCameraFit radius={boundingRadius} />}

        {showGrid && (
          <Grid
            args={[r * 4, r * 4]}
            cellSize={r * 0.1}
            cellColor="#1a1a3e"
            sectionColor="#2a2a5e"
            fadeDistance={r * 3}
            infiniteGrid
          />
        )}
        {showAxes && <axesHelper args={[r * 0.5]} />}

        <OrbitControls
          enableDamping
          dampingFactor={0.1}
          minDistance={r * 0.05}
          maxDistance={r * 10}
        />
      </Canvas>

      {anyLoading && (
        <div style={styles.loadingOverlay}>
          <div style={styles.loadingText}>Loading point clouds...</div>
        </div>
      )}

      {!hasData && !anyLoading && (
        <div style={styles.emptyOverlay}>
          <div style={styles.emptyText}>
            Add point clouds from the sidebar to start the implant check
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Renders a single point cloud layer with shared centering offset. */
function PointCloudLayer({
  positions,
  offset,
  color,
  heatmapColors,
  pointSize,
}: {
  positions: Float32Array;
  offset: number[];
  color: string;
  heatmapColors: Float32Array | null;
  pointSize: number;
}) {
  const geomRef = useRef<THREE.BufferGeometry>(null);
  const matRef = useRef<THREE.PointsMaterial>(null);
  const numPoints = positions.length / 3;

  // Apply shared centering offset to positions
  const centeredPositions = useMemo(() => {
    const centered = new Float32Array(positions.length);
    const [ox, oy, oz] = offset;
    for (let i = 0; i < numPoints; i++) {
      centered[i * 3] = positions[i * 3] - ox;
      centered[i * 3 + 1] = positions[i * 3 + 1] - oy;
      centered[i * 3 + 2] = positions[i * 3 + 2] - oz;
    }
    return centered;
  }, [positions, offset, numPoints]);

  // Update material size when pointSize changes
  useEffect(() => {
    if (matRef.current) {
      matRef.current.size = pointSize;
      matRef.current.needsUpdate = true;
    }
  }, [pointSize]);

  // Update vertex colors when heatmapColors change
  useEffect(() => {
    if (geomRef.current && heatmapColors) {
      const colorAttr = geomRef.current.getAttribute('color');
      if (colorAttr) {
        (colorAttr as THREE.BufferAttribute).array = heatmapColors;
        (colorAttr as THREE.BufferAttribute).needsUpdate = true;
      }
    }
  }, [heatmapColors]);

  return (
    <points>
      <bufferGeometry ref={geomRef}>
        <bufferAttribute
          attach="attributes-position"
          args={[centeredPositions, 3]}
          count={numPoints}
          itemSize={3}
        />
        {heatmapColors && (
          <bufferAttribute
            attach="attributes-color"
            args={[heatmapColors, 3]}
            count={numPoints}
            itemSize={3}
          />
        )}
      </bufferGeometry>
      <pointsMaterial
        ref={matRef}
        size={pointSize}
        sizeAttenuation
        vertexColors={!!heatmapColors}
        color={heatmapColors ? undefined : color}
      />
    </points>
  );
}

/** Imperatively repositions camera to fit the shared bounding sphere. */
function SharedCameraFit({ radius }: { radius: number }) {
  const { camera } = useThree();
  useEffect(() => {
    if (radius > 0) {
      const distance = radius * 2.5;
      camera.position.set(0, 0, distance);
      camera.lookAt(0, 0, 0);
      if ('far' in camera) {
        (camera as THREE.PerspectiveCamera).far = distance * 10;
        (camera as THREE.PerspectiveCamera).near = distance * 0.001;
      }
      camera.updateProjectionMatrix();
    }
  }, [radius, camera]);
  return null;
}

// ---------------------------------------------------------------------------
// NPY parser (shared with PointCloudViewer)
// ---------------------------------------------------------------------------

function parseNpyPositions(buffer: ArrayBuffer): Float32Array {
  const view = new DataView(buffer);
  const headerLen = view.getUint16(8, true);
  const dataOffset = 10 + headerLen;

  const headerBytes = new Uint8Array(buffer, 10, headerLen);
  const header = new TextDecoder().decode(headerBytes);

  if (header.includes('float64') || header.includes('<f8')) {
    const float64 = new Float64Array(buffer, dataOffset);
    return new Float32Array(float64);
  }

  return new Float32Array(buffer, dataOffset);
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles: Record<string, React.CSSProperties> = {
  loadingOverlay: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'rgba(0,0,0,0.5)',
    pointerEvents: 'none',
    zIndex: 10,
  },
  loadingText: { color: '#fff', fontSize: '14px' },
  emptyOverlay: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    pointerEvents: 'none',
  },
  emptyText: { color: '#555', fontSize: '14px' },
};
