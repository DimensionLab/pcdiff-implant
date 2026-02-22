/**
 * Point cloud viewer using React Three Fiber.
 *
 * Loads NPY point cloud data from the API, supports SDF colorization
 * via vertex colors. Auto-fits camera to the bounding sphere of the
 * loaded point cloud so data of any scale renders correctly.
 */
import { useEffect, useMemo, useRef } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { TrackballControls, Grid } from '@react-three/drei';
import * as THREE from 'three';
import { usePointCloudData } from '../../hooks/usePointClouds';
import { useSDFColors } from '../../hooks/useSDFColors';

interface PointCloudViewerProps {
  pointCloudId: string | null;
  colorProfileId: string | null;
  pointSize: number;
  showGrid: boolean;
  showAxes: boolean;
}

export function PointCloudViewer({
  pointCloudId,
  colorProfileId,
  pointSize,
  showGrid,
  showAxes,
}: PointCloudViewerProps) {
  const { data: rawData, isLoading: dataLoading } = usePointCloudData(pointCloudId);
  const { colors: sdfColors, loading: colorLoading } = useSDFColors(
    pointCloudId,
    colorProfileId,
  );

  // Parse NPY, center the cloud, and compute bounding sphere radius
  const { positions, boundingRadius } = useMemo(() => {
    if (!rawData) return { positions: null, boundingRadius: 0 };
    const pts = parseNpyPositions(rawData);
    const n = pts.length / 3;
    if (n === 0) return { positions: pts, boundingRadius: 0 };

    // Compute centroid
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < n; i++) {
      cx += pts[i * 3];
      cy += pts[i * 3 + 1];
      cz += pts[i * 3 + 2];
    }
    cx /= n; cy /= n; cz /= n;

    // Center and find max distance from origin
    let maxR2 = 0;
    for (let i = 0; i < n; i++) {
      pts[i * 3] -= cx;
      pts[i * 3 + 1] -= cy;
      pts[i * 3 + 2] -= cz;
      const r2 = pts[i * 3] ** 2 + pts[i * 3 + 1] ** 2 + pts[i * 3 + 2] ** 2;
      if (r2 > maxR2) maxR2 = r2;
    }
    return { positions: pts, boundingRadius: Math.sqrt(maxR2) };
  }, [rawData]);

  const loading = dataLoading || (colorProfileId ? colorLoading : false);
  const r = boundingRadius || 1;

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas
        camera={{ position: [0, 0, 3], fov: 50, near: 0.01, far: 10000 }}
        style={{ background: '#0a0a1a' }}
      >
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />

        {positions && (
          <>
            <PointCloudMesh
              positions={positions}
              colors={sdfColors}
              pointSize={pointSize}
            />
            <CameraFit radius={boundingRadius} />
          </>
        )}

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

        <TrackballControls
          rotateSpeed={2}
          zoomSpeed={1.2}
          panSpeed={0.8}
          minDistance={r * 0.05}
          maxDistance={r * 10}
          noPan={false}
          noRotate={false}
          noZoom={false}
          staticMoving={false}
          dynamicDampingFactor={0.1}
        />
      </Canvas>

      {loading && (
        <div style={styles.loadingOverlay}>
          <div style={styles.loadingText}>
            {dataLoading ? 'Loading point cloud...' : 'Computing colors...'}
          </div>
        </div>
      )}

      {!pointCloudId && !loading && (
        <div style={styles.emptyOverlay}>
          <div style={styles.emptyText}>
            Select a point cloud from the Data Browser
          </div>
        </div>
      )}
    </div>
  );
}

/** Imperatively repositions camera to fit the bounding sphere. */
function CameraFit({ radius }: { radius: number }) {
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

function PointCloudMesh({
  positions,
  colors,
  pointSize,
}: {
  positions: Float32Array;
  colors: Float32Array | null;
  pointSize: number;
}) {
  const geomRef = useRef<THREE.BufferGeometry>(null);
  const matRef = useRef<THREE.PointsMaterial>(null);
  const numPoints = positions.length / 3;

  // Update material size when pointSize changes
  useEffect(() => {
    if (matRef.current) {
      matRef.current.size = pointSize;
      matRef.current.needsUpdate = true;
    }
  }, [pointSize]);

  // Update vertex colors when colors change
  useEffect(() => {
    if (geomRef.current && colors) {
      const colorAttr = geomRef.current.getAttribute('color');
      if (colorAttr) {
        (colorAttr as THREE.BufferAttribute).array = colors;
        (colorAttr as THREE.BufferAttribute).needsUpdate = true;
      }
    }
  }, [colors]);

  return (
    <points>
      <bufferGeometry ref={geomRef}>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
          count={numPoints}
          itemSize={3}
        />
        {colors && (
          <bufferAttribute
            attach="attributes-color"
            args={[colors, 3]}
            count={numPoints}
            itemSize={3}
          />
        )}
      </bufferGeometry>
      <pointsMaterial
        ref={matRef}
        size={pointSize}
        sizeAttenuation
        vertexColors={!!colors}
        color={colors ? undefined : '#aaaaaa'}
      />
    </points>
  );
}

/**
 * Parse a .npy ArrayBuffer into a Float32Array of positions.
 * NPY format: magic (6 bytes) + version (2) + header_len (2-4) + header + data
 */
function parseNpyPositions(buffer: ArrayBuffer): Float32Array {
  const view = new DataView(buffer);
  // Read header length (version 1.0: 2 bytes at offset 8)
  const headerLen = view.getUint16(8, true);
  const dataOffset = 10 + headerLen;

  // Parse header to determine dtype
  const headerBytes = new Uint8Array(buffer, 10, headerLen);
  const header = new TextDecoder().decode(headerBytes);

  // Default: assume float32
  if (header.includes('float64') || header.includes('<f8')) {
    const float64 = new Float64Array(buffer, dataOffset);
    return new Float32Array(float64);
  }

  return new Float32Array(buffer, dataOffset);
}

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
