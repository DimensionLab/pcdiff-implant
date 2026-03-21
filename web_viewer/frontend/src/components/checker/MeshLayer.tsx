/**
 * Renders an STL mesh as solid geometry inside the R3F Canvas.
 *
 * Uses Three.js STLLoader to parse the binary STL buffer into
 * a BufferGeometry, then renders it with MeshStandardMaterial.
 */
import { useMemo } from 'react';
import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';

interface MeshLayerProps {
  stlBuffer: ArrayBuffer;
  offset: number[];
  color: string;
  opacity: number;
}

export function MeshLayer({ stlBuffer, offset, color, opacity }: MeshLayerProps) {
  const geometry = useMemo(() => {
    const loader = new STLLoader();
    const geom = loader.parse(stlBuffer);

    // Apply shared centering offset (same as PointCloudLayer)
    const positions = geom.attributes.position;
    const [ox, oy, oz] = offset;
    for (let i = 0; i < positions.count; i++) {
      positions.setXYZ(
        i,
        positions.getX(i) - ox,
        positions.getY(i) - oy,
        positions.getZ(i) - oz,
      );
    }
    positions.needsUpdate = true;
    geom.computeVertexNormals();

    return geom;
  }, [stlBuffer, offset]);

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial
        color={color}
        transparent={opacity < 1}
        opacity={opacity}
        side={THREE.DoubleSide}
        metalness={0.1}
        roughness={0.6}
      />
    </mesh>
  );
}
