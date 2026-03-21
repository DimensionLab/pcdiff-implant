import { useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { TrackballControls, Grid } from '@react-three/drei';
import * as THREE from 'three';

interface PointCloudProps {
  geometry: THREE.BufferGeometry | null;
  color: string;
  visible: boolean;
  size?: number;
}

const PointCloud = ({ geometry, color, visible, size = 0.005 }: PointCloudProps) => {
  const pointsRef = useRef<THREE.Points>(null);
  const matRef = useRef<THREE.PointsMaterial>(null);

  useEffect(() => {
    if (pointsRef.current && geometry) {
      pointsRef.current.geometry = geometry;
    }
  }, [geometry]);

  // Update material size when size prop changes
  useEffect(() => {
    if (matRef.current) {
      matRef.current.size = size;
      matRef.current.needsUpdate = true;
    }
  }, [size]);

  if (!geometry) return null;

  return (
    <points ref={pointsRef} visible={visible}>
      <bufferGeometry attach="geometry" {...geometry} />
      <pointsMaterial
        ref={matRef}
        attach="material"
        color={color}
        size={size}
        sizeAttenuation={true}
      />
    </points>
  );
};

interface Viewer3DProps {
  inputGeometry: THREE.BufferGeometry | null;
  sampleGeometry: THREE.BufferGeometry | null;
  showInput: boolean;
  showSample: boolean;
  onCameraReset?: () => void;
}

export const Viewer3D = ({
  inputGeometry,
  sampleGeometry,
  showInput,
  showSample,
}: Viewer3DProps) => {
  return (
    <div style={{ width: '100%', height: '100%' }}>
      <Canvas
        camera={{ position: [1, 1, 1.5], fov: 75 }}
        style={{ background: '#1a1a1a' }}
      >
        {/* Lighting */}
        <ambientLight intensity={0.6} />
        <directionalLight position={[1, 1, 1]} intensity={0.8} />
        <directionalLight position={[-1, -1, -1]} intensity={0.3} />

        {/* Point Clouds */}
        <PointCloud
          geometry={inputGeometry}
          color="#cccccc"
          visible={showInput}
        />
        <PointCloud
          geometry={sampleGeometry}
          color="#ff6464"
          visible={showSample}
        />

        {/* Grid and Axes */}
        <Grid
          args={[10, 10]}
          cellSize={0.2}
          cellThickness={0.5}
          cellColor="#444444"
          sectionSize={1}
          sectionThickness={1}
          sectionColor="#666666"
          fadeDistance={30}
          fadeStrength={1}
          followCamera={false}
          infiniteGrid={true}
        />
        <axesHelper args={[1]} />

        {/* Camera Controls */}
        <TrackballControls
          rotateSpeed={2}
          zoomSpeed={1.2}
          panSpeed={0.8}
          minDistance={0.5}
          maxDistance={10}
          noPan={false}
          noRotate={false}
          noZoom={false}
          staticMoving={false}
          dynamicDampingFactor={0.05}
        />
      </Canvas>
    </div>
  );
};

export default Viewer3D;

