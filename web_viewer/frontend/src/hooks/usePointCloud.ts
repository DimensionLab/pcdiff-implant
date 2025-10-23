import { useState, useEffect } from 'react';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import * as THREE from 'three';
import { apiService } from '../services/api';

export const usePointCloud = (resultId: string | null, fileType: 'input' | 'sample', sampleIndex = 0) => {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!resultId) {
      setGeometry(null);
      return;
    }

    const loadPointCloud = async () => {
      try {
        setLoading(true);
        setError(null);

        // Determine filename
        let filename: string;
        if (fileType === 'input') {
          filename = 'input.ply';
        } else {
          filename = sampleIndex > 0 ? `sample_${sampleIndex}.ply` : 'sample.ply';
        }

        const url = apiService.getFileUrl(resultId, filename);

        // Load PLY file
        const loader = new PLYLoader();
        const loadedGeometry = await new Promise<THREE.BufferGeometry>((resolve, reject) => {
          loader.load(
            url,
            (geometry) => resolve(geometry),
            undefined,
            (error) => reject(error)
          );
        });

        // Center the geometry
        loadedGeometry.computeBoundingBox();
        if (loadedGeometry.boundingBox) {
          const center = new THREE.Vector3();
          loadedGeometry.boundingBox.getCenter(center);
          loadedGeometry.translate(-center.x, -center.y, -center.z);
        }

        setGeometry(loadedGeometry);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load point cloud');
        setGeometry(null);
      } finally {
        setLoading(false);
      }
    };

    loadPointCloud();

    // Cleanup
    return () => {
      if (geometry) {
        geometry.dispose();
      }
    };
  }, [resultId, fileType, sampleIndex]);

  return { geometry, loading, error };
};

