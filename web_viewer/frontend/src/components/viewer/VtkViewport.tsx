/**
 * vtk.js volume rendering viewport.
 *
 * Creates a WebGL context in a plain <div> ref and mounts the vtk.js
 * rendering pipeline imperatively. vtk.js cannot share a canvas with
 * Three.js / React Three Fiber.
 *
 * Volume data is loaded from the backend as raw uint8 binary (the backend
 * reads NRRD with pynrrd) since vtk.js v30 does not ship an NRRDReader.
 */
import { useRef, useEffect, useState } from 'react';
import { scanApi } from '../../services/scan-api';

// @ts-ignore - vtk.js types are incomplete
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
// @ts-ignore
import vtkVolume from '@kitware/vtk.js/Rendering/Core/Volume';
// @ts-ignore
import vtkVolumeMapper from '@kitware/vtk.js/Rendering/Core/VolumeMapper';
// @ts-ignore
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
// @ts-ignore
import vtkPiecewiseFunction from '@kitware/vtk.js/Common/DataModel/PiecewiseFunction';
// @ts-ignore
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
// @ts-ignore
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';

interface VtkViewportProps {
  scanId: string | null;
  onReady?: () => void;
  onError?: (err: string) => void;
}

export function VtkViewport({ scanId, onReady, onError }: VtkViewportProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const contextRef = useRef<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!containerRef.current || !scanId) return;

    // Cleanup previous context
    if (contextRef.current) {
      contextRef.current.delete();
      contextRef.current = null;
    }

    setLoading(true);
    let cancelled = false;

    const genericRenderer = vtkGenericRenderWindow.newInstance();
    genericRenderer.setContainer(containerRef.current);
    genericRenderer.resize();

    const renderer = genericRenderer.getRenderer();
    const renderWindow = genericRenderer.getRenderWindow();

    // Color transfer function (bone CT preset)
    const ctfun = vtkColorTransferFunction.newInstance();
    ctfun.addRGBPoint(0, 0, 0, 0);           // air = black
    ctfun.addRGBPoint(80, 0.55, 0.25, 0.15); // soft tissue
    ctfun.addRGBPoint(160, 0.88, 0.81, 0.76); // bone
    ctfun.addRGBPoint(255, 1.0, 1.0, 0.95);   // dense bone

    // Opacity transfer function
    const ofun = vtkPiecewiseFunction.newInstance();
    ofun.addPoint(0, 0.0);      // air transparent
    ofun.addPoint(40, 0.0);     // near-air transparent
    ofun.addPoint(80, 0.02);    // soft tissue slight
    ofun.addPoint(120, 0.15);   // bone starts
    ofun.addPoint(200, 0.6);    // dense bone
    ofun.addPoint(255, 0.8);    // max

    // Load volume data from backend (parsed NRRD as raw uint8)
    scanApi
      .loadVolumeData(scanId)
      .then(({ data, metadata }) => {
        if (cancelled) return;

        const { dims, spacing, origin } = metadata;

        // Create vtkImageData from raw binary
        const imageData = vtkImageData.newInstance();
        imageData.setDimensions(dims[0], dims[1], dims[2]);
        imageData.setSpacing(spacing[0], spacing[1], spacing[2]);
        imageData.setOrigin(origin[0], origin[1], origin[2]);

        // Create scalar data array
        const scalars = vtkDataArray.newInstance({
          name: 'Scalars',
          numberOfComponents: 1,
          values: data,
        });
        imageData.getPointData().setScalars(scalars);

        const mapper = vtkVolumeMapper.newInstance();
        mapper.setInputData(imageData);
        mapper.setSampleDistance(1.0);

        const actor = vtkVolume.newInstance();
        actor.setMapper(mapper);
        actor.getProperty().setRGBTransferFunction(0, ctfun);
        actor.getProperty().setScalarOpacity(0, ofun);
        actor.getProperty().setInterpolationTypeToLinear();
        actor.getProperty().setShade(true);
        actor.getProperty().setAmbient(0.2);
        actor.getProperty().setDiffuse(0.7);
        actor.getProperty().setSpecular(0.3);

        renderer.addVolume(actor);
        renderer.resetCamera();
        renderer.getActiveCamera().elevation(30);
        renderWindow.render();

        setLoading(false);
        onReady?.();
      })
      .catch((err: Error) => {
        if (cancelled) return;
        console.error('Failed to load volume:', err);
        onError?.(`Failed to load volume: ${err.message}`);
        setLoading(false);
      });

    contextRef.current = genericRenderer;

    // Handle container resize
    const resizeObserver = new ResizeObserver(() => {
      if (contextRef.current) {
        contextRef.current.resize();
      }
    });
    resizeObserver.observe(containerRef.current);

    return () => {
      cancelled = true;
      resizeObserver.disconnect();
      if (contextRef.current) {
        contextRef.current.delete();
        contextRef.current = null;
      }
    };
  }, [scanId]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div
        ref={containerRef}
        style={{ width: '100%', height: '100%', background: '#0a0a1a' }}
      />
      {loading && (
        <div style={styles.loadingOverlay}>
          <div style={styles.loadingText}>Loading volume...</div>
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  loadingOverlay: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'rgba(0,0,0,0.5)',
    zIndex: 10,
  },
  loadingText: {
    color: '#fff',
    fontSize: '14px',
  },
};
