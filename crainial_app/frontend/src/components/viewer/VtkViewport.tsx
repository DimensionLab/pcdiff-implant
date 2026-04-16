/**
 * vtk.js volume rendering viewport with optional implant overlay.
 *
 * Renders one base scan (e.g. defective skull NRRD) and an optional
 * second volume (e.g. cran-2 implant mask NRRD) blended on top with
 * a different color/opacity ramp.
 *
 * vtk.js is mounted imperatively in a plain <div> — it can't share a
 * canvas with Three.js / R3F.
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
  /** Optional second scan rendered on top (e.g. cran-2 implant mask). */
  overlayScanId?: string | null;
  /** RGB tint applied to the overlay (defaults to bright red). */
  overlayColor?: [number, number, number];
  /** Peak opacity of the overlay (0..1). */
  overlayOpacity?: number;
  onReady?: () => void;
  onError?: (err: string) => void;
}

function buildBoneTransferFunctions() {
  const ctfun = vtkColorTransferFunction.newInstance();
  ctfun.addRGBPoint(0, 0, 0, 0);
  ctfun.addRGBPoint(80, 0.55, 0.25, 0.15);
  ctfun.addRGBPoint(160, 0.88, 0.81, 0.76);
  ctfun.addRGBPoint(255, 1.0, 1.0, 0.95);

  const ofun = vtkPiecewiseFunction.newInstance();
  ofun.addPoint(0, 0.0);
  ofun.addPoint(40, 0.0);
  ofun.addPoint(80, 0.02);
  ofun.addPoint(120, 0.15);
  ofun.addPoint(200, 0.6);
  ofun.addPoint(255, 0.8);

  return { ctfun, ofun };
}

function buildOverlayTransferFunctions(
  color: [number, number, number],
  peakOpacity: number,
  scalarRange: [number, number],
) {
  const [r, g, b] = color;
  const [lo, hi] = scalarRange;
  // For binary/probability masks the meaningful values cluster near hi;
  // tint anything above ~halfway with the chosen color.
  const mid = lo + (hi - lo) * 0.5;
  const ctfun = vtkColorTransferFunction.newInstance();
  ctfun.addRGBPoint(lo, r, g, b);
  ctfun.addRGBPoint(hi, r, g, b);

  const ofun = vtkPiecewiseFunction.newInstance();
  ofun.addPoint(lo, 0.0);
  ofun.addPoint(mid, 0.0);
  ofun.addPoint(mid + 1, peakOpacity);
  ofun.addPoint(hi, peakOpacity);

  return { ctfun, ofun };
}

function makeImageData(
  data: Uint8Array,
  metadata: { dims: number[]; spacing: number[]; origin: number[] },
) {
  const imageData = vtkImageData.newInstance();
  imageData.setDimensions(metadata.dims as [number, number, number]);
  imageData.setSpacing(metadata.spacing as [number, number, number]);
  imageData.setOrigin(metadata.origin as [number, number, number]);
  const scalars = vtkDataArray.newInstance({
    name: 'Scalars',
    numberOfComponents: 1,
    values: data,
  });
  imageData.getPointData().setScalars(scalars);
  return imageData;
}

export function VtkViewport({
  scanId,
  overlayScanId = null,
  overlayColor = [1.0, 0.2, 0.25],
  overlayOpacity = 0.7,
  onReady,
  onError,
}: VtkViewportProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const contextRef = useRef<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!containerRef.current || !scanId) return;

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

    const baseLoad = scanApi.loadVolumeData(scanId);
    const overlayLoad = overlayScanId
      ? scanApi.loadVolumeData(overlayScanId)
      : Promise.resolve(null);

    Promise.all([baseLoad, overlayLoad])
      .then(([base, overlay]) => {
        if (cancelled) return;

        const { ctfun: baseCtf, ofun: baseOfun } = buildBoneTransferFunctions();
        const baseImage = makeImageData(base.data, base.metadata);

        const baseMapper = vtkVolumeMapper.newInstance();
        baseMapper.setInputData(baseImage);
        baseMapper.setSampleDistance(1.0);

        const baseActor = vtkVolume.newInstance();
        baseActor.setMapper(baseMapper);
        baseActor.getProperty().setRGBTransferFunction(0, baseCtf);
        baseActor.getProperty().setScalarOpacity(0, baseOfun);
        baseActor.getProperty().setInterpolationTypeToLinear();
        baseActor.getProperty().setShade(true);
        baseActor.getProperty().setAmbient(0.2);
        baseActor.getProperty().setDiffuse(0.7);
        baseActor.getProperty().setSpecular(0.3);
        renderer.addVolume(baseActor);

        if (overlay) {
          const { ctfun: ovCtf, ofun: ovOfun } = buildOverlayTransferFunctions(
            overlayColor,
            overlayOpacity,
            overlay.metadata.scalar_range,
          );
          const overlayImage = makeImageData(overlay.data, overlay.metadata);

          const overlayMapper = vtkVolumeMapper.newInstance();
          overlayMapper.setInputData(overlayImage);
          overlayMapper.setSampleDistance(1.0);

          const overlayActor = vtkVolume.newInstance();
          overlayActor.setMapper(overlayMapper);
          overlayActor.getProperty().setRGBTransferFunction(0, ovCtf);
          overlayActor.getProperty().setScalarOpacity(0, ovOfun);
          overlayActor.getProperty().setInterpolationTypeToLinear();
          overlayActor.getProperty().setShade(false);
          renderer.addVolume(overlayActor);
        }

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
  }, [scanId, overlayScanId, overlayColor, overlayOpacity]);

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
