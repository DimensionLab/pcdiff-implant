/**
 * vtk.js volume rendering viewport with optional implant overlay.
 *
 * Renders one base scan (e.g. defective skull NRRD) and an optional
 * second volume (e.g. cran-2 implant mask NRRD) blended on top with
 * a different color/opacity ramp.
 *
 * The overlay is automatically resampled server-side to match the base
 * scan's grid (via the align_to query parameter) so both volumes occupy
 * the same world-space bounding box.
 */
import { useRef, useEffect, useState, useImperativeHandle, forwardRef } from 'react';
import { scanApi } from '../../services/scan-api';

import '@kitware/vtk.js/Rendering/Profiles/Volume';
import '@kitware/vtk.js/Rendering/Profiles/Geometry';

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
// @ts-ignore
import vtkPlane from '@kitware/vtk.js/Common/DataModel/Plane';
// @ts-ignore
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
// @ts-ignore
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
// @ts-ignore
import vtkSTLReader from '@kitware/vtk.js/IO/Geometry/STLReader';
// @ts-ignore
import vtkPLYReader from '@kitware/vtk.js/IO/Geometry/PLYReader';
// @ts-ignore
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';

export interface VtkViewportHandle {
  resetCamera: () => void;
  setCameraPreset: (preset: CameraPreset) => void;
  setClippingPlane: (axis: 'x' | 'y' | 'z', position: number) => void;
  clearClippingPlanes: () => void;
  setBaseOpacity: (opacity: number) => void;
  setBaseWindowLevel: (window: number, level: number) => void;
  setBaseVisible: (visible: boolean) => void;
  setOverlayVisible: (visible: boolean) => void;
  render: () => void;
}

export type CameraPreset = 'anterior' | 'posterior' | 'left' | 'right' | 'superior' | 'inferior';
export type OverlayRenderMode = 'volume' | 'mesh' | 'pointcloud';

interface VtkViewportProps {
  scanId: string | null;
  overlayScanId?: string | null;
  overlayColor?: [number, number, number];
  overlayOpacity?: number;
  /** How to render the overlay: 'volume' (NRRD), 'mesh' (STL/PLY), or 'pointcloud'. */
  overlayRenderMode?: OverlayRenderMode;
  /** URL to an STL or PLY file for mesh mode. */
  overlayMeshUrl?: string | null;
  onReady?: () => void;
  onError?: (err: string) => void;
}

function buildBoneTransferFunctions(windowWidth = 175, windowLevel = 168) {
  const lo = windowLevel - windowWidth / 2;
  const hi = windowLevel + windowWidth / 2;

  const ctfun = vtkColorTransferFunction.newInstance();
  ctfun.addRGBPoint(0, 0, 0, 0);
  ctfun.addRGBPoint(Math.max(lo, 1), 0, 0, 0);
  ctfun.addRGBPoint(lo + (hi - lo) * 0.2, 0.55, 0.25, 0.15);
  ctfun.addRGBPoint(lo + (hi - lo) * 0.6, 0.88, 0.81, 0.76);
  ctfun.addRGBPoint(hi, 1.0, 1.0, 0.95);
  ctfun.addRGBPoint(255, 1.0, 1.0, 0.95);

  // Higher per-sample opacity so bone accumulates to near-opaque and occludes
  // objects behind skull walls during ray marching.
  const ofun = vtkPiecewiseFunction.newInstance();
  ofun.addPoint(0, 0.0);
  ofun.addPoint(Math.max(lo - 10, 1), 0.0);
  ofun.addPoint(lo, 0.05);
  ofun.addPoint(lo + (hi - lo) * 0.2, 0.35);
  ofun.addPoint(lo + (hi - lo) * 0.5, 0.85);
  ofun.addPoint(hi, 1.0);
  ofun.addPoint(255, 1.0);

  return { ctfun, ofun };
}

function buildOverlayTransferFunctions(
  color: [number, number, number],
  peakOpacity: number,
  scalarRange: [number, number],
) {
  const [r, g, b] = color;
  const [lo, hi] = scalarRange;
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

const DEFAULT_OVERLAY_COLOR: [number, number, number] = [1.0, 0.2, 0.25];

const CAMERA_PRESETS: Record<CameraPreset, { position: [number, number, number]; viewUp: [number, number, number] }> = {
  anterior:  { position: [0, -1,  0], viewUp: [0, 0, 1] },
  posterior: { position: [0,  1,  0], viewUp: [0, 0, 1] },
  left:      { position: [-1, 0,  0], viewUp: [0, 0, 1] },
  right:     { position: [1,  0,  0], viewUp: [0, 0, 1] },
  superior:  { position: [0,  0,  1], viewUp: [0, 1, 0] },
  inferior:  { position: [0,  0, -1], viewUp: [0, 1, 0] },
};

export const VtkViewport = forwardRef<VtkViewportHandle, VtkViewportProps>(function VtkViewport(
  {
    scanId,
    overlayScanId = null,
    overlayColor = DEFAULT_OVERLAY_COLOR,
    overlayOpacity = 0.7,
    overlayRenderMode = 'volume',
    overlayMeshUrl = null,
    onReady,
    onError,
  },
  ref,
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const contextRef = useRef<any>(null);
  const baseActorRef = useRef<any>(null);
  const overlayActorRef = useRef<any>(null);
  const baseMapperRef = useRef<any>(null);
  const overlayMapperRef = useRef<any>(null);
  const rendererRef = useRef<any>(null);
  const renderWindowRef = useRef<any>(null);
  const baseImageRef = useRef<any>(null);
  const [loading, setLoading] = useState(false);
  const colorKey = overlayColor.join(',');

  useImperativeHandle(ref, () => ({
    resetCamera: () => {
      if (!rendererRef.current || !renderWindowRef.current) return;
      rendererRef.current.resetCamera();
      renderWindowRef.current.render();
    },

    setCameraPreset: (preset: CameraPreset) => {
      if (!rendererRef.current || !renderWindowRef.current || !baseImageRef.current) return;
      const camera = rendererRef.current.getActiveCamera();
      const bounds = baseImageRef.current.getBounds();
      const center = [
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2,
      ];
      const size = Math.max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]);
      const dist = size * 2;
      const p = CAMERA_PRESETS[preset];
      camera.setPosition(
        center[0] + p.position[0] * dist,
        center[1] + p.position[1] * dist,
        center[2] + p.position[2] * dist,
      );
      camera.setFocalPoint(...center);
      camera.setViewUp(...p.viewUp);
      rendererRef.current.resetCameraClippingRange();
      renderWindowRef.current.render();
    },

    setClippingPlane: (axis: 'x' | 'y' | 'z', position: number) => {
      if (!baseMapperRef.current || !baseImageRef.current || !renderWindowRef.current) return;
      const bounds = baseImageRef.current.getBounds();
      const normals: Record<string, [number, number, number]> = {
        x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1],
      };
      const origins: Record<string, [number, number, number]> = {
        x: [bounds[0] + (bounds[1] - bounds[0]) * position, 0, 0],
        y: [0, bounds[2] + (bounds[3] - bounds[2]) * position, 0],
        z: [0, 0, bounds[4] + (bounds[5] - bounds[4]) * position],
      };

      const plane = vtkPlane.newInstance();
      plane.setNormal(...normals[axis]);
      plane.setOrigin(...origins[axis]);

      for (const mapper of [baseMapperRef.current, overlayMapperRef.current]) {
        if (!mapper) continue;
        mapper.removeAllClippingPlanes();
        mapper.addClippingPlane(plane);
      }
      renderWindowRef.current.render();
    },

    clearClippingPlanes: () => {
      for (const mapper of [baseMapperRef.current, overlayMapperRef.current]) {
        if (!mapper) continue;
        mapper.removeAllClippingPlanes();
      }
      renderWindowRef.current?.render();
    },

    setBaseOpacity: (opacity: number) => {
      if (!baseActorRef.current || !renderWindowRef.current) return;
      const ofun = vtkPiecewiseFunction.newInstance();
      ofun.addPoint(0, 0.0);
      ofun.addPoint(40, 0.0);
      ofun.addPoint(80, 0.05 * opacity);
      ofun.addPoint(120, 0.35 * opacity);
      ofun.addPoint(180, 0.85 * opacity);
      ofun.addPoint(255, 1.0 * opacity);
      baseActorRef.current.getProperty().setScalarOpacity(0, ofun);
      renderWindowRef.current.render();
    },

    setBaseWindowLevel: (windowWidth: number, level: number) => {
      if (!baseActorRef.current || !renderWindowRef.current) return;
      const { ctfun, ofun } = buildBoneTransferFunctions(windowWidth, level);
      baseActorRef.current.getProperty().setRGBTransferFunction(0, ctfun);
      baseActorRef.current.getProperty().setScalarOpacity(0, ofun);
      renderWindowRef.current.render();
    },

    setBaseVisible: (visible: boolean) => {
      if (!baseActorRef.current || !renderWindowRef.current) return;
      baseActorRef.current.setVisibility(visible);
      renderWindowRef.current.render();
    },

    setOverlayVisible: (visible: boolean) => {
      if (!overlayActorRef.current || !renderWindowRef.current) return;
      overlayActorRef.current.setVisibility(visible);
      renderWindowRef.current.render();
    },

    render: () => {
      renderWindowRef.current?.render();
    },
  }));

  useEffect(() => {
    if (!containerRef.current || !scanId) return;

    if (contextRef.current) {
      contextRef.current.delete();
      contextRef.current = null;
    }
    baseActorRef.current = null;
    overlayActorRef.current = null;
    baseMapperRef.current = null;
    overlayMapperRef.current = null;
    baseImageRef.current = null;

    setLoading(true);
    let cancelled = false;

    const genericRenderer = vtkGenericRenderWindow.newInstance();
    genericRenderer.setContainer(containerRef.current);
    genericRenderer.resize();

    const renderer = genericRenderer.getRenderer();
    const renderWindow = genericRenderer.getRenderWindow();
    rendererRef.current = renderer;
    renderWindowRef.current = renderWindow;

    // Enable depth peeling for correct volume compositing with overlapping volumes
    renderer.setUseDepthPeeling(true);
    renderer.setMaximumNumberOfPeels(6);
    renderer.setOcclusionRatio(0.1);

    const baseLoad = scanApi.loadVolumeData(scanId);

    // Overlay loading depends on render mode
    const overlayLoadPromise: Promise<any> = (() => {
      if (!overlayScanId && !overlayMeshUrl) return Promise.resolve(null);
      if (overlayRenderMode === 'mesh' && overlayMeshUrl) {
        return fetch(overlayMeshUrl).then((r) => {
          if (!r.ok) throw new Error(`Failed to load mesh: ${r.statusText}`);
          return r.arrayBuffer();
        });
      }
      if (overlayScanId) {
        return scanApi.loadVolumeData(overlayScanId, scanId);
      }
      return Promise.resolve(null);
    })();

    Promise.all([baseLoad, overlayLoadPromise])
      .then(([base, overlayData]) => {
        if (cancelled) return;

        const { ctfun: baseCtf, ofun: baseOfun } = buildBoneTransferFunctions();
        const baseImage = makeImageData(base.data, base.metadata);
        baseImageRef.current = baseImage;

        const baseMapper = vtkVolumeMapper.newInstance();
        baseMapper.setInputData(baseImage);
        baseMapper.setSampleDistance(0.5);
        baseMapperRef.current = baseMapper;

        const baseActor = vtkVolume.newInstance();
        baseActor.setMapper(baseMapper);
        baseActor.getProperty().setRGBTransferFunction(0, baseCtf);
        baseActor.getProperty().setScalarOpacity(0, baseOfun);
        baseActor.getProperty().setInterpolationTypeToLinear();
        baseActor.getProperty().setShade(true);
        baseActor.getProperty().setAmbient(0.2);
        baseActor.getProperty().setDiffuse(0.7);
        baseActor.getProperty().setSpecular(0.3);
        baseActorRef.current = baseActor;
        renderer.addVolume(baseActor);

        if (overlayData && overlayRenderMode === 'mesh') {
          // STL/PLY mesh overlay — mesh is in 256^3 index space, transform
          // to match the skull volume's world coordinates.
          const [r, g, b] = overlayColor;
          const url = overlayMeshUrl || '';
          const isStl = url.toLowerCase().includes('.stl') || url.toLowerCase().includes('/stl');
          const reader = isStl ? vtkSTLReader.newInstance() : vtkPLYReader.newInstance();
          reader.parseAsArrayBuffer(overlayData);

          const meshMapper = vtkMapper.newInstance();
          meshMapper.setInputConnection(reader.getOutputPort());
          overlayMapperRef.current = meshMapper;

          const meshActor = vtkActor.newInstance();
          meshActor.setMapper(meshMapper);
          meshActor.getProperty().setColor(r, g, b);
          meshActor.getProperty().setOpacity(overlayOpacity);
          meshActor.getProperty().setLighting(true);

          // Build a 4x4 affine matrix to map from 256^3 mesh space to skull world space.
          // The mesh vertices are in [0, ~256] (marching_cubes with spacing=1).
          // Skull world space: origin + index * spacing, where index ranges [0, dims-1].
          const implantRes = 256;
          const { dims, spacing, origin: orig } = base.metadata;
          const sx = (dims[0] * spacing[0]) / implantRes;
          const sy = (dims[1] * spacing[1]) / implantRes;
          const sz = (dims[2] * spacing[2]) / implantRes;
          // vtk.js uses row-major 4x4 matrix
          // prettier-ignore
          const userMatrix = new Float64Array([
            sx, 0,  0,  orig[0],
            0,  sy, 0,  orig[1],
            0,  0,  sz, orig[2],
            0,  0,  0,  1,
          ]);
          meshActor.setUserMatrix(userMatrix as any);

          overlayActorRef.current = meshActor;
          renderer.addActor(meshActor);
        } else if (overlayData && overlayRenderMode === 'volume') {
          // Volume overlay (NRRD)
          const { ctfun: ovCtf, ofun: ovOfun } = buildOverlayTransferFunctions(
            overlayColor,
            overlayOpacity,
            overlayData.metadata.scalar_range,
          );
          const overlayImage = makeImageData(overlayData.data, overlayData.metadata);

          const overlayMapper = vtkVolumeMapper.newInstance();
          overlayMapper.setInputData(overlayImage);
          overlayMapper.setSampleDistance(1.0);
          overlayMapperRef.current = overlayMapper;

          const overlayActor = vtkVolume.newInstance();
          overlayActor.setMapper(overlayMapper);
          overlayActor.getProperty().setRGBTransferFunction(0, ovCtf);
          overlayActor.getProperty().setScalarOpacity(0, ovOfun);
          overlayActor.getProperty().setInterpolationTypeToLinear();
          overlayActor.getProperty().setShade(false);
          overlayActorRef.current = overlayActor;
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
  }, [scanId, overlayScanId, colorKey, overlayOpacity, overlayRenderMode, overlayMeshUrl]);

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
});

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
