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
// @ts-ignore
import vtkLight from '@kitware/vtk.js/Rendering/Core/Light';
// @ts-ignore
import vtkWindowedSincPolyDataFilter from '@kitware/vtk.js/Filters/General/WindowedSincPolyDataFilter';
// @ts-ignore
import vtkPolyDataNormals from '@kitware/vtk.js/Filters/Core/PolyDataNormals';

export interface VtkViewportHandle {
  resetCamera: () => void;
  setCameraPreset: (preset: CameraPreset) => void;
  setClippingPlane: (axis: 'x' | 'y' | 'z', position: number) => void;
  clearClippingPlanes: () => void;
  setBaseOpacity: (opacity: number) => void;
  setBaseWindowLevel: (window: number, level: number) => void;
  setBaseVisible: (visible: boolean) => void;
  setOverlayVisible: (visible: boolean) => void;
  setOverlayColor: (r: number, g: number, b: number) => void;
  setOverlayOpacity: (opacity: number) => void;
  setOverlayEdgeVisibility: (visible: boolean) => void;
  setOverlaySmoothing: (iterations: number) => void;
  setLightIntensity: (intensity: number) => void;
  setLightElevation: (elevation: number) => void;
  setTwoSidedLighting: (enabled: boolean) => void;
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
  // Mesh smoothing pipeline: keep reader output so we can re-smooth
  const meshReaderRef = useRef<any>(null);
  const meshSmoothFilterRef = useRef<any>(null);
  const meshNormalsFilterRef = useRef<any>(null);
  // Lighting
  const keyLightRef = useRef<any>(null);
  const baseImageRef = useRef<any>(null);
  const [loading, setLoading] = useState(false);

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

    setOverlayColor: (r: number, g: number, b: number) => {
      if (!overlayActorRef.current || !renderWindowRef.current) return;
      // For mesh/geometry actors, setColor works directly
      if (overlayActorRef.current.getProperty().setColor) {
        overlayActorRef.current.getProperty().setColor(r, g, b);
      }
      // For volume actors, update the RGB transfer function
      if (overlayActorRef.current.getProperty().setRGBTransferFunction) {
        const ctfun = vtkColorTransferFunction.newInstance();
        ctfun.addRGBPoint(0, r, g, b);
        ctfun.addRGBPoint(255, r, g, b);
        overlayActorRef.current.getProperty().setRGBTransferFunction(0, ctfun);
      }
      renderWindowRef.current.render();
    },

    setOverlayOpacity: (opacity: number) => {
      if (!overlayActorRef.current || !renderWindowRef.current) return;
      // For mesh/geometry actors
      if (overlayActorRef.current.getProperty().setOpacity) {
        overlayActorRef.current.getProperty().setOpacity(opacity);
      }
      // For volume actors, update the scalar opacity function
      if (overlayActorRef.current.getProperty().setScalarOpacity) {
        const ofun = vtkPiecewiseFunction.newInstance();
        ofun.addPoint(0, 0.0);
        ofun.addPoint(127, 0.0);
        ofun.addPoint(128, opacity);
        ofun.addPoint(255, opacity);
        overlayActorRef.current.getProperty().setScalarOpacity(0, ofun);
      }
      renderWindowRef.current.render();
    },

    setOverlayEdgeVisibility: (visible: boolean) => {
      if (!overlayActorRef.current || !renderWindowRef.current) return;
      const prop = overlayActorRef.current.getProperty();
      if (prop.setEdgeVisibility) {
        prop.setEdgeVisibility(visible);
        if (visible) {
          prop.setEdgeColor(0.15, 0.15, 0.15);
        }
      }
      renderWindowRef.current.render();
    },

    setOverlaySmoothing: (iterations: number) => {
      if (!meshSmoothFilterRef.current || !overlayMapperRef.current || !renderWindowRef.current) return;
      meshSmoothFilterRef.current.setNumberOfIterations(iterations);
      if (meshNormalsFilterRef.current) {
        meshNormalsFilterRef.current.update();
      }
      renderWindowRef.current.render();
    },

    setLightIntensity: (intensity: number) => {
      if (!keyLightRef.current || !renderWindowRef.current) return;
      keyLightRef.current.setIntensity(intensity);
      renderWindowRef.current.render();
    },

    setLightElevation: (elevation: number) => {
      if (!keyLightRef.current || !renderWindowRef.current) return;
      keyLightRef.current.setDirectionAngle(elevation, 45);
      renderWindowRef.current.render();
    },

    setTwoSidedLighting: (enabled: boolean) => {
      if (!rendererRef.current || !renderWindowRef.current) return;
      rendererRef.current.setTwoSidedLighting(enabled);
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
    meshReaderRef.current = null;
    meshSmoothFilterRef.current = null;
    meshNormalsFilterRef.current = null;
    keyLightRef.current = null;

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

    // Configurable scene lighting
    renderer.removeAllLights();
    renderer.setTwoSidedLighting(true);
    const keyLight = vtkLight.newInstance();
    keyLight.setLightTypeToSceneLight();
    keyLight.setDirectionAngle(45, 45);
    keyLight.setIntensity(1.0);
    keyLight.setColor(1, 1, 1);
    renderer.addLight(keyLight);
    keyLightRef.current = keyLight;
    // Soft fill light from opposite side
    const fillLight = vtkLight.newInstance();
    fillLight.setLightTypeToSceneLight();
    fillLight.setDirectionAngle(-30, -135);
    fillLight.setIntensity(0.4);
    fillLight.setColor(0.9, 0.9, 1.0);
    renderer.addLight(fillLight);

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
          meshReaderRef.current = reader;

          // Smoothing filter: Windowed Sinc (0 iterations = no smoothing by default)
          const smoother = vtkWindowedSincPolyDataFilter.newInstance({
            numberOfIterations: 0,
            passBand: 0.1,
            boundarySmoothing: false,
            nonManifoldSmoothing: false,
          });
          smoother.setInputConnection(reader.getOutputPort());
          meshSmoothFilterRef.current = smoother;

          // Recompute normals after smoothing for correct shading
          const normals = vtkPolyDataNormals.newInstance({
            computePointNormals: true,
            computeCellNormals: false,
            splitting: false,
          });
          normals.setInputConnection(smoother.getOutputPort());
          meshNormalsFilterRef.current = normals;

          const meshMapper = vtkMapper.newInstance();
          meshMapper.setInputConnection(normals.getOutputPort());
          meshMapper.setScalarVisibility(false);
          overlayMapperRef.current = meshMapper;

          const meshActor = vtkActor.newInstance();
          meshActor.setMapper(meshMapper);
          meshActor.getProperty().setColor(r, g, b);
          meshActor.getProperty().setOpacity(overlayOpacity);
          meshActor.getProperty().setAmbient(0.5);
          meshActor.getProperty().setDiffuse(0.6);
          meshActor.getProperty().setSpecular(0.2);

          // Build a 4x4 affine matrix to map from 256^3 mesh space to skull world
          // space. marching_cubes produces vertices in numpy index order (axis0,
          // axis1, axis2). vtk.js vtkImageData maps X=numpy-axis2, Y=numpy-axis1,
          // Z=numpy-axis0 (C-order bytes with Fortran-order indexing). So we need:
          //   vtk_x = mesh_v2 * scale2 + origin_x
          //   vtk_y = mesh_v1 * scale1 + origin_y
          //   vtk_z = mesh_v0 * scale0 + origin_z
          // This is a permutation (swap axis 0 ↔ axis 2) plus scale + translate.
          const implantRes = 256;
          const { dims, spacing, origin: orig } = base.metadata;
          // dims[0]=numpy-axis0, dims[1]=numpy-axis1, dims[2]=numpy-axis2
          // vtk X extent = dims[2]*spacing[2], vtk Y = dims[1]*spacing[1], vtk Z = dims[0]*spacing[0]
          const scaleX = (dims[2] * spacing[2]) / implantRes;  // mesh axis2 → vtk X
          const scaleY = (dims[1] * spacing[1]) / implantRes;  // mesh axis1 → vtk Y
          const scaleZ = (dims[0] * spacing[0]) / implantRes;  // mesh axis0 → vtk Z
          // Row-major 4x4: mesh (v0, v1, v2) → vtk (v2*scaleX + ox, v1*scaleY + oy, v0*scaleZ + oz)
          // Column vector convention: [v0, v1, v2, 1]^T
          // Row 0 (vtk X): takes mesh v2 → col 2
          // Row 1 (vtk Y): takes mesh v1 → col 1
          // Row 2 (vtk Z): takes mesh v0 → col 0
          // prettier-ignore
          const userMatrix = new Float64Array([
            0,       0,       scaleX,  orig[0],
            0,       scaleY,  0,       orig[1],
            scaleZ,  0,       0,       orig[2],
            0,       0,       0,       1,
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
  // Color and opacity are handled imperatively via the ref handle, so they
  // are NOT in the dependency array — changing them won't reload the scene.
  }, [scanId, overlayScanId, overlayRenderMode, overlayMeshUrl]);

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
