/**
 * Implant Checker page — workspace for verifying implant fit.
 *
 * Doctors can overlay defective skull + implant point clouds, compute
 * fit metrics (Dice, HD, HD95, BDC), view SDF distance heatmaps, and
 * toggle individual layers on/off.
 */
import { useCallback, useMemo, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { AppLayout } from '../components/layout/AppLayout';
import { WorkspaceBrowser } from '../components/checker/WorkspaceBrowser';
import { MultiPointCloudViewer } from '../components/checker/MultiPointCloudViewer';
import { CheckerControls } from '../components/checker/CheckerControls';
import { STLGenerateDialog } from '../components/checker/STLGenerateDialog';
import { useCheckerLayers } from '../hooks/useCheckerLayers';
import { useSDFHeatmap } from '../hooks/useFitMetrics';
import { useColorProfile } from '../hooks/useColorProfiles';
import { useGenerateSTL, useMultiSTLData } from '../hooks/useSTLGeneration';
import { pointCloudApi } from '../services/point-cloud-api';
import { parseColorStops, interpolateColor } from '../types/color-profile';
import type { ColorProfile } from '../types/color-profile';

export function ImplantCheckerPage() {
  const queryClient = useQueryClient();

  // Workspace state
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);

  // Layer state
  const {
    layers,
    addLayer,
    addMeshLayer,
    removeLayer,
    toggleVisibility,
    setLayerColor,
    setLayerHeatmap,
    clearLayers,
    addFromAutoMatch,
  } = useCheckerLayers();

  // Display settings
  const [pointSize, setPointSize] = useState(0.01);
  const [showGrid, setShowGrid] = useState(false);
  const [showAxes, setShowAxes] = useState(false);

  // STL mesh generation
  const generateSTL = useGenerateSTL();
  const [generatingSTLForId, setGeneratingSTLForId] = useState<string | null>(null);
  const [stlDialogTarget, setStlDialogTarget] = useState<{
    pcId: string;
    layerName: string;
  } | null>(null);

  // Fetch STL binary data for all mesh layers
  const meshLayers = layers.filter((l) => l.layerType === 'mesh');
  const meshPcIds = meshLayers.map((l) => l.pointCloudId);
  const meshQueries = useMultiSTLData(meshPcIds);

  const meshData = useMemo(() => {
    const map = new Map<string, ArrayBuffer>();
    for (let i = 0; i < meshLayers.length; i++) {
      const q = meshQueries[i];
      if (q?.data) {
        map.set(meshLayers[i].id, q.data);
      }
    }
    return map;
  }, [meshLayers, meshQueries]);

  // Open the STL settings dialog for a point cloud layer
  const handleOpenSTLDialog = useCallback(
    (pcId: string) => {
      const layer = layers.find((l) => l.pointCloudId === pcId && l.layerType === 'points');
      setStlDialogTarget({
        pcId,
        layerName: layer?.name ?? pcId,
      });
    },
    [layers],
  );

  // Confirm generation from the dialog with user-selected settings
  const handleConfirmGenerate = useCallback(
    (pcId: string, method: string, depth: number) => {
      setStlDialogTarget(null);
      setGeneratingSTLForId(pcId);
      generateSTL.mutate(
        { pcId, method, depth },
        {
          onSuccess: (stlPc) => {
            addMeshLayer(stlPc, pcId);
            setGeneratingSTLForId(null);
            // Force refetch STL binary data (clears any cached errors from previous attempts)
            queryClient.invalidateQueries({ queryKey: ['stl-data', stlPc.id] });
          },
          onError: () => {
            setGeneratingSTLForId(null);
          },
        },
      );
    },
    [generateSTL, addMeshLayer, queryClient],
  );

  const handleDownloadSTL = useCallback((pcId: string) => {
    const url = pointCloudApi.stlDownloadUrl(pcId);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'implant_mesh.stl';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, []);

  // SDF Heatmap state
  const [heatmapLayerId, setHeatmapLayerId] = useState<string | null>(null);
  const [heatmapReferenceId, setHeatmapReferenceId] = useState<string | null>(null);
  const [colorProfileId, setColorProfileId] = useState<string | null>(null);

  // Fetch SDF distances for heatmap
  const heatmapLayer = layers.find((l) => l.id === heatmapLayerId);
  const heatmapRef = layers.find((l) => l.id === heatmapReferenceId);
  const { data: sdfDistances } = useSDFHeatmap(
    heatmapLayer?.pointCloudId ?? null,
    heatmapRef?.pointCloudId ?? null,
  );
  const { data: colorProfile } = useColorProfile(colorProfileId);

  // Toggle heatmap mode on the target layer when selection changes
  useMemo(() => {
    // Enable heatmap on the selected layer, disable on all others
    for (const layer of layers) {
      if (layer.id === heatmapLayerId && !layer.useHeatmap) {
        setLayerHeatmap(layer.id, true);
      } else if (layer.id !== heatmapLayerId && layer.useHeatmap) {
        setLayerHeatmap(layer.id, false);
      }
    }
  }, [heatmapLayerId, layers, setLayerHeatmap]);

  // Compute heatmap colors from SDF distances + color profile
  const heatmapColors = useMemo(() => {
    const map = new Map<string, Float32Array>();
    if (!sdfDistances || !colorProfile || !heatmapLayerId) return map;
    const colors = computeHeatmapColors(sdfDistances, colorProfile);
    map.set(heatmapLayerId, colors);
    return map;
  }, [sdfDistances, colorProfile, heatmapLayerId]);

  return (
    <>
      <AppLayout
        sidebar={
          <WorkspaceBrowser
            selectedProjectId={selectedProjectId}
            onSelectProject={setSelectedProjectId}
            layers={layers}
            onAddLayer={addLayer}
            onToggleVisibility={toggleVisibility}
            onSetColor={setLayerColor}
            onSetHeatmap={setLayerHeatmap}
            onRemoveLayer={removeLayer}
            onAddFromAutoMatch={addFromAutoMatch}
            onClearLayers={clearLayers}
            onGenerateSTL={handleOpenSTLDialog}
            onDownloadSTL={handleDownloadSTL}
            generatingSTLForId={generatingSTLForId}
          />
        }
        main={
          <MultiPointCloudViewer
            layers={layers}
            pointSize={pointSize}
            showGrid={showGrid}
            showAxes={showAxes}
            heatmapColors={heatmapColors}
            meshData={meshData}
          />
        }
        controls={
          <CheckerControls
            layers={layers}
            pointSize={pointSize}
            onPointSizeChange={setPointSize}
            showGrid={showGrid}
            onShowGridChange={setShowGrid}
            showAxes={showAxes}
            onShowAxesChange={setShowAxes}
            heatmapLayerId={heatmapLayerId}
            heatmapReferenceId={heatmapReferenceId}
            onHeatmapLayerChange={setHeatmapLayerId}
            onHeatmapReferenceChange={setHeatmapReferenceId}
            colorProfileId={colorProfileId}
            onColorProfileChange={setColorProfileId}
          />
        }
      />

      {stlDialogTarget && (
        <STLGenerateDialog
          layerName={stlDialogTarget.layerName}
          pcId={stlDialogTarget.pcId}
          onGenerate={handleConfirmGenerate}
          onClose={() => setStlDialogTarget(null)}
          isGenerating={generatingSTLForId === stlDialogTarget.pcId}
        />
      )}
    </>
  );
}

/**
 * Convert SDF distance values to RGB vertex colors using a color profile.
 * Reuses the same interpolation logic as useSDFColors.
 */
function computeHeatmapColors(
  distances: Float32Array,
  profile: ColorProfile,
): Float32Array {
  const stops = parseColorStops(profile);
  if (stops.length === 0) {
    return new Float32Array(distances.length * 3).fill(1.0);
  }

  const rangeMin = profile.sdf_range_min;
  const rangeMax = profile.sdf_range_max;
  const rangeSpan = rangeMax - rangeMin;

  const rgb = new Float32Array(distances.length * 3);
  for (let i = 0; i < distances.length; i++) {
    const t = rangeSpan > 0 ? (distances[i] - rangeMin) / rangeSpan : 0.5;
    const [r, g, b] = interpolateColor(stops, t);
    rgb[i * 3] = r;
    rgb[i * 3 + 1] = g;
    rgb[i * 3 + 2] = b;
  }

  return rgb;
}
