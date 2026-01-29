import { useCallback, useState } from 'react';
import type { PointCloud } from '../types/point-cloud';
import type { CheckerLayer, SkullImplantPair } from '../types/checker';
import { DEFAULT_LAYER_COLORS } from '../types/checker';

/**
 * Manages the list of point cloud layers in the Implant Checker viewer.
 */
export function useCheckerLayers() {
  const [layers, setLayers] = useState<CheckerLayer[]>([]);

  const addLayer = useCallback((pc: PointCloud) => {
    setLayers((prev) => {
      // Don't add duplicates
      if (prev.some((l) => l.pointCloudId === pc.id)) return prev;

      const isSTL = pc.file_format === 'stl';

      // Extract source_pc_id from metadata for STL records
      let sourcePcId: string | undefined;
      if (isSTL && pc.metadata_json) {
        try {
          const meta = JSON.parse(pc.metadata_json);
          sourcePcId = meta.source_pc_id;
        } catch { /* ignore */ }
      }

      return [
        ...prev,
        isSTL
          ? {
              id: `mesh-${pc.id}`,
              pointCloudId: pc.id,
              name: pc.name,
              visible: true,
              color: DEFAULT_LAYER_COLORS.stl_mesh ?? '#e879f9',
              opacity: 0.85,
              category: pc.scan_category,
              useHeatmap: false,
              layerType: 'mesh' as const,
              sourcePointCloudId: sourcePcId,
            }
          : {
              id: pc.id,
              pointCloudId: pc.id,
              name: pc.name,
              visible: true,
              color: DEFAULT_LAYER_COLORS[pc.scan_category ?? 'other'] ?? '#f59e0b',
              opacity: 1,
              category: pc.scan_category,
              useHeatmap: false,
              layerType: 'points' as const,
            },
      ];
    });
  }, []);

  const addMeshLayer = useCallback((stlPc: PointCloud, sourcePcId: string) => {
    setLayers((prev) => {
      if (prev.some((l) => l.pointCloudId === stlPc.id)) return prev;
      return [
        ...prev,
        {
          id: `mesh-${stlPc.id}`,
          pointCloudId: stlPc.id,
          name: `${stlPc.name}`,
          visible: true,
          color: DEFAULT_LAYER_COLORS.stl_mesh ?? '#e879f9',
          opacity: 0.85,
          category: stlPc.scan_category,
          useHeatmap: false,
          layerType: 'mesh' as const,
          sourcePointCloudId: sourcePcId,
        },
      ];
    });
  }, []);

  const removeLayer = useCallback((layerId: string) => {
    setLayers((prev) => prev.filter((l) => l.id !== layerId));
  }, []);

  const toggleVisibility = useCallback((layerId: string) => {
    setLayers((prev) =>
      prev.map((l) =>
        l.id === layerId ? { ...l, visible: !l.visible } : l,
      ),
    );
  }, []);

  const setLayerColor = useCallback((layerId: string, color: string) => {
    setLayers((prev) =>
      prev.map((l) => (l.id === layerId ? { ...l, color } : l)),
    );
  }, []);

  const setLayerHeatmap = useCallback((layerId: string, useHeatmap: boolean) => {
    setLayers((prev) =>
      prev.map((l) => (l.id === layerId ? { ...l, useHeatmap } : l)),
    );
  }, []);

  const clearLayers = useCallback(() => {
    setLayers([]);
  }, []);

  const addFromAutoMatch = useCallback((pairs: SkullImplantPair[]) => {
    setLayers((prev) => {
      const existing = new Set(prev.map((l) => l.pointCloudId));
      const newLayers: CheckerLayer[] = [];

      for (const pair of pairs) {
        const skull = pair.defective_skull;
        if (!existing.has(skull.id)) {
          existing.add(skull.id);
          newLayers.push({
            id: skull.id,
            pointCloudId: skull.id,
            name: skull.name,
            visible: true,
            color: DEFAULT_LAYER_COLORS.defective_skull,
            opacity: 1,
            category: skull.scan_category,
            useHeatmap: false,
            layerType: 'points' as const,
          });
        }

        for (const implant of pair.implants) {
          if (!existing.has(implant.id)) {
            existing.add(implant.id);
            newLayers.push({
              id: implant.id,
              pointCloudId: implant.id,
              name: implant.name,
              visible: true,
              color:
                DEFAULT_LAYER_COLORS[implant.scan_category ?? 'other'] ??
                '#f59e0b',
              opacity: 1,
              category: implant.scan_category,
              useHeatmap: false,
              layerType: 'points' as const,
            });
          }
        }
      }

      return [...prev, ...newLayers];
    });
  }, []);

  return {
    layers,
    addLayer,
    addMeshLayer,
    removeLayer,
    toggleVisibility,
    setLayerColor,
    setLayerHeatmap,
    clearLayers,
    addFromAutoMatch,
  };
}
