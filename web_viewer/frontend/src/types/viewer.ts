export type ViewerMode = 'point_cloud' | 'volume' | 'split';

export interface ViewerState {
  mode: ViewerMode;
  selectedScanId: string | null;
  selectedPointCloudId: string | null;
  activeColorProfileId: string | null;
  pointSize: number;
  showGrid: boolean;
  showAxes: boolean;
}

export const DEFAULT_VIEWER_STATE: ViewerState = {
  mode: 'point_cloud',
  selectedScanId: null,
  selectedPointCloudId: null,
  activeColorProfileId: null,
  pointSize: 0.005,
  showGrid: true,
  showAxes: true,
};
