import { apiV1 } from './api-v1';

export interface FileEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size: number | null;
  extension: string | null;
}

export interface DirectoryListing {
  path: string;
  parent: string | null;
  entries: FileEntry[];
}

export interface CommonPath {
  name: string;
  path: string;
}

export const filesystemApi = {
  async browse(params?: {
    path?: string;
    show_hidden?: boolean;
    filter_extensions?: boolean;
  }): Promise<DirectoryListing> {
    const { data } = await apiV1.get<DirectoryListing>('/filesystem/browse', { params });
    return data;
  },

  async getHome(): Promise<{ path: string }> {
    const { data } = await apiV1.get<{ path: string }>('/filesystem/home');
    return data;
  },

  async getCommonPaths(): Promise<CommonPath[]> {
    const { data } = await apiV1.get<CommonPath[]>('/filesystem/common-paths');
    return data;
  },
};
