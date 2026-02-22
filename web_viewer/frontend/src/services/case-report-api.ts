/**
 * API client for case report endpoints.
 */
import { apiV1 } from './api-v1';
import type { CaseReport, CaseReportCreate, CaseReportSummary } from '../types/case-report';

export const caseReportApi = {
  async list(params?: { project_id?: string; limit?: number; offset?: number }): Promise<CaseReportSummary[]> {
    const { data } = await apiV1.get<CaseReportSummary[]>('/case-reports/', { params });
    return data;
  },

  async get(id: string): Promise<CaseReport> {
    const { data } = await apiV1.get<CaseReport>(`/case-reports/${id}`);
    return data;
  },

  async generate(body: CaseReportCreate, model?: string): Promise<CaseReport> {
    const params = model ? { model } : undefined;
    const { data } = await apiV1.post<CaseReport>('/case-reports/', body, { params });
    return data;
  },

  async delete(id: string): Promise<void> {
    await apiV1.delete(`/case-reports/${id}`);
  },

  getHtmlUrl(id: string): string {
    return `${apiV1.defaults.baseURL}/case-reports/${id}/html`;
  },

  getPdfUrl(id: string): string {
    return `${apiV1.defaults.baseURL}/case-reports/${id}/pdf`;
  },
};
